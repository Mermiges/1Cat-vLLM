# SPDX-License-Identifier: Apache-2.0
"""Real mtp.0..2, independent FP32 per-layer golden on L28 golden inputs.

This is a component gate, not a target-driven proposal or an acceptance test.
Artifacts include the source golden input, independent output, port output,
and rel-RMS. All checkpoint reads verify the shard first. Run pinned to one
assigned idle V100; pytest cwd /tmp. TP4 / full-server parity stays external.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import set_current_vllm_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.contracts import (
    StagePlan,
    allocate_shared_attn_buffers,
)
from vllm.models.deepseek_v41.common.hc import hc_mixes
from vllm.models.deepseek_v41.common.rope import apply_rope_torch
from vllm.models.deepseek_v41.sm70.dspark import (
    DSparkBlock,
    DSparkMetadata,
)
from vllm.models.deepseek_v41.sm70.dspark import (
    DSparkDeepseekV41ForCausalLM as Draft,
)

from .test_attn_layers import dist_env  # noqa: F401
from .test_core_model import _vllm_config
from .test_moe_realweights import FFNTensors, reference_moe, rms_norm

pytestmark = [pytest.mark.sm70, pytest.mark.weights]
GOLD = Path("/mnt/nvme2/scratch/ds41/golden/p0_smoke__v100-semantic__prefill")
OUT = Path(
    os.environ.get("DS41_DSPARK_GOLDEN_DIR", "/mnt/nvme2/scratch/ds41/dspark/golden")
)


def rel(a, b):
    return float(
        (a.double() - b.double()).square().mean().sqrt()
        / b.double().square().mean().sqrt().clamp_min(1e-20)
    )


def reader():
    # The lane's mandatory reference reader verifies every touched shard.
    sys.path.insert(0, "/mnt/hdd/v100-research/tools/ds41_ref")
    from weights import CheckpointReader

    return CheckpointReader()


def load_block(i, vc):
    from safetensors import safe_open

    rd = reader()
    shell = object.__new__(Draft)
    nn.Module.__init__(shell)
    shell.model = nn.Module()
    stage = StagePlan(0, 1, 40, 42, (), (), ())
    shared = allocate_shared_attn_buffers(256, stage, torch.device("cuda"))
    prev = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("cuda"), set_current_vllm_config(vc):
            block = DSparkBlock(vc, f"model.layers.{i}", 40 + i, stage, shared)
    finally:
        torch.set_default_dtype(prev)
    shell.model.layers = nn.ModuleDict({str(i): block})
    names = [
        n
        for n in rd.weight_map
        if n.startswith(f"mtp.{i}.")
        and (Draft._remap_dspark_name(n) or "").startswith(f"model.layers.{i}.")
    ]

    def weights():
        for name in names:
            info = rd.info(name)  # verifies before opening safetensors
            with safe_open(str(rd.shard_path(info.shard)), framework="pt") as f:
                yield name, f.get_tensor(name)

    shell.load_weights(weights())
    vc.model_config.quantization = "deepseek_v41_fp8"
    process_weights_after_loading(shell, vc.model_config, torch.device("cuda"))
    return shell, block, rd


def ref_mixes(h, rd, prefix, sub):
    from ref_model import hc_split_sinkhorn

    x = h.double().flatten(1)
    fn = rd.get_f32(f"{prefix}.hc_{sub}_fn", "cuda").double()
    scale = rd.get_f32(f"{prefix}.hc_{sub}_scale", "cuda").double()
    base = rd.get_f32(f"{prefix}.hc_{sub}_base", "cuda").double()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    return tuple(t.float() for t in hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6))


def ref_pre(h, p):
    return (h.float() * p[..., None]).sum(1).bfloat16()


def ref_post(out, h, p, comb):
    return (
        p[..., None] * out[:, None].float()
        + torch.einsum("tjc,tjd->tcd", comb, h.float())
    ).bfloat16()


def ref_attention(block, rd, prefix, x, main_x, anchor):
    def w(n):
        return rd.get_f32(f"{prefix}.attn.{n}", "cuda")

    attn = block.attn
    cs = attn.rotary_emb.cos_sin_cache
    pos = torch.arange(anchor, anchor + 5, device="cuda")
    kvmain = rms_norm(main_x.float() @ w("wkv.weight").t(), w("kv_norm.weight"))
    kvmain = qat.fp8_block32_qdq(
        apply_rope_torch(kvmain, torch.arange(anchor, device="cuda"), cs), impl="torch"
    )
    qr = rms_norm(x.float() @ w("wq_a.weight").t(), w("q_norm.weight")).half()
    q = (qr.float() @ w("wq_b.weight").t()).half().view(5, 64, 512)
    q = apply_rope_torch(q, pos, cs)
    kv = rms_norm(x.float() @ w("wkv.weight").t(), w("kv_norm.weight"))
    kv = qat.fp8_block32_qdq(apply_rope_torch(kv, pos, cs), impl="torch")
    keys = torch.cat((kvmain[max(0, anchor - 128) :], kv)).float()
    scores = torch.einsum("thd,kd->thk", q.float(), keys) * 512**-0.5
    mx = scores.amax(-1).clamp_min(-1e30)
    prob = torch.exp(scores - mx[..., None])
    den = prob.sum(-1) + torch.exp(w("attn_sink")[None] - mx)
    o = (torch.einsum("thk,kd->thd", prob, keys) / den[..., None]).half()
    o = apply_rope_torch(o, pos, cs, inverse=True).view(5, 8, 4096)
    z = (
        torch.einsum("tgd,grd->tgr", o.float(), w("wo_a.weight").view(8, 1024, 4096))
        .half()
        .flatten(1)
    )
    return z.float() @ w("wo_b.weight").t()


@pytest.mark.parametrize("i", [0, 1, 2])
@pytest.mark.parametrize("anchor", [7, 130])
@torch.inference_mode()
@pytest.mark.usefixtures("dist_env")
def test_draft_single_layer_golden(i, anchor, ds41_checkpoint_dir):
    from safetensors import safe_open
    from safetensors.torch import save_file

    vc = _vllm_config(ds41_checkpoint_dir, max_tokens=256)
    vc.model_config.max_model_len = 1024
    vc.cache_config.block_size = 64
    vc.scheduler_config.async_scheduling = False
    shell, block, rd = load_block(i, vc)
    with safe_open(str(GOLD / "L28.safetensors"), framework="pt") as f:
        stream = f.get_tensor("stream_in")[:5].cuda().bfloat16()
    pre = torch.zeros((5, 4), device="cuda")
    pre[:, 0] = 1
    # Realistic normed context from the same source golden; explicit component
    # input, not represented as a target-produced main_proj context.
    main_x = rms_norm(stream[0].mean(0), torch.ones(5120, device="cuda"))
    main_x = main_x.half().expand(anchor, -1).contiguous()
    pos = torch.arange(anchor, anchor + 5, device="cuda")
    cache = block.attn.swa_cache
    assert cache.get_kv_cache_spec(vc).sliding_window == 133
    cache.kv_cache = torch.zeros((16, 32, 512), dtype=torch.half, device="cuda")
    block.attn.store_context(
        main_x, torch.arange(anchor, device="cuda"), torch.arange(anchor, device="cuda")
    )
    slots = torch.arange(max(0, anchor - 128), anchor + 5, device="cuda")
    slots = F.pad(slots, (133 - slots.numel(), 0), value=-1).expand(5, -1).contiguous()
    md = DSparkMetadata(
        num_reqs=1,
        num_actual_tokens=5,
        query_start_loc=torch.tensor([0, 5], device="cuda"),
        query_start_loc_cpu=np.array([0, 5]),
        seq_lens_cpu=np.array([anchor + 5]),
        token_to_req_indices=torch.zeros(5, dtype=torch.int32, device="cuda"),
        positions=pos,
        positions_cpu=pos.cpu().numpy(),
        block_table=torch.arange(16, device="cuda")[None],
        block_size=32,
        slot_mapping=pos,
        window_slots=slots,
    )
    prefix = f"mtp.{i}"
    ap, post, comb = ref_mixes(stream, rd, prefix, "attn")
    attn_mixes = hc_mixes(
        stream, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    assert all(rel(a, b) <= 1e-6 for a, b in zip(attn_mixes, (ap, post, comb)))
    x = rms_norm(
        ref_pre(stream, pre), rd.get_f32(f"{prefix}.attn_norm.weight", "cuda")
    ).half()
    aout = ref_attention(block, rd, prefix, x, main_x, anchor)
    h = ref_post(aout, stream, post, comb)
    fp, post, comb = ref_mixes(h, rd, prefix, "ffn")
    ffn_mixes = hc_mixes(h, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    assert all(rel(a, b) <= 1e-6 for a, b in zip(ffn_mixes, (fp, post, comb)))
    xff = rms_norm(
        ref_pre(h, ap), rd.get_f32(f"{prefix}.ffn_norm.weight", "cuda")
    ).half()
    moe = reference_moe(FFNTensors(prefix), xff, 3)
    golden = ref_post(moe["out"], h, post, comb)
    ctx = ForwardContext(
        no_compile_layers=vc.compilation_config.static_forward_context,
        attn_metadata={cache.layer_name: md},
        slot_mapping={cache.layer_name: pos},
    )
    with override_forward_context(ctx):
        actual, actual_pre = block(stream, pre, pos)
        actual_aout = block.attn(pos, x)
    rms = rel(actual, golden)
    attn_rms = rel(actual_aout, aout)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"mtp{i}_anchor{anchor}"
    save_file(
        {
            "stream_in": stream.cpu(),
            "pre_in": pre.cpu(),
            "stream_golden": golden.cpu(),
            "stream_port": actual.cpu(),
            "pre_golden": fp.cpu(),
            "pre_port": actual_pre.cpu(),
        },
        str(OUT / f"{stem}.safetensors"),
    )
    report = {
        "stage": i,
        "anchor": anchor,
        "tp": 1,
        "source_golden": str(GOLD),
        "stream_rel_rms": rms,
        "attention_rel_rms": attn_rms,
        "pre_rel_rms": rel(actual_pre, fp),
    }
    (OUT / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
    assert torch.isfinite(actual).all()
    assert attn_rms <= 2e-3, report
    assert rms <= 5e-3, report
