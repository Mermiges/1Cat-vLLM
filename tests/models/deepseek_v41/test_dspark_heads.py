# SPDX-License-Identifier: Apache-2.0
"""Real-weight DSpark main projection, Markov and confidence component gates."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from torch import nn

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.deepseek_v41.sm70.dspark import DSparkModel
from vllm.models.deepseek_v41.sm70.model import DeepseekV41Norm

from .test_attn_layers import dist_env  # noqa: F401
from .test_core_model import _vllm_config
from .test_dspark_golden import GOLD, OUT, Draft, reader, rel
from .test_moe_realweights import rms_norm

pytestmark = [pytest.mark.sm70, pytest.mark.weights]


@torch.inference_mode()
@pytest.mark.usefixtures("dist_env")
def test_real_draft_projection_and_heads(ds41_checkpoint_dir):
    vc = _vllm_config(ds41_checkpoint_dir)
    shell = object.__new__(Draft)
    nn.Module.__init__(shell)
    model = object.__new__(DSparkModel)
    nn.Module.__init__(model)
    shell.model = model
    with torch.device("cuda"), set_current_vllm_config(vc):
        model.main_proj = ReplicatedLinear(
            15360,
            5120,
            bias=False,
            return_bias=False,
            quant_config=vc.quant_config,
            prefix="model.main_proj",
        )
        model.main_norm = DeepseekV41Norm(5120)
        model.norm = DeepseekV41Norm(5120)
        model.markov_embed = nn.Embedding(129280, 256, dtype=torch.half)
        model.markov_weight = nn.Parameter(torch.empty(129280, 256, dtype=torch.half))
        model.confidence_weight = nn.Parameter(
            torch.empty(1, 5376, dtype=torch.float32)
        )
    rd = reader()
    names = [
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj.scale",
        "mtp.0.main_norm.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.embed.weight",
        "mtp.2.markov_head.head.weight",
        "mtp.2.confidence_head.proj.weight",
    ]

    def weights():
        for name in names:
            info = rd.info(name)
            with safe_open(str(rd.shard_path(info.shard)), framework="pt") as f:
                yield name, f.get_tensor(name)

    shell.load_weights(weights())
    vc.model_config.quantization = "deepseek_v41_fp8"
    process_weights_after_loading(shell, vc.model_config, torch.device("cuda"))
    with safe_open(str(GOLD / "L28.safetensors"), framework="pt") as f:
        h = f.get_tensor("stream_in")[:5].cuda().bfloat16()
    aux = h.mean(1).float().repeat(1, 3)
    assert aux.abs().max() > 65504  # exercises the overflow protection
    actual = shell.combine_hidden_states(aux)
    projection = aux @ rd.get_f32("mtp.0.main_proj.weight", "cuda").t()
    expected = rms_norm(projection, rd.get_f32("mtp.0.main_norm.weight", "cuda")).half()
    main_rms = rel(actual, expected)
    assert torch.isfinite(actual).all()
    assert main_rms <= 1e-3, main_rms
    with pytest.raises(TypeError, match="preserve BF16 range"):
        shell.combine_hidden_states(aux.half())
    ids = torch.tensor([0, 128799, 129279], device="cuda")
    embeds = shell.markov_embed(ids)
    ref_embeds = rd.get_f32("mtp.2.markov_head.embed.weight", "cuda")[ids]
    assert torch.equal(embeds.float(), ref_embeds)
    markov = shell.markov_bias(embeds)
    ref_markov = ref_embeds @ rd.get_f32("mtp.2.markov_head.head.weight", "cuda").t()
    markov_rms = rel(markov, ref_markov)
    assert markov.dtype == torch.float32 and markov_rms <= 1e-5
    assert torch.equal(markov.argmax(-1), ref_markov.argmax(-1))
    hidden = h[:3].float().mean(1).bfloat16()
    conf = shell.confidence_logits(hidden, embeds)
    ref_conf = (
        torch.cat((hidden.float(), ref_embeds), dim=-1)
        @ rd.get_f32("mtp.2.confidence_head.proj.weight", "cuda").t()
    )
    confidence_rms = rel(conf, ref_conf.flatten())
    assert conf.dtype == torch.float32 and confidence_rms <= 1e-6
    report = {
        "main_rel_rms": main_rms,
        "markov_rel_rms": markov_rms,
        "confidence_rel_rms": confidence_rms,
        "max_aux": float(aux.abs().max()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heads.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


@torch.inference_mode()
def test_capture_materializes_pending_golden_post():
    from vllm.models.deepseek_v41.sm70.dspark import capture_target_input
    from vllm.models.deepseek_v41.sm70.hc_kernels import hc_fused_step

    from .test_core_hc_kernels import _assert_rounding_of

    with safe_open(str(GOLD / "L28.safetensors"), framework="pt") as f:
        h = f.get_tensor("stream_attn")[:5].cuda().bfloat16()
        sub = f.get_tensor("moe.out")[:5].cuda().float()
        post = f.get_tensor("hc.ffn_post")[:5].cuda().float()
        comb = f.get_tensor("hc.ffn_comb")[:5].cuda().float()
        pre = f.get_tensor("pre_out")[:5].cuda().float()
    truth = post.double()[..., None] * sub.double()[:, None] + torch.einsum(
        "tjc,tjd->tcd", comb.double(), h.double()
    )
    materialized, aux = capture_target_input(h, (sub, post, comb))
    _assert_rounding_of(materialized, truth)
    assert torch.equal(aux, materialized.mean(1).float())
    norm = torch.ones(5120, device="cuda")
    original = hc_fused_step(
        h, sub_out=sub, post=post, comb=comb, collapse_pre=pre, norm_weight=norm
    )[2]
    captured = hc_fused_step(materialized, collapse_pre=pre, norm_weight=norm)[2]
    assert torch.equal(original, captured)
