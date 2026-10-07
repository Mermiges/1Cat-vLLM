# SPDX-License-Identifier: Apache-2.0
"""TP1 real-weight full-draft smoke and latency, never target/spec tok/s."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch
from safetensors import safe_open

from vllm.config import set_current_vllm_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.deepseek_v41.sm70.dspark import DSparkMetadata

from .test_attn_layers import dist_env  # noqa: F401
from .test_core_model import _vllm_config
from .test_dspark_golden import GOLD, OUT, Draft, reader

pytestmark = [pytest.mark.sm70, pytest.mark.weights]


@torch.inference_mode()
@pytest.mark.usefixtures("dist_env")
def test_full_real_draft_load_and_latency(ds41_checkpoint_dir):
    vc = _vllm_config(ds41_checkpoint_dir, max_tokens=256)
    vc.scheduler_config.async_scheduling = False
    vc.model_config.max_model_len = 1024
    vc.cache_config.block_size = 64
    rd = reader()
    torch.cuda.reset_peak_memory_stats()
    prev = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("cuda"), set_current_vllm_config(vc):
            model = Draft(vllm_config=vc)
    finally:
        torch.set_default_dtype(prev)

    def weights():
        for name in rd.weight_map:
            if model.skip_checkpoint_weight(name):
                continue
            info = rd.info(name)
            with safe_open(str(rd.shard_path(info.shard)), framework="pt") as f:
                yield name, f.get_tensor(name)

    loaded = model.load_weights(weights())
    assert loaded == set(dict(model.named_parameters()))
    vc.model_config.quantization = "deepseek_v41_fp8"
    process_weights_after_loading(model, vc.model_config, torch.device("cuda"))
    with safe_open(str(GOLD / "L28.safetensors"), framework="pt") as f:
        h = f.get_tensor("stream_in")[:5].cuda().bfloat16()
    aux = h.mean(1).float().repeat(1, 3)
    context = model.combine_hidden_states(aux)
    pos = torch.arange(130, 135, device="cuda")
    ids = torch.tensor([0, 128799, 128799, 128799, 128799], device="cuda")
    ctxpos = torch.arange(130, device="cuda")
    slots = torch.arange(2, 135, device="cuda").expand(5, -1).contiguous()
    metadata, mappings = {}, {}
    for layer in model.model.layers:
        cache = layer.attn.swa_cache
        cache.kv_cache = torch.zeros((16, 32, 512), dtype=torch.half, device="cuda")
        metadata[cache.layer_name] = DSparkMetadata(
            num_reqs=1,
            num_actual_tokens=5,
            query_start_loc=torch.tensor([0, 5], device="cuda"),
            query_start_loc_cpu=np.array([0, 5]),
            seq_lens_cpu=np.array([135]),
            token_to_req_indices=torch.zeros(5, dtype=torch.int32, device="cuda"),
            positions=pos,
            positions_cpu=pos.cpu().numpy(),
            block_table=torch.arange(16, device="cuda")[None],
            block_size=32,
            slot_mapping=pos,
            window_slots=slots,
        )
        mappings[cache.layer_name] = ctxpos
    model.precompute_and_store_context_kv(
        context[:1].expand(130, -1).contiguous(), ctxpos, mappings
    )
    ctx = ForwardContext(
        no_compile_layers=vc.compilation_config.static_forward_context,
        attn_metadata=metadata,
        slot_mapping={name: pos for name in metadata},
    )

    def propose() -> torch.Tensor:
        context = model.combine_hidden_states(aux)
        model.precompute_and_store_context_kv(
            context,
            torch.arange(125, 130, device="cuda"),
            {name: torch.arange(125, 130, device="cuda") for name in metadata},
        )
        hidden = model(ids, pos)
        logits = model.compute_logits(hidden)
        assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
        prev_id = ids[:1]
        result = []
        for index in range(5):
            prev_id = (
                logits[index : index + 1]
                + model.markov_bias(model.markov_embed(prev_id))
            ).argmax(-1)
            result.append(prev_id)
        return torch.cat(result)

    with override_forward_context(ctx):
        expected = propose()
        for _ in range(3):
            assert torch.equal(propose(), expected)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            got = propose()
        torch.cuda.synchronize()
        round_ms = (time.perf_counter() - start) * 100
    assert torch.equal(got, expected)
    # Profile path must run projections without a bound metadata/cache lookup.
    model.precompute_and_store_context_kv(context, pos)
    with override_forward_context(
        ForwardContext(
            no_compile_layers=vc.compilation_config.static_forward_context,
            attn_metadata=None,
            slot_mapping={},
            is_dummy_run=True,
        )
    ):
        assert torch.isfinite(model(ids, pos)).all()
    report = {
        "tp": 1,
        "draft_block": 5,
        "loaded_parameters": len(loaded),
        "draft_round_ms": round_ms,
        "includes_context_tokens": 5,
        "includes_full_vocab_markov": True,
        "draft_ids": expected.tolist(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope": "component smoke with synthetic target context; no server metrics",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full-draft-tp1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
