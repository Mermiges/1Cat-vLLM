# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mandatory per-layer decode MoE gates fed captured L-REF inputs."""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from vllm.models.deepseek_v41.sm70.moe_decode import decode_front

from .test_moe_realweights import build_port, load_ffn, metrics

pytestmark = [pytest.mark.sm70, pytest.mark.weights]
GOLDEN = Path(os.environ.get("DS41_GOLDEN_DIR", "/mnt/nvme2/scratch/ds41/golden"))


@pytest.mark.parametrize("layer", [3, 20])
def test_fused_decode_per_layer_golden(ds41_dist_single: None, layer: int) -> None:
    tensors = load_ffn(f"layers.{layer}")
    block = build_port(tensors, 384, 6, layer)
    block.decode_front = True
    prompts = ("p0_smoke", "p1_legal", "p2_code", "p3_doc")
    phases = ("prefill", "decode1", "decode2", "decode3", "decode4")
    for prompt, phase in product(prompts, phases):
        path = (
            GOLDEN / f"{prompt}__v100-semantic__{phase}" / f"L{layer:02d}.safetensors"
        )
        if not path.exists():
            raise FileNotFoundError(f"mandatory MoE golden missing: {path}")
        golden = load_file(str(path), device="cpu")
        for row in sorted(
            {0, golden["moe.x"].shape[0] // 2, golden["moe.x"].shape[0] - 1}
        ):
            x = golden["moe.x"][row : row + 1].cuda().half().contiguous()
            w13, _ = block.shared_experts.fp16_weights()
            _, ids, _, _ = decode_front(
                x,
                block.gate.weight,
                block.gate.e_score_correction_bias,
                w13,
                block._decode_scratch,
                top_k=6,
                alpha=2.0**-block.gate.weight_exp,
                scale=1.5,
                limit=10.0,
                phys_map=None,
                n_resident=384,
                spill=False,
            )
            logits = block._decode_scratch.logits[None].clone()
            torch.testing.assert_close(logits, block.gate_logits(x), rtol=0, atol=0)
            reference = golden["moe.out"][row : row + 1].cuda()
            result = block(x)
            report = metrics(result, reference)
            print(prompt, phase, layer, row, report, flush=True)
            assert report["rel_rms"] <= 3e-3, report
            # The activation contract rounds golden FP32 inputs to FP16. Gate
            # logits against a reference fed those same inputs; report the raw
            # golden delta separately so input rounding is visible.
            expected_logits = x.float() @ tensors["ffn.gate.weight"].cuda().float().t()
            assert metrics(logits, expected_logits)["rel_rms"] <= 1e-5
            print(
                "raw golden logits",
                metrics(logits, golden["moe.logits"][row : row + 1].cuda()),
                flush=True,
            )
            assert torch.equal(
                ids.sort(-1)[0].long().cpu(),
                golden["moe.topk_ids"][row : row + 1].sort(-1)[0].long(),
            )
