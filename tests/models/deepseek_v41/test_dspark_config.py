# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.platforms import current_platform


def make_spec(monkeypatch, checkpoint: Path, cap: int = 5, block: int = 5):
    monkeypatch.setattr(current_platform, "device_count", lambda: 12)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "14,14,12")
    target = ModelConfig(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        runner="generate",
        dtype=torch.float16,
        max_model_len=2048,
    )
    return SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(
            pipeline_parallel_size=3, tensor_parallel_size=4
        ),
        method="dspark",
        num_speculative_tokens=block,
        dspark_max_verification_tokens=cap,
    )


@pytest.mark.parametrize("cap", range(1, 6))
def test_v41_config_selects_v41_draft(monkeypatch, ds41_checkpoint_dir, cap):
    spec = make_spec(monkeypatch, ds41_checkpoint_dir, cap)
    hf = spec.draft_model_config.hf_config
    assert hf.model_type == "deepseek_v41"
    assert hf.architectures == ["DSparkV41DraftModel"]
    assert hf.dspark_target_layer_ids == [37, 38, 39]
    assert spec.draft_parallel_config.pipeline_parallel_size == 1
    assert spec.draft_parallel_config.tensor_parallel_size == 4
    assert spec.dspark_max_verification_tokens == cap
    assert spec.parallel_drafting


@pytest.mark.parametrize("block", [1, 4, 6, 7])
def test_v41_rejects_wrong_draft_block(monkeypatch, ds41_checkpoint_dir, block):
    with pytest.raises(ValueError, match="num_speculative_tokens=5"):
        make_spec(monkeypatch, ds41_checkpoint_dir, cap=1, block=block)
