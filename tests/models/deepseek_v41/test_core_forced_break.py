# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forced eager host breaks in FULL runtime mode (D21)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.compilation import breakable_cudagraph as bc
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def config(arch="DeepseekV41ForCausalLM", ubatching=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[arch]),
        parallel_config=SimpleNamespace(use_ubatching=ubatching),
    )


@pytest.mark.parametrize(
    "force,named,expected", [(False, False, 0), (True, False, 1), (False, True, 1)]
)
def test_full_mode_forced_break(monkeypatch, force, named, expected):
    # Decoration precedes config auto-enable in a real model import.
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    calls = []

    def op():
        calls.append("op")

    decorate = (
        bc.forced_eager_break_during_capture if force else bc.eager_break_during_capture
    )
    wrapped = decorate(op)
    name = f"{op.__module__}.{op.__qualname__}"
    cap = SimpleNamespace(
        _capturing=True,
        forced_eager_breaks={name} if named else set(),
        add_eager=Mock(side_effect=lambda fn: fn()),
    )
    monkeypatch.setattr(bc.BreakableCUDAGraphCapture, "current", lambda: cap)
    monkeypatch.setattr(bc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        bc,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    wrapped()
    assert cap.add_eager.call_count == expected
    assert calls == ["op"]


@pytest.mark.parametrize(
    "enabled,arch,ubatching,allowed",
    [
        (True, "DeepseekV41ForCausalLM", False, True),
        (False, "DeepseekV41ForCausalLM", False, False),
        (True, "DeepseekV4ForCausalLM", False, False),
        (True, "DeepseekV41ForCausalLM", True, False),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_full_engram_requires_forced_break_wrapper(
    monkeypatch, enabled, arch, ubatching, allowed, mode
):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(enabled)))
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = config(arch, ubatching)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=mode)
    runner._engram_bind_service_cache = object()
    if allowed:
        assert bc.ds41_forced_eager_breaks(runner.vllm_config) == {
            bc.DS41_ENGRAM_WAIT,
            bc.DS41_MIRROR_INGEST,
        }
        runner._refuse_full_cudagraphs_with_engram()
    else:
        with pytest.raises(ValueError, match="Engram"):
            runner._refuse_full_cudagraphs_with_engram()


@pytest.mark.sm70
@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_actual_engram_wait_replays_fresh_rows(tmp_path, monkeypatch, mode):
    import numpy as np
    import torch

    from vllm.models.deepseek_v41.common import engram as E

    from .test_engram_module import _fc, bind, make_module, random_stream, synthetic_wkv
    from .test_engram_synth import build_synthetic_engram

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_IMPL", "sm70")
    E._eager_wait.cache_clear()
    syn = build_synthetic_engram(tmp_path / "forced", seed=7)
    mod, svc = make_module(syn, 14, 0, 1, synthetic_wkv(15), max_tokens=4)
    try:
        real = random_stream(1, 5).cuda()
        static = torch.zeros_like(real)
        positions = torch.zeros(1, dtype=torch.long, device="cuda")
        cs = torch.cuda.Stream()
        cs.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(cs):
            with _fc(True):
                # Existing fixture context leaves mode unset; set it for this test.
                bc.get_forward_context().cudagraph_runtime_mode = mode
                mod(static, positions)
                cs.synchronize()
                cap = bc.BreakableCUDAGraphCapture(
                    forced_eager_breaks=bc.ds41_forced_eager_breaks(config())
                )
                with cap:
                    mod(static, positions)
            assert cap.num_eager_breaks == 1 and cap.num_graphs == 2
            previous_rows = None
            for step, token in enumerate((101, 777, 2021, 9001)):
                ids = np.array([token], np.int32)
                bind(svc, 2 * step, ids, 1)
                static.copy_(real)
                with _fc(False):
                    bc.get_forward_context().cudagraph_runtime_mode = mode
                    cap.replay()
                cs.synchronize()
                rows = svc.wait_rows(14).clone()
                if previous_rows is not None:
                    assert not torch.equal(rows, previous_rows)
                previous_rows = rows
                svc.end_step(2 * step)
                bind(svc, 2 * step + 1, ids, 1)
                eager = real.clone()
                with _fc(False):
                    mod(eager, positions)
                cs.synchronize()
                assert not torch.equal(eager, real), "Engram contributed nothing"
                assert torch.equal(static, eager)
                svc.end_step(2 * step + 1)
    finally:
        svc.shutdown()
        E._eager_wait.cache_clear()
