# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core graph dispatch and precision policy (no GPU required)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.config.vllm import _disable_ds41_compile
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker import gpu_model_runner as gm


@pytest.mark.parametrize("mode", list(CompilationMode))
@pytest.mark.parametrize("breakable", [False, True])
def test_ds41_compile_policy(monkeypatch, mode, breakable):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(breakable)))
    cfg = CompilationConfig(mode=mode)
    model = SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
    assert _disable_ds41_compile(model, cfg)
    assert cfg.mode == CompilationMode.NONE
    other = CompilationConfig(mode=mode)
    assert not _disable_ds41_compile(
        SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]), other
    )
    assert other.mode == mode


def _backend(module, support):
    builder = SimpleNamespace(get_cudagraph_support=lambda *args: support)
    return type(
        "TestBackend",
        (),
        {
            "__module__": module,
            "get_builder_cls": staticmethod(lambda: builder),
        },
    )


@pytest.mark.parametrize(
    "enabled,arch,module,expected",
    [
        (True, "DeepseekV41ForCausalLM", "vllm.models.deepseek_v41.sm70.sparse", True),
        (
            False,
            "DeepseekV41ForCausalLM",
            "vllm.models.deepseek_v41.sm70.sparse",
            False,
        ),
        (True, "DeepseekV4ForCausalLM", "vllm.models.deepseek_v41.sm70.sparse", False),
        (True, "DeepseekV41ForCausalLM", "another.backend", False),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_never_attention_is_only_piecewise_compatible(
    monkeypatch, enabled, arch, module, expected, mode
):
    monkeypatch.setattr(gm, "is_breakable_cudagraph_enabled", lambda: enabled)
    cfg = SimpleNamespace(cudagraph_mode=mode)
    cfg.resolve_cudagraph_mode_and_sizes = Mock(
        side_effect=lambda *args, **kwargs: cfg.cudagraph_mode
    )
    runner = SimpleNamespace(
        vllm_config=object(),
        compilation_config=cfg,
        model_config=SimpleNamespace(architectures=[arch]),
        uniform_decode_query_len=1,
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        kv_cache_config=object(),
        max_num_reqs=4,
        speculative_config=None,
        cudagraph_dispatcher=SimpleNamespace(initialize_cudagraph_keys=Mock()),
    )
    gm.GPUModelRunner._check_and_update_cudagraph_mode(
        runner,
        [{_backend(module, AttentionCGSupport.NEVER)}],
        [SimpleNamespace(kv_cache_spec=object())],
    )
    support = cfg.resolve_cudagraph_mode_and_sizes.call_args.args[0]
    assert support == (
        AttentionCGSupport.ALWAYS if expected else AttentionCGSupport.NEVER
    )
    assert cfg.cudagraph_mode == (
        CUDAGraphMode.PIECEWISE if expected and mode != CUDAGraphMode.NONE else mode
    )
    runner.cudagraph_dispatcher.initialize_cudagraph_keys.assert_called_once_with(
        cfg.cudagraph_mode, 1
    )


def test_decorator_import_before_auto_enable(monkeypatch):
    from vllm.compilation import breakable_cudagraph as bc

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    calls = []

    @bc.eager_break_during_capture
    def op():
        calls.append(1)

    cap = SimpleNamespace(_capturing=True, add_eager=Mock(side_effect=lambda fn: fn()))
    monkeypatch.setattr(bc.BreakableCUDAGraphCapture, "current", lambda: cap)
    monkeypatch.setattr(bc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        bc,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE),
    )
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    op()
    assert calls == [1]
    cap.add_eager.assert_called_once()


@pytest.mark.sm70
@pytest.mark.parametrize("enabled", [False, True])
def test_full_config_pins_compile_off(ds41_checkpoint_dir, monkeypatch, enabled):
    from vllm.config import ModelConfig, VllmConfig

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(enabled)))
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(ds41_checkpoint_dir),
            dtype="half",
            skip_tokenizer_init=True,
            max_model_len=32768,
        ),
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    assert cfg.compilation_config.mode == CompilationMode.NONE
