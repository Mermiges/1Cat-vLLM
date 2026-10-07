# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-layer core subset: real fused HC, eager attention/Engram, replay parity."""

from types import SimpleNamespace

import pytest
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v41.common import contracts as C
from vllm.models.deepseek_v41.common.engram import _eager_wait
from vllm.sequence import IntermediateTensors

from .test_core_hc_kernels import _fake_model, _inputs

pytestmark = pytest.mark.sm70


def subset(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    _eager_wait.cache_clear()
    run = _fake_model(pp_is_last=False, fused=True)
    model = run.args[0]
    counts = {"attention": 0, "rows": 0}
    rows = torch.zeros(1, 6, 264, device="cuda", dtype=torch.uint8)
    current = [0]

    def dummy_rows(lid, n):
        rows.zero_()
        return rows

    def wait_rows(lid):
        assert not torch.cuda.is_current_stream_capturing()
        counts["rows"] += 1
        rows.fill_(current[0])
        return rows

    svc = SimpleNamespace(dummy_rows=dummy_rows, wait_rows=wait_rows)
    for layer in model.layers:
        original = layer.attn

        def attention(positions, x, fn=original):
            out = torch.empty_like(x, dtype=torch.float32)

            @eager_break_during_capture
            def op(positions, x, out):
                assert not torch.cuda.is_current_stream_capturing()
                counts["attention"] += 1
                out.copy_(fn(positions, x))

            op(positions, x, out)
            return out

        layer.attn = attention

    def engram(stream, positions):
        _eager_wait()(svc, 1, rows)
        # Static row buffer is consumed by a captured kernel after the eager wait.
        return (stream.float() + rows.float().mean() * 0.125).to(torch.bfloat16)

    model.layers[1].engram = engram
    stream, _, _, _, pre = _inputs(1, seed=21)
    payload = IntermediateTensors({C.PP_KEY_HIDDEN: stream, C.PP_KEY_PRE_MIX: pre})
    positions = torch.zeros(1, device="cuda", dtype=torch.long)
    return lambda: run(None, positions, payload), current, counts


def test_subset_replay_eager_breaks_and_fresh_engram_rows(monkeypatch):
    run, current, counts = subset(monkeypatch)
    vc = VllmConfig()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        with set_forward_context(
            None,
            vc,
            num_tokens=1,
            is_dummy_run=True,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ):
            for _ in range(3):
                run()
            stream.synchronize()
            cap = BreakableCUDAGraphCapture()
            with cap:
                captured = run()
        assert cap.num_eager_breaks == 5  # 4 attention operations + 1 row wait
        assert cap.num_graphs == 6
        previous = None
        for value in (1, 7, 11, 19):
            current[0] = value
            before = dict(counts)
            with set_forward_context(
                None, vc, num_tokens=1, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
            ):
                cap.replay()
                replay = {k: v.clone() for k, v in captured.tensors.items()}
                eager = run()
                stream.synchronize()
            assert counts["rows"] - before["rows"] == 2
            assert counts["attention"] - before["attention"] == 8
            for key in (C.PP_KEY_HIDDEN, C.PP_KEY_PRE_MIX):
                assert torch.equal(replay[key], eager.tensors[key]), key
            if previous is not None:
                assert not torch.equal(previous, replay[C.PP_KEY_HIDDEN])
            previous = replay[C.PP_KEY_HIDDEN]
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.weights
@pytest.mark.parametrize("layer", [0, 1, 2, 3, 14, 20, 21, 24, 28])
@pytest.mark.parametrize(
    "case", ["p1_legal__v100-semantic__decode1", "p0_smoke__v100-semantic__decode2"]
)
def test_graph_replay_passes_per_layer_hc_goldens(
    ds41_checkpoint_dir, verified_shard, monkeypatch, layer, case
):
    import json

    from torch.utils._pytree import tree_flatten

    from . import test_core_hc_kernels as hc

    index = json.loads(
        (ds41_checkpoint_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    shards = {
        shard
        for name, shard in index.items()
        if name.startswith(f"layers.{layer}.hc_")
        or name
        in (f"layers.{layer}.attn_norm.weight", f"layers.{layer}.ffn_norm.weight")
    }
    for shard in sorted(shards):
        verified_shard(shard)
    step = hc._step

    def captured_step(*args, **kwargs):
        cs = torch.cuda.Stream()
        cs.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(cs):
            eager = step(*args, **kwargs)
            for _ in range(2):
                step(*args, **kwargs)
            cs.synchronize()
            cap = BreakableCUDAGraphCapture()
            with cap:
                out = step(*args, **kwargs)
            cap.replay()
            cs.synchronize()
            for a, b in zip(tree_flatten(out)[0], tree_flatten(eager)[0]):
                if isinstance(a, torch.Tensor):
                    assert torch.equal(a, b)
        torch.cuda.current_stream().wait_stream(cs)
        return out

    monkeypatch.setattr(hc, "_step", captured_step)
    hc.test_golden_hc_steps(ds41_checkpoint_dir, case, layer)
