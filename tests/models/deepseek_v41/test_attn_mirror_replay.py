# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sustained stage-3 mirror/decode replay with runner-bound profiling caches."""
# ruff: noqa: F811

import faulthandler
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
import torch
import torch.multiprocessing as mp

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, StagePlan
from vllm.models.deepseek_v41.kv_mirror import kv20_crc
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .test_attn_harness import ref_config, synthetic_attn_weights, topology
from .test_attn_layers import DEV, Stage, dist_env  # noqa: F401
from .test_attn_stage_graph import _stage


def _bind_profile_cache(st: Stage) -> None:
    """Real local grouping + runner reshape: reproduces F3 before its fix."""
    specs = {
        name: cache._spec for name, cache in st.ctx.items() if hasattr(cache, "_spec")
    }
    st.vcfg.scheduler_config.disable_hybrid_kv_cache_manager = False
    st.vcfg.max_in_flight_tokens = st.vcfg.scheduler_config.max_num_batched_tokens
    st.vcfg.cache_config.num_gpu_blocks_override = None
    runner = SimpleNamespace(
        vllm_config=st.vcfg,
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=st.sim.num_blocks
        ),
        runner_only_attn_layers=set(),
        cache_config=st.vcfg.cache_config,
        device=DEV,
        get_kv_cache_spec=lambda: specs,
    )

    def initialize(config: KVCacheConfig, *, is_profiling: bool) -> None:
        assert is_profiling
        runner.kv_cache_config = config
        attn_groups = []
        for gid, group in enumerate(config.kv_cache_groups):
            for name in group.layer_names:
                attn_groups.append(
                    SimpleNamespace(
                        kv_cache_spec=group.kv_cache_spec.kv_cache_specs[name],
                        backend=st.ctx[name].get_attn_backend(),
                        kv_cache_group_id=gid,
                        layer_names=[name],
                    )
                )
        runner._kv_cache_spec_attn_group_iterator = lambda: iter(attn_groups)
        raw = GPUModelRunner._allocate_kv_cache_tensors(runner, config)
        blocks = [g.kv_cache_spec.block_size for g in config.kv_cache_groups]
        views = GPUModelRunner._reshape_kv_cache_tensors(runner, raw, blocks)
        for name, view in views.items():
            st.ctx[name].kv_cache = view
            rows = st.ctx[name].rows()  # exact F3 rejection before the fix
            assert rows.data_ptr() == view.data_ptr() == raw[name].data_ptr()
            assert specs[name].page_size_padded is None
            view.copy_(st.sim.caches[name].tensor)
            st.sim.caches[name].tensor = view

    runner.initialize_kv_cache = initialize
    GPUModelRunner._init_minimal_kv_cache_for_profiling(runner)


@pytest.mark.sm70
@pytest.mark.parametrize("replay", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("check", [False, True], ids=["crc0", "crc1"])
def test_attn_stage3_sustained_mirror_decode(
    dist_env: None, monkeypatch: pytest.MonkeyPatch, replay: bool, check: bool
) -> None:
    """192 steps cross page boundaries, vs eager and exact mirror bytes.

    Graph capture starts with dummy/null slots, as runner profiling does, then
    replays with live slots and changing payload bytes/CRC. A used eager decode;
    both modes are required here. TP collectives are covered separately.
    """
    st = _stage(monkeypatch, 28, 28, 1)
    st.mirror.check = check
    _exercise(st, replay, check)


def _exercise(
    st: Stage,
    replay: bool,
    check: bool,
    *,
    collective: bool = False,
    receive: bool = False,
) -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.distributed.parallel_state import graph_capture
    from vllm.models.deepseek_v41.common import dump as ds41_dump
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41ForCausalLM

    _bind_profile_cache(st)
    pos = torch.zeros(1, dtype=torch.int64, device=DEV)
    x = torch.randn(1, 5120, device=DEV).half()
    ckv = torch.randn(1, 512, device=DEV).half()
    ik = torch.randn(1, 128, device=DEV).half()
    cand = torch.full((1, 2048), -1, dtype=torch.int32, device=DEV)
    cand[:, 0] = CAND_ALL
    crc = kv20_crc(ckv, ik)
    md, _ = st.sim.metadata([("a", 0, 1)])
    for m in md.values():
        m.slot_mapping.fill_(-1)
        if hasattr(m, "latent_slots"):
            m.latent_slots.fill_(-1)
        if hasattr(m, "window_slots"):
            m.window_slots.fill_(-1)
        if hasattr(m, "num_visible"):
            m.num_visible.zero_()
    fc = ForwardContext(
        no_compile_layers=st.ctx, attn_metadata=md, slot_mapping={}, is_dummy_run=True
    )

    def step() -> torch.Tensor:
        with override_forward_context(fc):
            st.mirror.ingest(pos, ckv, ik, cand, crc if check else None)
            return st.attn[28](pos, x)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if replay:
        # Register graph addresses with the actual TP device communicator.
        comm_capture = (
            graph_capture(torch.device("cuda")) if collective else nullcontext()
        )
        with comm_capture, torch.cuda.graph(graph):
            captured = step()
    head = (
        SimpleNamespace(
            lm_head=SimpleNamespace(
                weight=torch.randn(
                    129280 // get_tensor_model_parallel_world_size(), 5120, device=DEV
                ).half()
            ),
            config=SimpleNamespace(vocab_size=129280),
        )
        if collective
        else None
    )
    fc.is_dummy_run = False
    for context in range(192):
        fc.attn_metadata, real_pos = st.sim.metadata([("a", context, 1)])
        pos.copy_(real_pos)
        x.normal_()
        ckv.normal_()
        ik.normal_()
        crc.copy_(kv20_crc(ckv, ik))
        if receive:
            from vllm.distributed import get_pp_group

            for handle in get_pp_group().irecv_tensor_dict_static(
                dict(positions=pos, x=x, ckv=ckv, ik=ik, cand=cand, crc=crc)
            ):
                handle.wait()
        if replay:
            graph.replay()
            got = captured.clone()
        else:
            got = step().clone()
        if head is not None:
            # Keep the production boundary order: only TP rank 0's dump
            # synchronizes before the head/gather. Verify on every rank below.
            ds41_dump.begin_step(pos, 1)
            ds41_dump.dump(None, "final.h", got.half())
            ds41_dump.end_forward()
            logits = DeepseekV41ForCausalLM.compute_logits(head, got.half())
            assert logits.shape == (1, 129280) and torch.isfinite(logits).all()
        torch.cuda.synchronize()
        assert torch.equal(got, step()), context
        assert torch.isfinite(got).all(), context
        for cache, payload in ((st.mirror.ckv_cache, ckv), (st.mirror.ik_cache, ik)):
            m = fc.attn_metadata[cache.layer_name]
            assert torch.equal(cache.rows()[m.latent_slots[0]], payload[0])
        st.mirror.check_pending_errors()
        if context % 16 == 0:
            print(f"stage3 crc={check} graph={replay} step={context}", flush=True)
    if check:
        # Prove replay reads the current CRC buffer and reports corruption at
        # the next host boundary, rather than passing with the check inactive.
        crc.add_(1)
        if replay:
            graph.replay()
        else:
            step()
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="CRC/slot errors"):
            st.mirror.check_pending_errors()


def _tp_rank(
    rank: int, rendezvous: str, replay: bool, world: int = 2, pipeline: int = 1
) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_pp_group,
        get_tensor_model_parallel_rank,
        init_distributed_environment,
        initialize_model_parallel,
    )

    trace_path = Path(unquote(urlparse(rendezvous).path)).with_name(
        f"tp{world}-rank{rank}.stack"
    )
    trace = trace_path.open("w")
    faulthandler.dump_traceback_later(90, repeat=True, file=trace)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("VLLM_DS41_ATTN_DECODE_PATH", "1")
    monkeypatch.setenv("VLLM_DS41_ATTN_MIRROR_CHECK", "1")
    tp = world // pipeline
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp, pipeline_parallel_size=pipeline
        )
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world, rank, rendezvous, rank, backend="nccl", timeout=timedelta(seconds=90)
        )
        initialize_model_parallel(tp, pipeline)
        if pipeline > 1 and get_pp_group().is_first_rank:
            with torch.inference_mode():
                _produce_mirror_payload()
            cleanup_dist_env_and_memory()
            faulthandler.cancel_dump_traceback_later()
            trace.close()
            monkeypatch.undo()
            return
        cfg = ref_config()
        weights = {28: synthetic_attn_weights(cfg, topology(cfg, 28), DEV, seed=28)}
        st = Stage(
            cfg,
            (28,),
            weights,
            stage=StagePlan(2, 3, 28, 39, (20,), (), ()),
            mirror=True,
            max_model_len=512,
            max_tokens=4,
            num_blocks=32,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=tp,
            max_seqs=1,
        )
        for cache in st.sim.caches.values():
            cache.tensor.uniform_(-0.2, 0.2)
        torch.manual_seed(42)  # replicated live inputs/payloads on both ranks
        with torch.inference_mode():
            _exercise(st, replay, True, collective=True, receive=pipeline > 1)
        cleanup_dist_env_and_memory()
    monkeypatch.undo()
    faulthandler.cancel_dump_traceback_later()
    trace.close()


def _produce_mirror_payload() -> None:
    """Minimal earlier PP stage: CRC-verified exact records, no model weights."""
    from vllm.distributed import get_pp_group

    torch.manual_seed(42)
    payload = dict(
        positions=torch.zeros(1, dtype=torch.int64, device=DEV),
        x=torch.empty(1, 5120, dtype=torch.float16, device=DEV),
        ckv=torch.empty(1, 512, dtype=torch.float16, device=DEV),
        ik=torch.empty(1, 128, dtype=torch.float16, device=DEV),
        cand=torch.full((1, 2048), -1, dtype=torch.int32, device=DEV),
        crc=torch.empty(1, dtype=torch.int32, device=DEV),
    )
    payload["cand"][:, 0] = CAND_ALL
    for context in range(192):
        payload["positions"].fill_(context)
        for name in ("x", "ckv", "ik"):
            payload[name].normal_()
        payload["crc"].copy_(kv20_crc(payload["ckv"], payload["ik"]))
        for handle in get_pp_group().isend_tensor_dict_static(payload):
            handle.wait()
    torch.cuda.synchronize()


@pytest.mark.sm70
@pytest.mark.parametrize("replay", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("pipeline", [1, 2], ids=["tp2", "pp2tp1"])
def test_attn_two_card_stage3_crc_head_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay: bool, pipeline: int
) -> None:
    assert torch.cuda.device_count() >= 2, "requires both assigned lane GPUs"
    monkeypatch.setenv("VLLM_DS41_DUMP_DIR", str(tmp_path / "dump"))
    monkeypatch.setenv("VLLM_DS41_DUMP_STEPS", "0-191")
    mp.spawn(
        _tp_rank,
        args=((tmp_path / "two-card").as_uri(), replay, 2, pipeline),
        nprocs=2,
        join=True,
    )
