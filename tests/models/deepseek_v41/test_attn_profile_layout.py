# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-bound gate for the stage-local graph-profile cache layout (F3)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.sm70 import sparse as s70
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .test_attn_kv_specs import _cfg, _v41_specs


@pytest.mark.parametrize("bounds", [(0, 14), (14, 28), (28, 40)])
def test_attn_runner_graph_profile_layout(bounds):
    """Use the real profiling initializer, allocator and reshape, with CUDA hidden.

    Only model construction/attention-group discovery are replaced: real specs,
    grouping, tensor planning and all storage operations are production code.
    Stage 3 has no ratio-2 source; inherited V4 grouping pads its 128-KiB SWA
    pages to 256 KiB, producing the exact layout that default decode refuses.
    """
    cfg = _cfg()
    cfg.cache_config.cache_dtype = "auto"
    specs = _v41_specs(range(*bounds), mirror=bounds[0] == 28)
    layers = {}
    for name, spec in specs.items():
        cache = s70.DS41CacheLayer.__new__(s70.DS41CacheLayer)
        torch.nn.Module.__init__(cache)
        cache.layer_name, cache._spec = name, spec
        layers[name] = cache

    runner = SimpleNamespace(
        vllm_config=cfg,
        cache_config=cfg.cache_config,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=8),
        device=torch.device("cpu"),
        runner_only_attn_layers=set(),
        get_kv_cache_spec=lambda: specs,
    )

    def initialize(config, *, is_profiling):
        assert is_profiling
        runner.kv_cache_config = config
        groups = []
        for gid, group in enumerate(config.kv_cache_groups):
            for name in group.layer_names:
                spec = group.kv_cache_spec.kv_cache_specs[name]
                backend = (
                    s70.DS41StateBackend
                    if spec.dtype == torch.float32
                    else s70.DS41SWABackend
                    if isinstance(spec, SlidingWindowMLASpec)
                    else s70.DS41CompressedBackend
                )
                groups.append(
                    SimpleNamespace(
                        kv_cache_spec=spec,
                        backend=backend,
                        kv_cache_group_id=gid,
                        layer_names=[name],
                    )
                )
        runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
        raw = GPUModelRunner._allocate_kv_cache_tensors(runner, config)
        blocks = [g.kv_cache_spec.block_size for g in config.kv_cache_groups]
        views = GPUModelRunner._reshape_kv_cache_tensors(runner, raw, blocks)
        for name, view in views.items():
            # Check the accessor before the layout assertion: goes red with the
            # production F3 message, rather than merely inspecting a config.
            layers[name].kv_cache = view
            rows = layers[name].rows()
            assert rows.data_ptr() == view.data_ptr() == raw[name].data_ptr()
            assert rows.shape == (
                8 * specs[name].storage_block_size,
                specs[name].head_size,
            )
            assert specs[name].page_size_padded is None
            rows[-1].fill_(3)
            assert torch.equal(view[-1, -1], rows[-1])

    runner.initialize_kv_cache = initialize
    GPUModelRunner._init_minimal_kv_cache_for_profiling(runner)
    assert runner.cache_config.num_gpu_blocks == 8
    assert runner.cache_config.num_gpu_blocks_override is None
