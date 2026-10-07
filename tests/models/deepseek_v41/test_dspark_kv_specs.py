# SPDX-License-Identifier: Apache-2.0
"""PP3 draft registration must survive the global worker KV-spec merge."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v41.sm70 import dspark as d
from vllm.models.deepseek_v41.sm70 import sparse as s
from vllm.v1.core import kv_cache_utils as ku

from .test_attn_kv_specs import _cfg, _v41_specs


@pytest.mark.parametrize("prefix", ["model", "draft.model"])
def test_pp3_stage3_draft_cache_names_and_groups(monkeypatch, ds41_text_config, prefix):
    cfg = _cfg(max_len=262144)
    cfg.model_config.original_max_model_len = 262144
    cfg.model_config.hf_config = ds41_text_config
    cfg.scheduler_config.async_scheduling = False
    cfg.cache_config.block_size = 64
    cfg.quant_config = None
    cfg.compilation_config = SimpleNamespace(static_forward_context={})

    # Omit weight/projection allocations, but execute the real draft model's
    # prefix wiring, DSparkAttention's window update and cache registration.
    def attention_init(self, vc, name, topo, stage, shared):
        nn.Module.__init__(self)
        self.swa_cache = s.DS41CacheLayer(
            f"{name}.swa_cache", s.swa_cache_spec(64), s.DS41SWABackend, vc
        )
        self.layer_name = f"{name}.ds41_attention"
        vc.compilation_config.static_forward_context[self.layer_name] = self

    class Block(nn.Module):
        def __init__(self, vc, name, layer_id, stage, shared):
            super().__init__()
            self.attn = d.DSparkAttention(vc, f"{name}.attn", stage, shared, layer_id)

    monkeypatch.setattr(d.DeepseekV41Attention, "__init__", attention_init)
    monkeypatch.setattr(d, "DSparkBlock", Block)
    with torch.device("meta"):
        # Replicated/TP layers are irrelevant to cache registration.
        monkeypatch.setattr(d, "VocabParallelEmbedding", lambda *a, **kw: nn.Identity())
        monkeypatch.setattr(d, "ReplicatedLinear", lambda *a, **kw: nn.Identity())
        model = d.DSparkModel(cfg, prefix)
    draft = {
        layer.attn.swa_cache.layer_name: layer.attn.swa_cache.get_kv_cache_spec(cfg)
        for layer in model.layers
    }
    workers = [
        _v41_specs(range(0, 14), block=64),
        _v41_specs(range(14, 28), block=64),
        _v41_specs(range(28, 40), block=64, mirror=True),
    ]
    backbone_names = set().union(*(set(w) for w in workers))
    workers[-1].update(draft)
    # TP4 copies of each PP stage must merge consistently as well.
    configs = ku.get_kv_cache_configs(
        cfg, [w.copy() for w in workers for _ in range(4)], [5 << 30] * 12
    )
    assert not set(draft).intersection(backbone_names)
    assert len({c.num_blocks for c in configs}) == 1
    for config in configs[8:]:
        groups = [g for g in config.kv_cache_groups if set(draft) & set(g.layer_names)]
        assert len(groups) == 1
        assert set(groups[0].layer_names) == set(draft)
        assert all(
            spec.sliding_window == 133
            for spec in groups[0].kv_cache_spec.kv_cache_specs.values()
        )
    for config in configs[:8]:
        assert not any(set(draft) & set(g.layer_names) for g in config.kv_cache_groups)
    assert all(spec.sliding_window == 133 for spec in draft.values())
