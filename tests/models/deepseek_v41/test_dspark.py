# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v41.common.hc import hc_post
from vllm.models.deepseek_v41.sm70.dspark import (
    DSparkDeepseekV41ForCausalLM as Draft,
)
from vllm.models.deepseek_v41.sm70.dspark import (
    DSparkMetadataBuilder,
    capture_target_input,
    configure_target_capture,
)
from vllm.v1.attention.backend import CommonAttentionMetadata


@pytest.mark.parametrize("anchor", [1, 127, 128, 255, 513])
def test_noncausal_same_context_and_all_queries(anchor, monkeypatch):
    # CPU-only test: the CUDA platform's global default enables pinned memory.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    builder = object.__new__(DSparkMetadataBuilder)
    builder._mirror_checks = []
    builder.device = torch.device("cpu")
    builder.kernel_block_size = 16
    builder.kv_cache_spec = SimpleNamespace(block_size=16)
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=10)
    )
    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 5, 10], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 5, 10], dtype=torch.int32),
        seq_lens=torch.tensor([anchor + 5, anchor + 12], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([anchor + 99, anchor + 99]),
        num_reqs=2,
        num_actual_tokens=10,
        max_query_len=5,
        max_seq_len=anchor + 12,
        block_table_tensor=torch.stack((torch.arange(50), torch.arange(50) + 60)),
        slot_mapping=torch.arange(10),
        causal=False,
    )
    md = builder.build_for_drafting(cm, 0)
    assert md.causal is False
    for r, a in enumerate([anchor, anchor + 7]):
        row = md.window_slots[r * 5]
        logical = torch.arange(a - 128, a + 5)
        expected = torch.where(logical >= 0, logical + 60 * 16 * r, -1)
        assert torch.equal(row, expected)
        assert torch.equal(md.window_slots[r * 5 : r * 5 + 5], row.expand(5, -1))
    with pytest.raises(ValueError, match="noncausal"):
        builder.build_for_drafting(replace(cm, causal=True), 0)
    with pytest.raises(ValueError, match="five queries"):
        builder.build_for_drafting(
            replace(cm, query_start_loc_cpu=torch.tensor([0, 4, 10])), 0
        )


def test_capture_preserves_bf16_range_and_block_input(monkeypatch):
    stream = torch.full((2, 4, 5120), 200000.0, dtype=torch.bfloat16)
    sub = torch.ones((2, 5120))
    post = torch.ones((2, 4))
    comb = torch.eye(4).expand(2, 4, 4).contiguous()
    expected = hc_post(sub, stream, post, comb)
    from vllm.models.deepseek_v41.sm70 import hc_kernels

    monkeypatch.setattr(
        hc_kernels,
        "hc_fused_step",
        lambda h, **kw: (hc_post(kw["sub_out"], h, kw["post"], kw["comb"]),),
    )
    materialized, aux = capture_target_input(stream, (sub, post, comb))
    assert torch.equal(materialized, expected)
    assert aux.dtype == torch.float32
    assert torch.equal(aux, expected.mean(1).float())
    assert torch.isfinite(aux).all() and aux.max() > 65504


def test_configure_capture_only_last_stage():
    target = SimpleNamespace(pp_is_last=True, owns_layer=lambda i: 28 <= i < 40)
    configure_target_capture(target, (38, 39, 40))
    assert target.dspark_aux_layers == (37, 38, 39)
    with pytest.raises(ValueError, match="aux layers"):
        configure_target_capture(target, (37, 38, 39))
    target.owns_layer = lambda i: i < 39
    with pytest.raises(ValueError, match="final stage"):
        configure_target_capture(target, (38, 39, 40))
    target.pp_is_last = False
    configure_target_capture(target, (38, 39, 40))
    assert target.dspark_aux_layers == ()


@pytest.mark.parametrize(
    "source,dest",
    [
        ("embed.weight", "model.embed_tokens.weight"),
        ("head.weight", "lm_head.weight"),
        ("mtp.0.main_proj.scale", "model.main_proj.weight_scale_inv"),
        ("mtp.0.hc_attn_fn", "model.layers.0.hc_attn_fn"),
        (
            "mtp.1.ffn.experts.127.w2.scale",
            "model.layers.1.ffn.experts.127.w2.weight_scale",
        ),
        ("mtp.2.markov_head.embed.weight", "model.markov_embed.weight"),
        ("mtp.2.markov_head.head.weight", "model.markov_weight"),
        ("mtp.2.confidence_head.proj.weight", "model.confidence_weight"),
    ],
)
def test_weight_mapping(source, dest):
    assert Draft._remap_dspark_name(source) == dest


def test_mapping_fails_loud():
    for name in ["mtp.3.hc_attn_fn", "mtp.1.main_proj.weight", "mtp.0.norm.weight"]:
        with pytest.raises(KeyError):
            Draft._remap_dspark_name(name)
    assert Draft._remap_dspark_name("layers.0.attn.wq_a.weight") is None
    assert Draft._remap_dspark_name("mtp.0.ffn.gate.bias_vl") is None


def test_fp32_markov_and_confidence():
    draft = object.__new__(Draft)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.markov_weight = nn.Parameter(
        torch.full((16, 4), 100.0, dtype=torch.half)
    )
    draft.model.confidence_weight = nn.Parameter(torch.ones(1, 8))
    h = torch.full((2, 4), 100.0, dtype=torch.half)
    bias = draft.markov_bias(h)
    assert bias.dtype == torch.float32 and bias[0, 0] == 40000
    assert torch.equal(draft.confidence_logits(h, h), torch.full((2,), 800.0))


def _loader_shell():
    draft = object.__new__(Draft)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.layers = nn.ModuleList([nn.Module()])
    layer = draft.model.layers[0]
    layer.attn = nn.Module()
    layer.attn.fused_wqa_wkv = nn.Linear(2, 4, bias=False)

    def load(p, w, shard):
        p.data[shard * 2 : shard * 2 + 2].copy_(w)

    layer.attn.fused_wqa_wkv.weight.weight_loader = load
    return draft


def test_loader_rejects_partial_and_duplicate_fused_params():
    draft = _loader_shell()
    w = torch.ones((2, 2))
    with pytest.raises(RuntimeError, match="filled only part"):
        draft.load_weights([("mtp.0.attn.wq_a.weight", w)])
    with pytest.raises(ValueError, match="duplicate"):
        draft.load_weights([("mtp.0.attn.wq_a.weight", w)] * 2)
    loaded = draft.load_weights(
        [("mtp.0.attn.wq_a.weight", w), ("mtp.0.attn.wkv.weight", w * 2)]
    )
    assert loaded == {"model.layers.0.attn.fused_wqa_wkv.weight"}
    with pytest.raises(RuntimeError, match="did not provide"):
        draft.load_weights([])
    with pytest.raises(KeyError, match="absent"):
        draft.load_weights([("mtp.0.unexpected.weight", w)])


def test_attention_without_metadata_fails_loudly():
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v41.sm70.dspark import DSparkAttention

    attn = object.__new__(DSparkAttention)
    nn.Module.__init__(attn)
    with (
        override_forward_context(
            ForwardContext(no_compile_layers={}, attn_metadata=None, slot_mapping={})
        ),
        pytest.raises(RuntimeError, match="outside a dummy run"),
    ):
        attn(torch.arange(5), torch.zeros(5, 5120))
