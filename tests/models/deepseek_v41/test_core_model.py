# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2 item 5: sm70/model.py with test-only stand-ins for attention / MoE / Engram (core_stubs.py).

CPU, world size 1. Synthetic checkpoint tensors take their names, shapes and dtypes from the L-MODEL census
(docs/models/deepseek-v4.1-flash/tensor_census.json); real-weight loading at TP4 is the GPU smoke (item 7)."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v41.common import contracts as C
from vllm.models.deepseek_v41.common.topology import make_stage_plan

from . import core_stubs
from .test_core_hc import ref_hc_mixes, ref_hc_post, ref_hc_pre

CENSUS = Path("/mnt/hdd/v100-research/docs/models/deepseek-v4.1-flash/tensor_census.json")


class _Untouchable:
    """Stands in for a tensor the loader must skip without reading (e.g. 101 GB Engram tables)."""

    def __getattr__(self, name):
        raise AssertionError(f"skipped checkpoint tensor was accessed ({name})")


def _census_entries() -> list[dict]:
    if not CENSUS.is_file():
        pytest.skip(f"census missing: {CENSUS}")
    return json.loads(CENSUS.read_text())["non_routed_expert_tensors"]


def _synthetic(entry: dict, gen: torch.Generator) -> torch.Tensor:
    shape, dtype, name = entry["shape"], entry["dtype"], entry["name"]
    if dtype == "BF16":
        if name.endswith("norm.weight"):
            return (1 + 0.1 * torch.rand(shape, generator=gen)).to(torch.bfloat16)
        return (torch.randn(shape, generator=gen) * 0.02).to(torch.bfloat16)
    if dtype == "F32":
        scale = 0.01 if name.endswith("_fn") else 0.5
        return torch.randn(shape, generator=gen) * scale
    if dtype == "F8_E4M3":
        # realistic magnitudes: E4M3 codes ~N(0, 32) x block scales 2^-13..-11 -> weights ~ 0.004-0.016 std
        return (torch.randn(shape, generator=gen) * 32).clamp(-448, 448).to(torch.float8_e4m3fn)
    if dtype == "F8_E8M0":
        lo, hi = (-18, -14) if ".engram." in name else (-13, -11)
        return (torch.randint(lo, hi + 1, shape, generator=gen) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    raise AssertionError(f"unexpected dtype {dtype} for {name}")


def _weights_for(layers: set[int], top_level: bool = True):
    gen = torch.Generator().manual_seed(0)
    for entry in _census_entries():
        name = entry["name"]
        if name.startswith("layers."):
            layer = int(name.split(".")[1])
            if layer not in layers:
                continue
            if ".engram.embed." in name or name.endswith(".bias_vl"):
                yield name, _Untouchable()
                continue
        elif name in ("embed.weight", "head.weight", "norm.weight"):
            if not top_level:
                continue
        else:
            yield name, _Untouchable()      # vision / aligner / image_* / mtp
            continue
        yield name, _synthetic(entry, gen)


def _vllm_config(ckpt: Path, max_tokens: int = 32) -> VllmConfig:
    from vllm.models.deepseek_v41.quant_config import DeepseekV41FP8Config
    from vllm.transformers_utils.config import get_config

    hf = get_config(str(ckpt), trust_remote_code=False)
    vc = VllmConfig()
    object.__setattr__(vc, "model_config", SimpleNamespace(hf_config=hf, dtype=torch.float16, tokenizer=str(ckpt)))
    object.__setattr__(vc, "quant_config", DeepseekV41FP8Config.from_config(dict(hf.quantization_config)))
    vc.scheduler_config.max_num_batched_tokens = max_tokens
    return vc


def _build(ckpt: Path, monkeypatch: pytest.MonkeyPatch, subset: str = "0,1,2,3"):
    from vllm.utils.torch_utils import set_default_torch_dtype

    core_stubs.install(monkeypatch.setitem)
    monkeypatch.setattr(core_stubs, "ALLOCATE_EXPERTS", False)
    monkeypatch.setenv("VLLM_DS41_CORE_LAYER_SUBSET", subset)
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41ForCausalLM

    vc = _vllm_config(ckpt)
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.float16):
        model = DeepseekV41ForCausalLM(vllm_config=vc)
    return model, vc


def _finish_loading(model, vc) -> None:
    with set_current_vllm_config(vc):
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None and hasattr(method, "process_weights_after_loading"):
                method.process_weights_after_loading(module)


@pytest.fixture
def loaded_model(ds41_dist_single, ds41_checkpoint_dir, monkeypatch):
    model, vc = _build(ds41_checkpoint_dir, monkeypatch)
    loaded = model.load_weights(_weights_for({0, 1, 2, 3}))
    _finish_loading(model, vc)
    return model, vc, loaded


def test_load_every_parameter_exactly(loaded_model) -> None:
    model, _, loaded = loaded_model
    expected = {n for n, _ in model.named_parameters() if ".ffn.experts." not in n}
    assert loaded == expected, (sorted(expected - loaded)[:10], sorted(loaded - expected)[:10])
    p = dict(model.named_parameters())
    assert p["model.layers.0.attn_norm.weight"].dtype == torch.float32
    assert p["model.layers.2.hc_attn_fn"].dtype == torch.float32 and p["model.layers.2.hc_attn_fn"].shape == (24, 20480)
    assert p["model.embed_tokens.weight"].dtype == torch.float16 and p["lm_head.weight"].dtype == torch.float16
    assert p["model.norm.weight"].dtype == torch.float32
    assert p["model.layers.3.attn.wq_b.weight"].dtype == torch.float16          # FP8 -> FP16 after processing
    assert p["model.layers.2.attn.compressor.fused_wkv_wgate.weight"].shape == (1024, 5120)
    assert p["model.layers.1.engram.wkv_r"].shape == (25600, 6144)              # TP1: all 24 sub-tables
    assert p["model.layers.0.ffn.gate.e_score_correction_bias"].dtype == torch.float32
    assert [layer.__class__.__name__ for layer in model.model.layers[:5]] == [
        "DeepseekV41DecoderLayer"] * 4 + ["PPMissingLayer"]


def test_forward_and_fp32_logits(loaded_model) -> None:
    model, vc, _ = loaded_model
    ids = torch.tensor([0, 671, 6102, 294, 8760])
    with set_current_vllm_config(vc), torch.inference_mode():
        hidden = model(ids, torch.arange(5), None)
        logits = model.compute_logits(hidden)
    assert hidden.dtype == torch.float16 and hidden.shape == (5, C.HIDDEN) and bool(torch.isfinite(hidden).all())
    assert logits.dtype == torch.float32 and logits.shape == (5, 129280) and bool(torch.isfinite(logits).all())
    ref = hidden.double() @ model.lm_head.weight.double().t()
    assert float((logits.double() - ref).abs().max()) <= 1e-3 * float(ref.abs().max())


def _twin_forward(model, ids: torch.Tensor, positions: torch.Tensor, wrong_order: bool = False) -> torch.Tensor:
    """Reference structure (model.py Transformer.forward + Block.forward) with the port's sublayers."""
    def norm(x, w):
        x32 = x.float()
        return (w * (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + C.NORM_EPS))).to(torch.float16)

    h = model.model.embed_tokens(ids).to(torch.bfloat16).unsqueeze(0).unsqueeze(2).repeat(1, 1, C.HC, 1)
    pre_mix = torch.zeros(1, ids.shape[0], C.HC)
    pre_mix[..., 0] = 1
    for layer in model.model.layers[:4]:
        residual = h
        attn_pre, attn_post, attn_comb = ref_hc_mixes(h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        x = norm(ref_hc_pre(h, attn_pre if wrong_order else pre_mix)[0], layer.attn_norm.weight)
        x = layer.attn(positions, x)
        h = ref_hc_post(x.unsqueeze(0), residual, attn_post, attn_comb)
        residual = h
        ffn_pre, ffn_post, ffn_comb = ref_hc_mixes(h, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
        x = norm(ref_hc_pre(h, attn_pre)[0], layer.ffn_norm.weight)
        h = ref_hc_post(layer.ffn(x).unsqueeze(0), residual, ffn_post, ffn_comb)
        pre_mix = ffn_pre
    return norm(ref_hc_pre(h, pre_mix)[0], model.model.norm.weight)


def test_single_pass_shift_matches_reference_structure(loaded_model) -> None:
    model, vc, _ = loaded_model
    ids, pos = torch.tensor([0, 671, 6102, 294]), torch.arange(4)
    with set_current_vllm_config(vc), torch.inference_mode():
        ours = model(ids, pos, None).double()
        twin = _twin_forward(model, ids, pos).double()
        wrong = _twin_forward(model, ids, pos, wrong_order=True).double()
    rel = float((ours - twin).norm() / twin.norm())
    rel_wrong = float((wrong - twin).norm() / twin.norm())
    assert rel <= 5e-3, rel
    assert rel_wrong > 10 * max(rel, 1e-4), (rel, rel_wrong)    # the test can tell the pre orders apart


def test_zero_padding_rows_stay_finite(loaded_model) -> None:
    model, vc, _ = loaded_model
    layer = model.model.layers[2]
    stream = torch.zeros(3, C.HC, C.HIDDEN, dtype=torch.bfloat16)
    stream[0] = 0.5
    pre = torch.zeros(3, C.HC)
    pre[:, 0] = 1
    with set_current_vllm_config(vc), torch.inference_mode():
        out, ffn_pre = layer(stream, pre, torch.arange(3))
    assert bool(torch.isfinite(out.float()).all()) and bool(torch.isfinite(ffn_pre).all())


def test_skip_checkpoint_weight(loaded_model) -> None:
    model, _, _ = loaded_model
    skip = model.skip_checkpoint_weight
    for name in ("vision.blocks.0.attn.qkv.weight", "aligner.w1.bias", "image_start", "image_newline",
                 "mtp.0.ffn.experts.3.w1.weight", "layers.1.engram.embed.weight", "layers.14.engram.embed.scale",
                 "layers.3.ffn.gate.bias_vl", "layers.4.attn.wq_a.weight", "layers.39.ffn_norm.weight"):
        assert skip(name), name
    for name in ("layers.1.engram.wkv.weight", "layers.1.engram.q_weight", "layers.0.attn.attn_sink",
                 "layers.3.ffn.experts.383.w2.scale", "embed.weight", "head.weight", "norm.weight"):
        assert not skip(name), name


def test_unknown_tensor_fails_loudly(loaded_model) -> None:
    model, _, _ = loaded_model
    with pytest.raises(KeyError):
        model.load_weights([("layers.0.attn.not_a_tensor.weight", torch.zeros(1))])
    with pytest.raises(KeyError):
        model.load_weights([("lm_head_typo.weight", torch.zeros(1))])


def test_engram_delegation_gets_relative_names(loaded_model) -> None:
    model, _, _ = loaded_model
    assert model.model.layers[1].engram.loaded_names == ["k_weight", "q_weight", "wkv.scale", "wkv.weight"]


def test_map_checkpoint_name_all_census_names() -> None:
    from vllm.models.deepseek_v41.sm70.model import map_checkpoint_name

    for entry in _census_entries():
        name = entry["name"]
        if name.startswith(("vision.", "aligner.", "image_", "mtp.")):
            continue
        mapped = map_checkpoint_name(name)
        assert mapped.startswith(("model.", "lm_head.")), (name, mapped)
        assert not mapped.endswith(".scale"), mapped
    assert map_checkpoint_name("layers.3.ffn.experts.7.w2.scale") == "model.layers.3.ffn.experts.7.w2.weight_scale"
    assert map_checkpoint_name("layers.3.attn.wq_b.scale") == "model.layers.3.attn.wq_b.weight_scale_inv"
    assert map_checkpoint_name("layers.3.ffn.gate.bias") == "model.layers.3.ffn.gate.e_score_correction_bias"
    assert map_checkpoint_name("layers.3.ffn.shared_experts.w2.weight") == \
        "model.layers.3.ffn.shared_experts.down_proj.weight"


@pytest.mark.parametrize("partition", [[20, 20], [14, 14, 12]])
def test_pp_schema_sender_equals_receiver(ds41_text_config, partition: list[int]) -> None:
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model

    stages = []
    for rank in range(len(partition)):
        fake = SimpleNamespace(stage=make_stage_plan(ds41_text_config, rank, len(partition), partition),
                               pp_is_first=rank == 0, pp_is_last=rank == len(partition) - 1)
        fake._schema = functools.partial(DeepseekV41Model._schema, fake)
        stages.append(fake)
    for t in (1, 3, 8):
        for rank in range(len(partition) - 1):
            send = DeepseekV41Model.pp_send_schema(stages[rank], t)
            recv = DeepseekV41Model.pp_static_schema(stages[rank + 1], t)
            assert list(send.items()) == list(recv.items())
            empty = DeepseekV41Model.make_empty_intermediate_tensors(stages[rank + 1], t, torch.float16,
                                                                     torch.device("meta"))
            assert list(empty.tensors) == list(recv)
            assert all(empty.tensors[k].shape == recv[k][0] and empty.tensors[k].dtype == recv[k][1] for k in recv)
    payload = sum(torch.Size(shape).numel() * dtype.itemsize
                  for shape, dtype in DeepseekV41Model.pp_static_schema(stages[-1], 1).values())
    assert payload == (50448 if len(partition) == 3 else 40976)                # §3.5 bytes/token


def test_contract_mismatch_and_open_subset_rejected(ds41_dist_single, ds41_checkpoint_dir, monkeypatch) -> None:
    from vllm.models.deepseek_v41.sm70.model import check_contract_constants

    vc = _vllm_config(ds41_checkpoint_dir)
    check_contract_constants(vc.model_config.hf_config)
    bad = SimpleNamespace(**{**vc.model_config.hf_config.to_dict(), "o_lora_rank": 512})
    with pytest.raises(ValueError, match="o_lora_rank"):
        check_contract_constants(bad)
    with pytest.raises(ValueError, match="needs source layer 2"):
        _build(ds41_checkpoint_dir, monkeypatch, subset="0,3")


def test_engram_unconsumed_tensor_raises(loaded_model, monkeypatch) -> None:
    """PORT_DESIGN §9 AM-1: L-CORE raises when the Engram module does not consume every routed tensor."""
    model, _, _ = loaded_model
    engram = model.model.layers[1].engram
    monkeypatch.setattr(engram, "load_weights", lambda items: {"wkv.weight"})
    with pytest.raises(RuntimeError, match="unconsumed"):
        model.load_weights([("layers.1.engram.wkv.weight", torch.zeros(1)),
                            ("layers.1.engram.q_weight", torch.zeros(1))])


def test_dummy_flag_reaches_engram(loaded_model) -> None:
    """PORT_DESIGN §9 AM-2: the model sees ForwardContext.is_dummy_run (True only in GPUModelRunner._dummy_run)."""
    from vllm.forward_context import set_forward_context

    model, vc, _ = loaded_model
    ids, pos = torch.tensor([0, 671]), torch.arange(2)
    with set_current_vllm_config(vc), torch.inference_mode():
        with set_forward_context(None, vc, num_tokens=2, is_dummy_run=True):
            model(ids, pos, None)
        assert model.model.layers[1].engram.last_is_dummy is True
        with set_forward_context(None, vc, num_tokens=2):
            model(ids, pos, None)
        assert model.model.layers[1].engram.last_is_dummy is False


def test_ubatching_refused(ds41_dist_single, ds41_checkpoint_dir, monkeypatch) -> None:
    core_stubs.install(monkeypatch.setitem)
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model

    vc = _vllm_config(ds41_checkpoint_dir)
    vc.parallel_config.enable_dbo = True
    with set_current_vllm_config(vc), pytest.raises(NotImplementedError, match="micro-batching"):
        DeepseekV41Model(vllm_config=vc, prefix="model")
