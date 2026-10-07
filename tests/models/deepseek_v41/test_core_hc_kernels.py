# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P5-CORE item 1: fused HC kernels (sm70/hc_kernels.py) vs common/hc.py, the FP64 truth (AM-3) and the L-REF goldens.

Gates (PORT_DESIGN §4.5 + AM-3): mixes / pre / post / comb rel <= 1e-6 vs FP64; BF16 stream and collapse outputs are
the correct BF16 rounding of the FP64 truth (>= 99.9 % equal, every element within 1 ulp: the hc_out rule); the FP16
activation equals rmsnorm_to_act of the same collapse (>= 99.9 % equal, <= 1 FP16 ulp). Golden tests feed each step
the golden's (BF16-rounded) inputs per captured layer (prefill + decode) -- per-layer, not end-to-end."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.models.deepseek_v41.common.hc import (
    hc_mixes, hc_post, hc_pre, rmsnorm_to_act, sinkhorn_split)

pytestmark = pytest.mark.sm70

D, EPS_NORM = 5120, 1e-20
GOLDEN = Path(os.environ.get("DS41_GOLDEN_DIR", "/mnt/nvme2/scratch/ds41/golden"))
GOLDEN_CASES = ("p1_legal__v100-semantic__prefill", "p1_legal__v100-semantic__decode1",
                "p3_doc__v100-semantic__prefill", "p0_smoke__v100-semantic__decode2")
GOLDEN_LAYERS = (0, 1, 2, 3, 14, 20, 21, 24, 28)


def _step(*args, **kwargs):
    from vllm.models.deepseek_v41.sm70.hc_kernels import hc_fused_step

    return hc_fused_step(*args, **kwargs)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double().cpu() - b.double().cpu()).norm() / b.double().cpu().norm().clamp_min(1e-300))


def _mixes64(stream_bf16: torch.Tensor, fn, scale, base):
    x64 = stream_bf16.double().cpu().flatten(1)
    mix = (x64 @ fn.double().cpu().t()) * torch.rsqrt(x64.square().mean(-1, keepdim=True) + EPS_NORM)
    return sinkhorn_split(mix, scale.double().cpu(), base.double().cpu())


def _assert_mixes(got, truth, gate: float = 1e-6) -> None:
    for name, a, b in zip(("pre", "post", "comb"), got, truth):
        assert _rel(a, b) <= gate, (name, _rel(a, b))


def _ulp(x: torch.Tensor, mant_bits: int) -> torch.Tensor:
    """Spacing of a ``mant_bits``-mantissa float at |x| (CPU exp2: CUDA ldexp/pow are not exact in FP64)."""
    exp = torch.frexp(x.double().cpu().abs().clamp_min(1e-30))[1]
    return torch.exp2((exp - mant_bits - 1).double())


def _assert_rounding_of(cand: torch.Tensor, truth: torch.Tensor, frac: float = 0.999) -> None:
    """hc_out rule (L-REF compare.py): cand (BF16) is the correct BF16 rounding of the FP64 truth on >= frac, and
    every element is within 1 BF16 ulp of the truth -- or, where the truth cancels to near zero, within the FP32
    summation noise of its row (2^-20 x the row's max |truth|; any FP32 implementation, the eager one included,
    misses the 1-ulp rule there: e.g. -5.28e-6 vs -5.36e-6 in a row of magnitude 10)."""
    cand64, truth64 = cand.double().cpu(), truth.double().cpu()
    exact = truth64.float().to(torch.bfloat16).double()
    equal = float((cand64 == exact).double().mean())
    assert equal >= frac, f"only {equal:.5f} of the BF16 values are the rounding of the FP64 truth"
    rows = truth64.reshape(truth64.shape[0], -1)
    floor = (rows.abs().amax(dim=1) * 2.0 ** -20).view(-1, *([1] * (truth64.dim() - 1)))
    bad = (cand64 - truth64).abs() > torch.maximum(_ulp(truth64, 7), floor)
    assert not bool(bad.any()), f"{int(bad.sum())} BF16 values beyond 1 ulp / the FP32 noise floor"


def _assert_act(act: torch.Tensor, collapsed: torch.Tensor, weight: torch.Tensor) -> None:
    ref = rmsnorm_to_act(collapsed, weight).cpu()
    act = act.cpu()
    assert act.dtype == torch.float16
    equal = float((act == ref).float().mean())
    assert equal >= 0.999, equal
    assert bool(((act.double() - ref.double()).abs() <= _ulp(ref, 10)).all())


def _params(seed: int = 0, dev: str = "cuda"):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(24, 4 * D, generator=g) * 0.02
    scale = torch.tensor([0.9, 1.3, 2.0])
    base = torch.randn(24, generator=g) * 0.5
    w = torch.rand(D, generator=g) + 0.5
    return fn.to(dev), scale.to(dev), base.to(dev), w.to(dev)


def _inputs(t: int, seed: int = 1, dev: str = "cuda"):
    g = torch.Generator().manual_seed(seed)
    stream = (torch.randn(t, 4, D, generator=g) * 3).to(torch.bfloat16)
    sub = torch.randn(t, D, generator=g) * 7
    _, post, comb = sinkhorn_split(torch.randn(t, 24, generator=g), torch.tensor([0.9, 1.3, 2.0]),
                                   torch.randn(24, generator=g) * 0.5)
    cpre = torch.rand(t, 4, generator=g) + 0.1
    return stream.to(dev), sub.to(dev), post.to(dev), comb.to(dev), cpre.to(dev)


# ---------------------------------------------------------------- synthetic, all step shapes
@pytest.mark.parametrize("num_tokens", [1, 3, 4, 5, 16, 17, 300, 1100])
def test_shifted_post_pre_matches_reference(num_tokens: int) -> None:
    fn, scale, base, w = _params()
    stream, sub, post, comb, cpre = _inputs(num_tokens)
    out, mixes, act, coll = _step(stream, sub_out=sub, post=post, comb=comb, mix=(fn, scale, base),
                                  collapse_pre=cpre, norm_weight=w)
    exact = (post.double().unsqueeze(-1) * sub.double().unsqueeze(1)
             + torch.einsum("tjc,tjd->tcd", comb.double(), stream.double()))
    _assert_rounding_of(out, exact)
    ref_out = hc_post(sub, stream, post, comb)
    assert float((out != ref_out).float().mean()) < 1e-3
    _assert_mixes(mixes, _mixes64(out, fn, scale, base))
    _assert_rounding_of(coll, torch.sum(cpre.double().unsqueeze(-1) * out.double(), dim=1))
    assert float((coll != hc_pre(out, cpre)).float().mean()) < 1e-3
    _assert_act(act, coll, w)
    assert torch.equal(stream, _inputs(num_tokens)[0]), "the residual stream must not be modified"


@pytest.mark.parametrize("num_tokens", [1, 33])
def test_entry_and_post_only_and_final(num_tokens: int) -> None:
    fn, scale, base, w = _params(2)
    stream, sub, post, comb, cpre = _inputs(num_tokens, seed=4)
    same, mixes, act, coll = _step(stream, mix=(fn, scale, base), collapse_pre=cpre, norm_weight=w)
    assert same is stream
    _assert_mixes(mixes, _mixes64(stream, fn, scale, base))
    assert float((coll != hc_pre(stream, cpre)).float().mean()) < 1e-3
    _assert_act(act, coll, w)
    torch_mix = hc_mixes(stream, fn, scale, base)          # the eager path is within the same FP64 gate
    _assert_mixes(torch_mix, _mixes64(stream, fn, scale, base))

    out, none_mix, none_act, none_coll = _step(stream, sub_out=sub, post=post, comb=comb)
    assert none_mix is None and none_act is None and none_coll is None
    assert float((out != hc_post(sub, stream, post, comb)).float().mean()) < 1e-3

    out2, none_mix, act2, coll2 = _step(stream, sub_out=sub, post=post, comb=comb, collapse_pre=cpre, norm_weight=w)
    # kernel specialisations may contract FMAs differently: equal up to rare BF16 ties
    assert none_mix is None and float((out2 != out).float().mean()) < 1e-3
    _assert_act(act2, coll2, w)


def test_zero_rows_and_empty_batch() -> None:
    fn, scale, base, w = _params()
    stream = torch.zeros(5, 4, D, dtype=torch.bfloat16, device="cuda")
    out, mixes, act, _ = _step(stream, sub_out=torch.zeros(5, D, device="cuda"),
                               post=torch.ones(5, 4, device="cuda"), comb=torch.full((5, 4, 4), 0.25, device="cuda"),
                               mix=(fn, scale, base), collapse_pre=torch.ones(5, 4, device="cuda"), norm_weight=w)
    assert all(bool(torch.isfinite(t).all()) for t in mixes) and float(act.abs().sum()) == 0
    assert float(out.float().abs().sum()) == 0
    empty = torch.zeros(0, 4, D, dtype=torch.bfloat16, device="cuda")
    out, mixes, act, _ = _step(empty, mix=(fn, scale, base), collapse_pre=torch.zeros(0, 4, device="cuda"),
                               norm_weight=w)
    assert out.shape[0] == 0 and mixes[2].shape == (0, 4, 4) and act.shape == (0, D)


def test_input_validation() -> None:
    fn, scale, base, w = _params()
    stream, sub, post, comb, cpre = _inputs(2)
    with pytest.raises(ValueError):
        _step(stream.float(), mix=(fn, scale, base))
    with pytest.raises(ValueError):
        _step(stream, sub_out=sub, post=post)
    with pytest.raises(ValueError):
        _step(stream, collapse_pre=cpre)
    with pytest.raises(ValueError):
        _step(stream, mix=(fn.half(), scale, base))
    with pytest.raises(ValueError):
        _step(stream.cpu(), mix=(fn, scale, base))


# ---------------------------------------------------------------- model wiring: fused == eager structure
class _Norm(nn.Module):
    def __init__(self, w: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(w, requires_grad=False)


def _fake_layer(layer_id: int, seed: int, engram: bool):
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41DecoderLayer

    layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_id = layer_id
    fa, sa, ba, wa = _params(seed)
    ff, sf, bf, wf = _params(seed + 100)
    # fn / 10: random 20480-long mix rows at 0.02 make Sinkhorn chaotic (a 3e-4 stream difference -> 7e-4 in pre),
    # which would test the random weights, not the wiring
    for name, t in (("hc_attn_fn", fa / 10), ("hc_attn_scale", sa), ("hc_attn_base", ba),
                    ("hc_ffn_fn", ff / 10), ("hc_ffn_scale", sf), ("hc_ffn_base", bf)):
        setattr(layer, name, nn.Parameter(t, requires_grad=False))
    layer.attn_norm, layer.ffn_norm = _Norm(wa), _Norm(wf)
    g = torch.Generator().manual_seed(seed + 7)
    wa_lin = (torch.randn(D, generator=g) * 0.5).cuda()
    wf_lin = (torch.randn(D, generator=g) * 0.5).cuda()
    # elementwise-linear FP32 stand-ins: the paths differ only by rounding ties, which smooth sublayers do not
    # amplify (a real block's own gate is the §4.5 per-layer 5e-3; the golden tests below gate the HC math itself)
    layer.attn = lambda positions, x: x.float() * wa_lin + 0.25
    layer.ffn = lambda x: x.float() * wf_lin - 0.1
    layer.engram = (lambda stream, positions: (stream.float() * 1.03 + 0.01).to(torch.bfloat16)) if engram else None
    return layer


def _fake_model(pp_is_last: bool, fused: bool):
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model

    layers = [_fake_layer(i, 10 * i, engram=(i == 1)) for i in range(4)]
    fake = SimpleNamespace(pp_is_first=False, pp_is_last=pp_is_last, layers=layers, start_layer=0, end_layer=4,
                           hc_fused=fused, mirror=None, kv20_crc_check=False,
                           dspark_aux_layers=(),
                           stage=SimpleNamespace(exports_kv_sources=(), mirrored_kv_sources=()),
                           norm=_Norm(_params(999)[3]))
    return functools.partial(DeepseekV41Model.forward, fake)


@pytest.mark.parametrize("pp_is_last", [False, True])
def test_model_fused_path_matches_eager(pp_is_last: bool) -> None:
    from vllm.models.deepseek_v41.common import contracts as C
    from vllm.sequence import IntermediateTensors

    stream, _, _, _, pre = _inputs(9, seed=21)
    payload = lambda: IntermediateTensors({C.PP_KEY_HIDDEN: stream.clone(), C.PP_KEY_PRE_MIX: pre.clone()})  # noqa
    positions = torch.arange(9, device="cuda")
    with torch.inference_mode():
        eager = _fake_model(pp_is_last, fused=False)(None, positions, payload())
        fused = _fake_model(pp_is_last, fused=True)(None, positions, payload())
    if pp_is_last:
        assert fused.dtype == eager.dtype == torch.float16
        assert _rel(fused, eager) <= 2e-3
    else:
        for key in (C.PP_KEY_HIDDEN, C.PP_KEY_PRE_MIX):
            a, b = fused.tensors[key], eager.tensors[key]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert _rel(a, b) <= 2e-3, (key, _rel(a, b))


# ---------------------------------------------------------------- golden per-layer gates (L-REF, AM-3)
@functools.lru_cache(maxsize=None)
def _ckpt_weights(ckpt: str, layer: int) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads((Path(ckpt) / "model.safetensors.index.json").read_text())["weight_map"]
    names = [f"layers.{layer}.{n}" for n in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn",
                                              "hc_ffn_scale", "hc_ffn_base", "attn_norm.weight", "ffn_norm.weight")]
    out = {}
    for name in names:
        with safe_open(str(Path(ckpt) / index[name]), "pt") as f:
            out[name.split(".", 2)[2]] = f.get_tensor(name).float().cuda()
    return out


def _golden(case: str, layer: int) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = GOLDEN / case / f"L{layer:02d}.safetensors"
    if not path.is_file():
        pytest.skip(f"no golden {path}")
    return load_file(str(path))


def _rows(g: dict[str, torch.Tensor], t: torch.Tensor, n_full: int) -> torch.Tensor:
    """Heavy golden tensors are stored on a row subset (_rows); select those rows of a full-length tensor."""
    rows = g.get("_rows")
    if rows is None or t.shape[0] != n_full:
        return t
    return t[rows.long().to(t.device)]


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.float().to(torch.bfloat16).cuda().contiguous()


@pytest.mark.weights
@pytest.mark.parametrize("case", GOLDEN_CASES)
@pytest.mark.parametrize("layer", GOLDEN_LAYERS)
def test_golden_hc_steps(ds41_checkpoint_dir, case: str, layer: int) -> None:
    g = _golden(case, layer)
    w = _ckpt_weights(str(ds41_checkpoint_dir), layer)
    n = g["stream_in"].shape[0]
    rows_n = g["stream_attn"].shape[0]
    sel = (lambda t: _rows(g, t, n)) if rows_n != n else (lambda t: t)
    attn_in = _bf16(sel(g["engram.out"] if "engram.out" in g else g["stream_in"]))
    attn_mix = (w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
    ffn_mix = (w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
    pre_in = sel(g["pre_in"]).float().cuda()

    def truth(name):
        return sel(g[name])

    # (1) entry step on the golden attention input: attention mixes + collapse(pre_in) + attn_norm
    _, mixes, act, coll = _step(attn_in, mix=attn_mix, collapse_pre=pre_in, norm_weight=w["attn_norm.weight"])
    _assert_mixes(mixes, [truth(f"hc.attn_{k}_bf16in_f64") for k in ("pre", "post", "comb")])
    _assert_rounding_of(coll, truth("hc.attn_x_bf16in_f64"))
    _assert_act(act, coll, w["attn_norm.weight"])
    # (2) shifted post+pre fed the golden attn.out and the golden's own (FP32) attention mixes -- the inputs the
    # *_bf16in_f64 stream / collapse truths were computed from (checked: they reproduce the truths exactly)
    a_pre, a_post, a_comb = (truth(f"hc.attn_{k}").float().cuda() for k in ("pre", "post", "comb"))
    stream_attn, mixes, act, coll = _step(attn_in, sub_out=sel(g["attn.out"]).float().cuda().contiguous(),
                                          post=a_post, comb=a_comb, mix=ffn_mix, collapse_pre=a_pre,
                                          norm_weight=w["ffn_norm.weight"])
    _assert_rounding_of(stream_attn, truth("stream_attn_bf16in_f64"))
    _assert_mixes(mixes, _mixes64(stream_attn, *ffn_mix))                       # mixes of its own output
    _assert_rounding_of(coll, torch.sum(a_pre.double().cpu().unsqueeze(-1) * stream_attn.double().cpu(), dim=1))
    _assert_act(act, coll, w["ffn_norm.weight"])
    # (3) entry step on the golden (BF16-rounded) stream_attn: FFN mixes / collapse vs the golden FP64 refs
    golden_attn = _bf16(g["stream_attn"])
    _, mixes, _, coll = _step(golden_attn, mix=ffn_mix, collapse_pre=a_pre, norm_weight=w["ffn_norm.weight"])
    _assert_mixes(mixes, [truth(f"hc.ffn_{k}_bf16in_f64") for k in ("pre", "post", "comb")])
    _assert_rounding_of(coll, truth("hc.ffn_x_bf16in_f64"))
    # (4) post-only (stage end / before Engram) fed golden moe.out + the golden FFN mixes -> stream_out
    f_post, f_comb = (truth(f"hc.ffn_{k}").float().cuda() for k in ("post", "comb"))
    out = _step(golden_attn, sub_out=sel(g["moe.out"]).float().cuda().contiguous(), post=f_post, comb=f_comb)[0]
    _assert_rounding_of(out, truth("stream_out_bf16in_f64"))


@pytest.mark.weights
@pytest.mark.parametrize("case", ["p1_legal__v100-semantic__prefill", "p3_doc__v100-semantic__prefill"])
def test_golden_final_collapse(ds41_checkpoint_dir, case: str) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = GOLDEN / case / "final.safetensors"
    if not path.is_file():
        pytest.skip(f"no golden {path}")
    g = load_file(str(path))
    if "final.hc_bf16in_f64" not in g:
        pytest.skip("final golden without the AM-3 FP64 collapse")
    index = json.loads((ds41_checkpoint_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(ds41_checkpoint_dir / index["norm.weight"]), "pt") as f:
        norm = f.get_tensor("norm.weight").float().cuda()
    stream = _bf16(g["final.stream_in"])
    pre = g["final.pre_in"]
    if pre.shape[0] != stream.shape[0]:          # heavy tensors on the _rows subset, pre_in on all rows
        pre = pre[g["_rows"].long()]
    _, _, act, coll = _step(stream, collapse_pre=pre.float().cuda(), norm_weight=norm)
    truth = g["final.hc_bf16in_f64"]
    assert truth.shape == coll.shape
    _assert_rounding_of(coll, truth)
    _assert_act(act, coll, norm)


def test_dump_mode_fused_writes_the_eager_names(ds41_dist_single, monkeypatch, tmp_path) -> None:
    """While dumping, the fused path applies each layer's FFN post itself (no shift), so every §3.8 name exists and
    matches the eager path up to rounding ties."""
    import importlib

    from safetensors.torch import load_file

    from vllm.models.deepseek_v41.common import contracts as C
    from vllm.models.deepseek_v41.common import dump as ds41_dump
    from vllm.sequence import IntermediateTensors

    stream, _, _, _, pre = _inputs(5, seed=31)
    positions = torch.arange(5, device="cuda")
    out = {}
    try:
        for fused in (False, True):
            monkeypatch.setenv(ds41_dump.DIR_KNOB, str(tmp_path / str(fused)))
            monkeypatch.setenv(ds41_dump.LAYERS_KNOB, "all")
            monkeypatch.setenv(ds41_dump.STEPS_KNOB, "all")
            importlib.reload(ds41_dump)
            with torch.inference_mode():
                _fake_model(False, fused)(None, positions, IntermediateTensors(
                    {C.PP_KEY_HIDDEN: stream.clone(), C.PP_KEY_PRE_MIX: pre.clone()}))
            out[fused] = {f.name: load_file(str(f)) for f in sorted((tmp_path / str(fused)).rglob("L*.safetensors"))}
    finally:
        for knob in (ds41_dump.DIR_KNOB, ds41_dump.LAYERS_KNOB, ds41_dump.STEPS_KNOB):
            monkeypatch.delenv(knob, raising=False)
        importlib.reload(ds41_dump)
    assert list(out[True]) == list(out[False]) == [f"L{i:02d}.safetensors" for i in range(4)]
    for name, eager in out[False].items():
        fused = out[True][name]
        assert sorted(fused) == sorted(eager), name
        for key in eager:
            assert fused[key].shape == eager[key].shape and fused[key].dtype == eager[key].dtype, (name, key)
            assert _rel(fused[key], eager[key]) <= 2e-3, (name, key, _rel(fused[key], eager[key]))
