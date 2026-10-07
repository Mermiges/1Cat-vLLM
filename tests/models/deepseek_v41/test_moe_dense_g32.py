# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P5-MOE: dense attention projections kept FP8 (group 32) -- quant method, call-site dispatch, goldens (D9, D19).

Synthetic (no checkpoint): the dispatch helpers of ``quant_config`` (``v41_linear``, ``v41_grouped_linear``,
``v41_linear_fp32_input``) on group-32 layers against the FP16 fallback layers they replace -- one-hot rows bitwise on
every path, random inputs at the FP16/FP32 output floor; the fallback branches bitwise equal to the P2 call-site code;
the quant method's knob / prefix selection, exponent-window fallback and memory; CUDA-graph replay of the decode paths.

Real weights + L-REF goldens (``-m weights``): the attention per-op links that touch these projections (qr, q, o-proj,
indexer q) and L-ATTN's golden-input end-to-end layer test, both with the module's projections converted to group 32
from the RAW checkpoint bytes (``_attn_to_g32``) -- the same gates as the FP16 fallback (PORT_DESIGN §4.5).
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v41 import quant_config as qc
from vllm.models.deepseek_v41.sm70 import fp8_g32

pytestmark = pytest.mark.sm70

DEV = torch.device("cuda")
# (name, N, K, groups) at TP4 (the switched projections)
SHAPES = [("wq_a+wkv", 1792, 5120, 1), ("wq_b", 8192, 1280, 1), ("wo_a", 2048, 4096, 2),
          ("indexer_wq_b", 4096, 1280, 1)]
TOKENS = [1, 2, 3, 5, 8, 9, 64, 300]


@pytest.fixture(autouse=True)
def _fp32_cublas_reductions():
    matmul = torch.backends.cuda.matmul
    saved = matmul.allow_fp16_reduced_precision_reduction
    matmul.allow_fp16_reduced_precision_reduction = False      # PORT_DESIGN §4.1, as the model sets it
    yield
    matmul.allow_fp16_reduced_precision_reduction = saved


def _rand_fp8(n: int, k: int, seed: int, lo: int = -13, hi: int = -6) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0x3C
    e = torch.randint(lo, hi + 1, (n // 32, k // 32), device="cuda", generator=g)
    return codes.view(torch.float8_e4m3fn), torch.pow(2.0, e.float()).to(torch.float8_e8m0fnu)


class _Lin(nn.Module):
    def __init__(self, prefix: str = "t", groups: int = 1) -> None:
        super().__init__()
        self.prefix = prefix
        if groups > 1:
            self.is_bmm = True
            self.bmm_batch_size = groups


def _pair(n: int, k: int, groups: int, seed: int) -> tuple[_Lin, _Lin, torch.Tensor]:
    """(fallback layer with the exact FP16 weight, g32 layer, FP32 exact weight)."""
    w, s = _rand_fp8(n, k, seed)
    exact = fp8_g32.dequant_fp8_g32_reference(w, s)
    fb = _Lin(groups=groups)
    fb.weight = nn.Parameter(qc.dequantize_fp8_block32_exact(w, s), requires_grad=False)
    g = _Lin(groups=groups)
    qc.DeepseekV41SM70Fp8LinearMethod._keep_fp8_g32(g, w, s)
    return fb, g, exact


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt())


def _call(lin: _Lin, x: torch.Tensor, groups: int, out_dtype: torch.dtype) -> torch.Tensor:
    if groups > 1:
        return qc.v41_grouped_linear(lin, x, groups, out_dtype)
    return qc.v41_linear(lin, x, out_dtype)


def _ref(x: torch.Tensor, exact: torch.Tensor, groups: int) -> torch.Tensor:
    m = x.shape[0]
    k = exact.shape[1]
    xg = x.float().view(m, groups, k)
    return torch.einsum("mgk,gnk->mgn", xg, exact.view(groups, -1, k)).reshape(m, -1)


# ------------------------------------------------------------------------------------------------ dispatch exactness
@pytest.mark.parametrize("name,n,k,groups", SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_g32_dispatch_matches_fallback(name, n, k, groups, out_dtype):
    fb, g, exact = _pair(n, k, groups, seed=n + k)
    for m in TOKENS:
        # one-hot rows pick single weights: every path returns them exactly (rounded once to out_dtype)
        cols = torch.randint(0, groups * k, (m,), device="cuda")
        onehot = torch.zeros(m, groups * k, dtype=torch.float16, device="cuda")
        onehot[torch.arange(m), cols] = 1.0
        assert torch.equal(_call(g, onehot, groups, out_dtype), _call(fb, onehot, groups, out_dtype)), (name, m)
        x = torch.randn(m, groups * k, device="cuda").half()
        ref = _ref(x, exact, groups)
        got, base = _call(g, x, groups, out_dtype), _call(fb, x, groups, out_dtype)
        assert got.dtype == out_dtype and got.shape == (m, n)
        floor = 1e-3 if out_dtype == torch.float16 else 1e-5      # FP16 output rounding / FP32 accumulation order
        assert _rel(got, ref) <= floor and _rel(base, ref) <= floor, (name, m, _rel(got, ref), _rel(base, ref))


def test_grouped_fallback_is_bitwise_the_p2_bmm():
    fb, _, _ = _pair(2048, 4096, 2, seed=7)
    for m in (1, 4, 37, 512):
        o = torch.randn(m, 2, 4096, device="cuda").half()
        w = fb.weight.view(2, 1024, 4096)
        p2 = torch.bmm(o.transpose(0, 1), w.transpose(1, 2), out_dtype=torch.float32)
        p2 = p2.transpose(0, 1).reshape(m, 2048).to(torch.float16)
        assert torch.equal(qc.v41_grouped_linear(fb, o.reshape(m, 2 * 4096), 2, torch.float16), p2)


def test_fp32_input_is_bitwise_the_fallback_sgemm():
    """Indexer q: a g32 layer reproduces the fallback's FP32 SGEMM bit for bit (L-ATTN's bitwise idx.q gate)."""
    fb, g, _ = _pair(4096, 1280, 1, seed=11)
    for m in (1, 3, 8, 64, 2046):
        x = torch.randn(m, 1280, device="cuda")
        p2 = torch.mm(x, fb.weight.float().t())                     # compressor.mm_fp32_full's FP32 branch
        assert torch.equal(qc.v41_linear_fp32_input(fb, x), p2)
        assert torch.equal(qc.v41_linear_fp32_input(g, x), p2), m
        xh = x.half()
        assert torch.equal(qc.v41_linear_fp32_input(fb, xh), torch.mm(xh, fb.weight.t(), out_dtype=torch.float32))


def test_fp32_input_gemv_is_exact_on_one_hot_rows():
    _, g, exact = _pair(4096, 1280, 1, seed=12)
    x = torch.zeros(4, 1280, device="cuda")
    x[torch.arange(4), torch.tensor([0, 5, 640, 1279])] = torch.tensor([1.0, -2.0, 0.5, 3.0], device="cuda")
    got = fp8_g32.fp8_g32_gemv(x, g.ds41_g32, out_dtype=torch.float32)
    assert torch.equal(got, x @ exact.t())


def test_decode_paths_replay_in_cuda_graphs():
    pairs = {name: _pair(n, k, groups, seed=3) for name, n, k, groups in SHAPES}
    for m in (1, 2, 4, 8):
        xs = {name: torch.randn(m, groups * k, device="cuda").half() for name, n, k, groups in SHAPES}
        outs: dict[str, torch.Tensor] = {}

        def run(xs: dict = xs, outs: dict = outs) -> None:
            for name, _, _, groups in SHAPES:
                out_dtype = torch.float32 if name == "wq_a+wkv" else torch.float16
                outs[name] = _call(pairs[name][1], xs[name], groups, out_dtype)

        run()
        eager = {kk: v.clone() for kk, v in outs.items()}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for name, _, k, groups in SHAPES:
            xs[name].copy_(torch.randn(m, groups * k, device="cuda").half())
        graph.replay()
        torch.cuda.synchronize()
        for name, _, _, groups in SHAPES:
            out_dtype = torch.float32 if name == "wq_a+wkv" else torch.float16
            assert torch.equal(outs[name], _call(pairs[name][1], xs[name], groups, out_dtype)), (name, m)
            assert not torch.equal(outs[name], eager[name])


# ------------------------------------------------------------------------------------------------ quant method
QCFG = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0", "expert_dtype": "fp4"}


def _layer(prefix: str, n: int, k: int, monkeypatch, enabled: bool | None, groups: int = 1):
    from types import SimpleNamespace

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear

    if enabled is None:
        monkeypatch.delenv(qc.DENSE_G32_KNOB, raising=False)
    else:
        monkeypatch.setenv(qc.DENSE_G32_KNOB, "1" if enabled else "0")
    cfg = VllmConfig()
    object.__setattr__(cfg, "model_config", SimpleNamespace(dtype=torch.float16))
    with set_current_vllm_config(cfg), torch.device(DEV):
        layer = ReplicatedLinear(k, n, bias=False, quant_config=qc.DeepseekV41FP8Config.from_config(dict(QCFG)),
                                 prefix=prefix, disable_tp=True, params_dtype=torch.float16)
    if groups > 1:
        layer.is_bmm = True
        layer.bmm_batch_size = groups
    return layer


def _load(layer, n: int, k: int, seed: int, lo: int = -13, hi: int = -6) -> torch.Tensor:
    w, s = _rand_fp8(n, k, seed, lo, hi)
    if hi > fp8_g32.SCALE_EXP_MAX:      # keep |E4M3| <= 240 so the FP16 fallback stays exact at 2^8
        b = w.view(torch.uint8)
        b[(b & 0x78) == 0x78] = 0x3C
    layer.weight.weight_loader(layer.weight, w.cpu())
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, s.cpu())
    layer.quant_method.process_weights_after_loading(layer)
    return fp8_g32.dequant_fp8_g32_reference(w, s)


@pytest.mark.parametrize("prefix,switched", [
    ("model.layers.3.attn.fused_wqa_wkv", True), ("model.layers.3.attn.wq_b", True),
    ("model.layers.20.attn.indexer.wq_b", False), ("model.layers.3.attn.wo_b", False),
    ("model.layers.3.ffn.shared_experts.gate_up_proj", False), ("model.layers.3.ffn.shared_experts.down_proj", False),
    ("model.layers.3.attn.wq_a", False),
])
def test_knob_selects_the_switched_projections(ds41_dist_single, monkeypatch, prefix, switched):
    n, k = 256, 512
    layer = _layer(prefix, n, k, monkeypatch, enabled=True)
    exact = _load(layer, n, k, seed=5)
    assert (getattr(layer, "ds41_g32", None) is not None) == switched
    if switched:
        assert layer.weight.dtype == torch.uint8 and layer.weight.numel() == n * k
        assert layer.weight_scale_inv.dtype == torch.float16 and layer.weight_scale_inv.numel() == n * k // 32
        x = torch.randn(3, k, device="cuda").half()
        out, _ = layer(x)                                    # quant_method.apply (FP16 output)
        assert out.dtype == torch.float16 and _rel(out, x.float() @ exact.t()) <= 1e-3
    else:
        assert layer.weight.dtype == torch.float16 and torch.equal(layer.weight.float(), exact)


@pytest.mark.parametrize("enabled", [False, None])
def test_knob_off_keeps_the_fallback(ds41_dist_single, monkeypatch, enabled):
    layer = _layer("model.layers.3.attn.wq_b", 256, 512, monkeypatch, enabled=enabled)
    _load(layer, 256, 512, seed=6)
    expect_g32 = qc.DENSE_G32_DEFAULT if enabled is None else False
    assert (getattr(layer, "ds41_g32", None) is not None) == expect_g32


def test_grouped_layer_and_apply(ds41_dist_single, monkeypatch):
    layer = _layer("model.layers.3.attn.wo_a", 2 * 128, 64, monkeypatch, enabled=True, groups=2)
    exact = _load(layer, 256, 64, seed=8)
    assert layer.ds41_g32_groups == 2
    x = torch.randn(5, 2, 64, device="cuda").half()
    out = layer.quant_method.apply(layer, x)
    ref = torch.einsum("tgk,grk->tgr", x.float(), exact.view(2, 128, 64))
    assert out.shape == (5, 2, 128) and _rel(out, ref) <= 1e-3


def test_out_of_window_scales_keep_the_fallback_loudly(ds41_dist_single, monkeypatch):
    warned: list[str] = []
    monkeypatch.setattr(qc.logger, "warning", lambda msg, *a: warned.append(msg % a))
    layer = _layer("model.layers.3.attn.wq_b", 64, 64, monkeypatch, enabled=True)
    exact = _load(layer, 64, 64, seed=9, lo=8, hi=8)   # window refuses e = 8; the values themselves fit FP16
    assert getattr(layer, "ds41_g32", None) is None and torch.equal(layer.weight.float(), exact)
    assert len(warned) == 1 and "leave the exact group-32 window" in warned[0]


def test_unswitched_call_site_fails_loudly(ds41_dist_single, monkeypatch):
    layer = _layer("model.layers.3.attn.wq_b", 64, 64, monkeypatch, enabled=True)
    _load(layer, 64, 64, seed=10)
    with pytest.raises(RuntimeError, match="grouped"):
        layer.ds41_g32_groups = 2
        qc.v41_linear(layer, torch.zeros(1, 64, device="cuda").half(), torch.float16)
    layer.ds41_g32 = None
    with pytest.raises(RuntimeError, match="FP16"):
        qc.v41_linear(layer, torch.zeros(1, 64, device="cuda").half(), torch.float16)


def test_memory_per_switched_layer():
    """Bytes of the switched projections per TP4 rank: FP16 fallback vs FP8 + FP16 group scales (17/32 of FP16)."""
    report = {}
    for name, n, k, _ in SHAPES:
        w, s = _rand_fp8(n, k, seed=1)
        packed = fp8_g32.prepare_fp8_g32(w, s)
        report[name] = (n * k * 2, packed.nbytes)
        assert packed.nbytes == n * k + n * k // 32 * 2
    print(json.dumps(report))
