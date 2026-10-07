# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P5-MOE microbench: the candidate attention projections per layer per TP4 rank, FP16 fallback (the P2 call-site code)
vs group-32 FP8 (``quant_config`` dispatch), CUDA-graph replay with L2-busting weight rotation; plus HBM per stage.

``python -m tests.models.deepseek_v41.bench_moe_dense_g32 --tokens 1 2 4 8 16 64 512 --out <dir>``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

from vllm.models.deepseek_v41 import quant_config as qc
from vllm.models.deepseek_v41.common.contracts import IDX_DIM, IDX_HEADS, Q_LORA
from vllm.models.deepseek_v41.sm70 import fp8_g32

from .test_moe_bench import graph_us

# (call site, N, K, groups, x dtype, out dtype) at TP4
SITES = [
    ("wq_a+wkv", 1792, 5120, 1, torch.float16, torch.float32),
    ("wq_b", 8192, 1280, 1, torch.float16, torch.float16),
    ("wo_a", 2048, 4096, 2, torch.float16, torch.float16),
    ("indexer_wq_b", IDX_HEADS * IDX_DIM, Q_LORA, 1, torch.float32, torch.float32),   # measured; NOT switched
]


class _Lin(nn.Module):
    pass


def _rand(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0x3C
    e = torch.randint(-13, -5, (n // 32, k // 32), device="cuda", generator=g)
    return codes.view(torch.float8_e4m3fn), torch.pow(2.0, e.float()).to(torch.float8_e8m0fnu)


def _layers(n: int, k: int, groups: int, copies: int) -> tuple[list[_Lin], list[_Lin]]:
    fbs, g32s = [], []
    for i in range(copies):
        w, s = _rand(n, k, i)
        fb = _Lin()
        fb.weight = nn.Parameter(qc.dequantize_fp8_block32_exact(w, s), requires_grad=False)
        g = _Lin()
        if groups > 1:
            g.is_bmm, g.bmm_batch_size = True, groups
        qc.DeepseekV41SM70Fp8LinearMethod._keep_fp8_g32(g, w, s)
        fbs.append(fb)
        g32s.append(g)
    return fbs, g32s


def _p2_call(site: str, lin: _Lin, x: torch.Tensor, groups: int) -> torch.Tensor:
    """The P2 call-site code (a45266604) for each projection."""
    w = lin.weight
    if site == "wq_a+wkv":
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    if site == "wq_b":
        return torch.mm(x, w.t())
    if site == "wo_a":
        m, gk = x.shape
        k = gk // groups
        z = torch.bmm(x.view(m, groups, k).transpose(0, 1), w.view(groups, -1, k).transpose(1, 2),
                      out_dtype=torch.float32)
        return z.transpose(0, 1).reshape(m, -1).to(torch.float16)
    return torch.mm(x.float(), w.float().t())          # indexer: compressor.mm_fp32_full


def _g32_call(site: str, lin: _Lin, x: torch.Tensor, groups: int, out_dtype: torch.dtype) -> torch.Tensor:
    if site == "wo_a":
        return qc.v41_grouped_linear(lin, x, groups, out_dtype)
    if site == "indexer_wq_b":
        return qc.v41_linear_fp32_input(lin, x)
    return qc.v41_linear(lin, x, out_dtype)


def bench(tokens: list[int]) -> dict:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    rows = []
    for site, n, k, groups, xdt, odt in SITES:
        copies = max(2, (24 << 20) // (n * k) + 1)
        fbs, g32s = _layers(n, k, groups, copies)
        it = {"i": 0}

        def nxt(pool: list, it: dict = it):
            it["i"] = (it["i"] + 1) % len(pool)
            return pool[it["i"]]

        for m in tokens:
            x = torch.randn(m, groups * k, device="cuda").to(xdt)
            row = {"site": site, "n": n, "k": k, "groups": groups, "m": m,
                   "p2_us": round(graph_us(lambda x=x, s=site, p=fbs, g=groups: _p2_call(s, nxt(p), x, g)), 2),
                   "g32_us": round(graph_us(
                       lambda x=x, s=site, p=g32s, g=groups, o=odt: _g32_call(s, nxt(p), x, g, o)), 2)}
            if site == "indexer_wq_b" and m <= fp8_g32.GEMV_MAX_M:
                row["g32_gemv_fp32x_us"] = round(graph_us(
                    lambda x=x, p=g32s: fp8_g32.fp8_g32_gemv(x, nxt(p).ds41_g32, out_dtype=torch.float32)), 2)
            row["path"] = ("bitwise-sgemm" if site == "indexer_wq_b" else
                           ("gemv" if (groups > 1 and m <= fp8_g32.GEMV_MAX_M) else
                            ("dequant-bmm" if groups > 1 else fp8_g32.fp8_g32_path(m, odt))))
            rows.append(row)
            print(json.dumps(row), flush=True)
        del fbs, g32s
        torch.cuda.empty_cache()
    return {"rows": rows, "hbm": hbm_per_stage()}


def hbm_per_stage() -> dict:
    """Bytes per TP4 rank of the switched projections, FP16 fallback vs g32 (FP8 + FP16 group scales), per PP3 stage
    [0-13][14-27][28-39]. Index sources (own indexer.wq_b): 2, 8, 14, 20, 24, 28, 32, 36."""
    index_layers = {2, 8, 14, 20, 24, 28, 32, 36}

    def layer_bytes(layer: int, g32: bool) -> int:
        tot = 0
        for site, n, k, _, _, _ in SITES:
            if site == "indexer_wq_b" and layer not in index_layers:
                continue
            tot += (n * k + n * k // 32 * 2) if g32 else n * k * 2
        return tot

    out = {}
    for name, (a, b) in {"stage1_L0-13": (0, 13), "stage2_L14-27": (14, 27), "stage3_L28-39": (28, 39)}.items():
        fp16 = sum(layer_bytes(i, False) for i in range(a, b + 1))
        g32 = sum(layer_bytes(i, True) for i in range(a, b + 1))
        out[name] = {"fp16_MiB": round(fp16 / 2**20, 1), "g32_MiB": round(g32 / 2**20, 1),
                     "saved_MiB": round((fp16 - g32) / 2**20, 1)}
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 8, 16, 64, 512, 4096])
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    report = bench(args.tokens)
    report["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(report["hbm"]))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "dense_g32_bench.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
    sys.exit(0)
