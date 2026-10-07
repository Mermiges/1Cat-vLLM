# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F811
"""Decode kernels fed per-layer official golden inputs, at the unchanged §4.5 gates."""

from __future__ import annotations

import pytest
import torch

from vllm.models.deepseek_v41.sm70 import decode_kernels as dk
from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores

from .test_attn_golden import (  # noqa: F401
    PROMPTS,
    Case,
    _ownk_cases,
    _require,
    _weights,
    weights_cache,
)
from .test_attn_harness import ref_config, rel_rms, topology
from .test_attn_layers import DEV, Stage, _overlap, dist_env  # noqa: F401

pytestmark = [pytest.mark.sm70, pytest.mark.weights]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("layer", [0, 1, 2, 3, 14, 20, 21, 24])
def test_decode_ops_official_golden(
    dist_env, weights_cache, prompt: str, layer: int
) -> None:
    cases = _require(prompt, (layer,))
    ownk = _ownk_cases(prompt)
    if ownk is not None:
        cases = ownk
    cfg = ref_config()
    topo = topology(cfg, layer)
    w = _weights(weights_cache, layer)
    st = Stage(cfg, (layer,), {layer: w})
    attn = st.attn[layer]
    for case in cases[1:]:

        def g(name: str, _case: Case = case) -> torch.Tensor:
            return _case.get(layer, name)

        # Norms/projected q from the FP16 inputs actually consumed by the port.
        x = g("attn.x").half()
        qkv = torch.mm(x, attn.fused_wqa_wkv.weight.t(), out_dtype=torch.float32)
        qr32, _, _ = dk.qkv_norm(
            qkv, attn.q_norm.weight, attn.kv_norm.weight, attn.eps, 1280
        )
        assert rel_rms(qr32, g("attn.qr")) <= 2e-3

        # Golden records and picks isolate sparse arithmetic from QAT input rounding.
        q, want = g("attn.q").half().contiguous(), g("attn.o")
        tk = g("attn.topk_all").long()
        win = g("cache.win").half().contiguous()
        ckv = topk = bt = req = None
        if topo.compress_ratio:
            ckv = case.get(topo.kv_source, "cache.ckv").half().contiguous()
            topk = torch.where(tk[:, 128:] >= 0, tk[:, 128:] - win.shape[0], -1).to(
                torch.int32
            )
            bt = (
                torch.arange(
                    max(1, (ckv.shape[0] + 127) // 128), device=DEV, dtype=torch.int32
                )[None]
                .expand(q.shape[0], -1)
                .contiguous()
            )
            req = torch.zeros(q.shape[0], dtype=torch.int32, device=DEV)
        out = torch.empty_like(q)
        dk.sparse_attention_decode(
            q,
            win,
            tk[:, :128].contiguous(),
            ckv,
            topk,
            bt,
            req,
            128,
            attn.attn_sink,
            512**-0.5,
            out,
        )
        assert rel_rms(out, want) <= 2e-3, (prompt, layer, case.start, "sparse")

        if topo.owns_compressor and case.has(layer, "attn.latent"):
            comp = attn.compressor
            if topo.compress_ratio == 1:
                kv = torch.mm(g("attn.x"), comp._weight().float().t())
                latent = dk.rms_rows(kv, comp.norm.weight, comp.eps)
            else:
                # Completed pair's golden FP32 state, without extra rounding.
                kvs, scores = g("comp.kv_state"), g("comp.score_state")
                state = torch.cat([kvs, scores], dim=-1).contiguous()
                ks = state[1:2].clone()
                latent = dk.ratio2_decode(
                    ks,
                    state,
                    torch.tensor([1], device=DEV),
                    torch.tensor([0], device=DEV),
                    comp.norm.weight,
                    comp.eps,
                )
            assert rel_rms(latent, g("attn.latent")) <= 1e-3, (
                prompt,
                layer,
                case.start,
                "latent",
            )

        if topo.owns_compressor and case.has(layer, "idx.k"):
            ix = attn.indexer
            latent = g("attn.latent")
            k = torch.mm(latent, ix.wk.weight.float().t())
            lpos = torch.full(
                (latent.shape[0],),
                case.start + 1 - topo.compress_ratio,
                dtype=torch.int64,
                device=DEV,
            )
            records = torch.empty(latent.shape[0], 128, dtype=torch.float16, device=DEV)
            slots = torch.arange(latent.shape[0], dtype=torch.int64, device=DEV)
            dk.index_k_norm_store(
                k,
                ix.k_norm.weight,
                ix.k_norm.eps,
                lpos,
                attn.rotary_emb.cos_sin_cache,
                records,
                slots,
                None,
            )
            assert torch.equal(records.float(), g("idx.k")), (
                prompt,
                layer,
                case.start,
                "index-K",
            )

        if topo.owns_indexer and case.has(layer, "idx.score"):
            keys = case.get(topo.kv_source, "cache.ik").half().contiguous()
            n = g("idx.score").shape[1]
            if (
                topo.owns_compressor
                and topo.compress_ratio == 2
                and (case.start + 1) % 2
            ):
                # AM-9: official no-pair steps can score another layer's shared K.
                # Identify that source with the already gated general kernel, then
                # test decode arithmetic on the exact golden inputs. The composite
                # golden suite separately gates the port's own-K behavior.
                matches = []
                for source in sorted(case.captured):
                    if not case.has(source, "cache.ik"):
                        continue
                    candidate = case.get(source, "cache.ik")[:n].half().contiguous()
                    if candidate.shape[0] < n:
                        continue
                    score = torch.empty(q.shape[0], n, device=DEV)
                    index_scores(
                        g("idx.q").half().contiguous(), g("idx.w"), candidate, score
                    )
                    if rel_rms(score, g("idx.score")) < 1e-6:
                        matches.append(candidate)
                assert matches, "AM-9 golden score matches no captured K source"
                keys = matches[0]
            bt = torch.arange(max(1, (n + 127) // 128), device=DEV, dtype=torch.int32)[
                None
            ]
            ends = torch.full((q.shape[0],), n, device=DEV, dtype=torch.int64)
            sc = dk.index_scores_decode(
                g("idx.q").half().contiguous(),
                g("idx.w").contiguous(),
                1.0,
                keys,
                bt,
                torch.zeros(q.shape[0], device=DEV, dtype=torch.int32),
                128,
                ends,
                max(512, n),
            )
            assert rel_rms(sc[:, :n], g("idx.score")) <= 1e-3, (
                prompt,
                layer,
                case.start,
                "score",
            )
            picks = torch.empty(q.shape[0], 512, device=DEV, dtype=torch.int32)
            dk.topk_decode(sc, ends, 512, picks)
            mean, minimum = _overlap(picks, g("idx.topk"))
            assert mean >= 0.995 and minimum >= 0.98, (
                prompt,
                layer,
                case.start,
                mean,
                minimum,
            )
