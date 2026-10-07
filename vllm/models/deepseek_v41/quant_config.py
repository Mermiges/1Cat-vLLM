# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization config for DeepSeek-V4.1 on SM70 (PORT_DESIGN §0 A2, §1 "Dense FP8 weights", §2.2).

The official checkpoint stores every dense linear as FP8 E4M3 with UE8M0 scales per 32x32 block and the routed
experts as MXFP4. On Volta:

* every dense FP8 linear is dequantized to FP16 once at load (1Cat's SM70 dequant fallback,
  ``Fp8LinearMethod._dequantize_block_weight``) and run as an FP16 GEMM -- forced here, independent of the
  ``VLLM_SM70_FP8_*`` environment (TurboMind group-32 kernels are a P5 item). Dense scale exponents are
  -13..-6, so every E4M3 x 2^s value is an FP16 value; the load path verifies that bit-exactly and raises
  otherwise (no silent flush to zero);
* grouped (``is_bmm``) linears -- ``wo_a`` -- use the dequantized einsum path;
* FusedMoE layers get L-MOE's ``vllm.models.deepseek_v41.sm70.moe.make_v41_moe_method`` (imported lazily);
* Engram ``wkv`` must not come through here: its scales reach 2^-18 and it is loaded with a 2^10 bias by
  ``DeepseekV41Engram`` itself (quant_config=None). Building it through this config raises.

P5 (D9/D19, lane P5-MOE): with ``VLLM_DS41_MOE_DENSE_G32`` on (default: see ``DENSE_G32_DEFAULT``) the attention
projections of ``G32_PROJECTIONS`` (wq_a+wkv, wq_b, wo_a) stay FP8 in TurboMind's group-32 layout (``sm70/fp8_g32.py``: 1 B/param + an FP16
scale per 32 K-values of each row instead of 2 B/param) and run through ``v41_linear`` / ``v41_grouped_linear`` /
``v41_linear_fp32_input``, which the call sites use for every dense projection (FP16-weight layers take the P2
products there, unchanged). ``wo_b`` and the shared expert stay on the FP16 fallback (FP32 outputs above the GEMV
range would need an FP32-output TurboMind epilogue). A layer whose scale exponents leave the exact window
[-14, 7] (none in the official checkpoint: -13..-6) stays on the fallback with a warning.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter
from vllm.models.deepseek_v41 import knobs

logger = init_logger(__name__)

QUANT_METHOD_NAME = "deepseek_v41_fp8"
V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
V41_WEIGHT_BLOCK = [32, 32]

DENSE_G32_KNOB = "VLLM_DS41_MOE_DENSE_G32"
DENSE_G32_DEFAULT = True
# The attention call-site seam is integrated (dc5353555). SOL-MOE verified
# all 112 HANDOFF tests, including 39 g32 and 41 attention golden gates.
# The switched set wins at M <= 8 on board B (180 W; P5-MOE measurements).
# layer-prefix suffixes kept FP8 (group 32) when the knob is on; everything else keeps the FP16 fallback.
# attn.indexer.wq_b is NOT switched: its FP32-input product must stay bitwise the fallback's SGEMM (L-ATTN idx.q gate);
# the FP32-x GEMV flips 1e-6..3e-6 of the QAT'd q values on the goldens, and the bitwise path (row dequant + SGEMM)
# is slower than the fallback (177 vs 83 us at M = 1) for 5 MB/index layer of HBM.
G32_PROJECTIONS = ("attn.fused_wqa_wkv", "attn.wq_b", "attn.wo_a")


def dense_g32_enabled() -> bool:
    return knobs.env_bool(DENSE_G32_KNOB, DENSE_G32_DEFAULT)


def g32_projection(prefix: str) -> bool:
    return any(prefix == p or prefix.endswith("." + p) for p in G32_PROJECTIONS)


def dequantize_fp8_block32_exact(weight: torch.Tensor, scale: torch.Tensor,
                                 block: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """E4M3 [N, K] x UE8M0 [ceil(N/bn), ceil(K/bk)] -> FP16 [N, K], verified bit-exact.

    The product is formed in FP32 (exact: 4 significant bits x a power of two) and rounded once to FP16; the
    round trip FP16 -> FP32 must reproduce it for every element, otherwise some value under/overflowed FP16 and
    the load fails loudly (A2: dense scale exponents -13..-6 keep every value exact).
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn weight, got {weight.dtype}")
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"expected 2-D weight/scale, got {tuple(weight.shape)} / {tuple(scale.shape)}")
    bn, bk = block
    n, k = weight.shape
    if scale.shape[0] != -(-n // bn) or scale.shape[1] != -(-k // bk):
        raise ValueError(f"scale shape {tuple(scale.shape)} does not tile weight {tuple(weight.shape)} by {block}")
    scale32 = scale.to(torch.float32)
    expanded = scale32.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[:n, :k]
    exact = weight.to(torch.float32) * expanded
    out = exact.to(torch.float16)
    mismatch = out.to(torch.float32) != exact
    if bool(mismatch.any()):
        bad = int(mismatch.sum())
        raise ValueError(
            f"FP8 32x32 block weight is not exactly representable in FP16 ({bad} of {exact.numel()} values "
            f"change; scale range [{float(scale32.min())}, {float(scale32.max())}]). PORT_DESIGN A2 assumes "
            "dense scale exponents -13..-6; a layer outside that range needs a scale bias (as Engram wkv).")
    return out.contiguous()


class DeepseekV41SM70Fp8LinearMethod(Fp8LinearMethod):
    """Fp8LinearMethod pinned to the dequant-to-FP16 fallback (A2), with the grouped wo_a einsum; ``use_g32``
    keeps the weight FP8 (group 32, ``sm70/fp8_g32.py``) instead (P5)."""

    def __init__(self, quant_config: DeepseekV41FP8Config, use_g32: bool = False):
        super().__init__(quant_config)
        if self.weight_block_size != V41_WEIGHT_BLOCK:
            raise ValueError(f"DeepSeek-V4.1 dense FP8 expects weight_block_size {V41_WEIGHT_BLOCK}, "
                             f"got {self.weight_block_size}")
        self.use_sm70_dequant_fallback = True
        self.use_sm70_fp8_turbomind = False
        self.use_marlin = False
        self.use_g32 = use_g32

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, weight_scale_inv = process_fp8_weight_block_strategy(layer.weight, layer.weight_scale_inv)
        if weight.shape != layer.weight.shape:
            raise ValueError(f"FP8 weight padding changed the shape {tuple(layer.weight.shape)} -> "
                             f"{tuple(weight.shape)}; the V4.1 fallback does not support padded weights")
        layer.input_scale = None
        if self.use_g32 and self._g32_eligible(layer, weight_scale_inv):
            self._keep_fp8_g32(layer, weight, weight_scale_inv)
            return
        dequant = dequantize_fp8_block32_exact(weight, weight_scale_inv, tuple(self.weight_block_size))
        replace_parameter(layer, "weight", dequant)
        if getattr(layer, "is_bmm", False):
            layer.dequantized_bmm = True

    @staticmethod
    def _g32_eligible(layer: torch.nn.Module, scale: torch.Tensor) -> bool:
        from vllm.models.deepseek_v41.sm70 import fp8_g32

        lo, hi = fp8_g32.scale_exponent_range(scale)
        if lo >= fp8_g32.SCALE_EXP_MIN and hi <= fp8_g32.SCALE_EXP_MAX:
            return True
        logger.warning("%s: FP8 scale exponents %d..%d leave the exact group-32 window [%d, %d]; this layer keeps "
                       "the FP16 fallback", getattr(layer, "prefix", "?"), lo, hi, fp8_g32.SCALE_EXP_MIN,
                       fp8_g32.SCALE_EXP_MAX)
        return False

    @staticmethod
    def _keep_fp8_g32(layer: torch.nn.Module, weight: torch.Tensor, scale: torch.Tensor) -> None:
        from vllm.models.deepseek_v41.sm70 import fp8_g32

        packed = fp8_g32.prepare_fp8_g32(weight, scale)
        groups = int(layer.bmm_batch_size) if getattr(layer, "is_bmm", False) else 1
        if packed.n % (groups * fp8_g32.PANEL):
            raise ValueError(f"{getattr(layer, 'prefix', '?')}: {groups} groups do not split N={packed.n} into "
                             "32-row panels")
        # the parameters now hold the TurboMind operands (uint8 [K, N] codes, FP16 [K/32, N] scales): a call site
        # that still reads ``.weight`` as FP16 fails loudly instead of multiplying bytes
        replace_parameter(layer, "weight", packed.tm_weight)
        replace_parameter(layer, "weight_scale_inv", packed.tm_scales)
        layer.ds41_g32 = packed
        layer.ds41_g32_groups = groups

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(layer, "ds41_g32", None) is not None:
            if bias is not None:
                raise NotImplementedError("DeepSeek-V4.1 dense projections have no bias")
            groups = int(layer.ds41_g32_groups)
            if groups > 1:
                lead = x.shape[:-2]
                out = v41_grouped_linear(layer, x.reshape(-1, groups * x.shape[-1]), groups, torch.float16)
                return out.view(*lead, groups, -1)
            lead = x.shape[:-1]
            return v41_linear(layer, x.reshape(-1, x.shape[-1]), torch.float16).view(*lead, -1)
        if getattr(layer, "dequantized_bmm", False):
            # x ends in [groups, K]; group g multiplies rows g*R:(g+1)*R (as Fp8LinearMethod.apply).
            group_count = int(layer.bmm_batch_size)
            weight = layer.weight.view(group_count, -1, x.shape[-1])
            out = torch.einsum("...gk,grk->...gr", x, weight)
            if bias is not None:
                out.add_(bias.view(group_count, -1))
            return out
        return torch.nn.functional.linear(x, layer.weight, bias)


@register_quantization_config(QUANT_METHOD_NAME)
class DeepseekV41FP8Config(Fp8Config):
    """Claims DeepSeek-V4.1 FP8 checkpoints (model_type deepseek_v41 / deepseek_v41_text)."""

    is_scale_e8m0 = True

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return QUANT_METHOD_NAME  # type: ignore[return-value]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg: Any, user_quant: str | None,
                                     hf_config: Any = None) -> QuantizationMethods | None:
        if not (isinstance(hf_quant_cfg, dict)
                and hf_quant_cfg.get("quant_method") in ("fp8", QUANT_METHOD_NAME)):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type in V41_MODEL_TYPES or user_quant == QUANT_METHOD_NAME:
            return QUANT_METHOD_NAME  # type: ignore[return-value]
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DeepseekV41FP8Config:
        block = config.get("weight_block_size")
        if block != V41_WEIGHT_BLOCK:
            raise ValueError(f"DeepSeek-V4.1 FP8 checkpoint must use weight_block_size {V41_WEIGHT_BLOCK}, "
                             f"got {block}")
        if config.get("scale_fmt", "ue8m0") != "ue8m0":
            raise ValueError(f"DeepSeek-V4.1 FP8 checkpoint must use scale_fmt ue8m0, got {config['scale_fmt']}")
        expert_dtype = config.get("expert_dtype", "fp4")
        if expert_dtype != "fp4":
            raise ValueError(f"the SM70 DeepSeek-V4.1 port supports expert_dtype fp4 only, got {expert_dtype!r}")
        fp8 = super().from_config(config)
        assert isinstance(fp8, DeepseekV41FP8Config)
        return fp8

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if ".engram." in f".{prefix}.":
                raise ValueError(
                    f"{prefix}: Engram linears must be built with quant_config=None and load their FP8 weight "
                    "with the 2^10 scale bias (PORT_DESIGN §4.1); the plain FP16 dequant would flush 23% of "
                    "the Engram wkv blocks")
            if is_layer_skipped(prefix=prefix, ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping,
                                match_mode=self.ignored_layers_match_mode):
                return UnquantizedLinearMethod()
            return DeepseekV41SM70Fp8LinearMethod(self, use_g32=dense_g32_enabled() and g32_projection(prefix))
        if isinstance(layer, FusedMoE):
            from vllm.models.deepseek_v41.sm70.moe import make_v41_moe_method

            return make_v41_moe_method(layer.moe_config)
        return super().get_quant_method(layer, prefix)


# ------------------------------------------------------------------------------------------------ call-site dispatch
# Every dense projection of the attention stack goes through these; FP16-weight layers (fallback, quant_config None in
# unit tests) run exactly the P2 products, g32 layers the group-32 FP8 kernels (FP32 accumulation on every path).


def _fp16_weight_of(lin: torch.nn.Module, k: int) -> torch.Tensor:
    w = lin.weight
    if w.dtype != torch.float16 or w.ndim != 2 or w.shape[1] != k:
        raise RuntimeError(f"{getattr(lin, 'prefix', type(lin).__name__)}: expected an FP16 [N, {k}] weight (dense "
                           f"FP8 dequantised at load, PORT_DESIGN A2) or a group-32 FP8 layer, got {w.dtype} "
                           f"{tuple(w.shape)}")
    return w


def v41_linear(lin: torch.nn.Module, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """x [M, K] fp16 @ W^T -> [M, N] in ``out_dtype`` (float16 or float32), FP32 accumulation."""
    packed = getattr(lin, "ds41_g32", None)
    if packed is not None:
        from vllm.models.deepseek_v41.sm70.fp8_g32 import fp8_g32_linear

        if int(lin.ds41_g32_groups) != 1:
            raise RuntimeError(f"{getattr(lin, 'prefix', '?')} is grouped: use v41_grouped_linear")
        return fp8_g32_linear(x, packed, out_dtype=out_dtype)
    w = _fp16_weight_of(lin, x.shape[1])
    if out_dtype == torch.float32:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    if out_dtype == torch.float16:
        return torch.mm(x, w.t())
    raise TypeError(f"v41_linear: out_dtype {out_dtype} not supported")


def v41_grouped_linear(lin: torch.nn.Module, x: torch.Tensor, groups: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Block-diagonal projection (``wo_a``): x [M, groups * K] fp16 (group g = columns g*K..) -> [M, groups * Ng]
    with block g = x_g @ W_g^T, FP32 accumulation, ONE rounding to ``out_dtype`` (the P2 bmm's FP32 output cast)."""
    packed = getattr(lin, "ds41_g32", None)
    if packed is not None:
        from vllm.models.deepseek_v41.sm70.fp8_g32 import fp8_g32_grouped

        if int(lin.ds41_g32_groups) != groups:
            raise RuntimeError(f"{getattr(lin, 'prefix', '?')}: prepared for {lin.ds41_g32_groups} groups, "
                               f"called with {groups}")
        return fp8_g32_grouped(x, packed, groups, out_dtype=out_dtype)
    m = x.shape[0]
    k = x.shape[1] // groups
    w = _fp16_weight_of(lin, k)
    wg = w.view(groups, -1, k)
    z = torch.bmm(x.view(m, groups, k).transpose(0, 1), wg.transpose(1, 2), out_dtype=torch.float32)  # [g, M, Ng]
    return z.transpose(0, 1).reshape(m, -1).to(out_dtype)


def v41_linear_fp32_input(lin: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """FP32 x [M, K] @ W^T -> FP32 [M, N] as an FP32 SGEMM (indexer q: no rounding before its FP4 QAT; L-ATTN's
    bitwise idx.q gate). fp16 x -> the HMMA product with FP32 output. A g32 layer rebuilds the exact FP16 weight
    (bitwise the fallback's) for the product, so the result is bitwise the fallback's."""
    if x.dtype == torch.float16:
        return v41_linear(lin, x, torch.float32)
    if x.dtype != torch.float32:
        raise TypeError(f"v41_linear_fp32_input: x must be float32 or float16, got {x.dtype}")
    packed = getattr(lin, "ds41_g32", None)
    if packed is not None:
        from vllm.models.deepseek_v41.sm70.fp8_g32 import dequant_fp8_g32_rows

        # FP32 [N, K]: the fallback's ``weight.float()`` value for value and stride for stride -> same SGEMM
        return torch.mm(x, dequant_fp8_g32_rows(packed, torch.float32).t())
    return torch.mm(x, _fp16_weight_of(lin, x.shape[1]).float().t())
