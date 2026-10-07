# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P5-CORE microbench: the HC work of one decoder layer, eager common/hc.py vs fused sm70/hc_kernels.py.

Per layer the eager path runs hc_mixes + hc_pre + rmsnorm (attn), hc_post, hc_mixes + hc_pre + rmsnorm (ffn), hc_post;
the fused path runs two shifted post+pre steps (steady state inside a stage). Reports kernels/layer (torch profiler),
CPU issue time and GPU time per layer (CUDA events, 200 iterations after warm-up).

    CUDA_VISIBLE_DEVICES=<one board-A UUID> PYTHONPATH=<tree> python tests/models/deepseek_v41/bench_core_hc.py [T ...]
"""

from __future__ import annotations

import sys
import time

import torch

from vllm.models.deepseek_v41.common.hc import hc_mixes, hc_post, hc_pre, rmsnorm_to_act
from vllm.models.deepseek_v41.sm70.hc_kernels import hc_fused_step

D = 5120


def _setup(t: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    fn = [torch.randn(24, 4 * D, device="cuda", generator=g) * 0.02 for _ in range(2)]
    scale = [torch.tensor([0.9, 1.3, 2.0], device="cuda") for _ in range(2)]
    base = [torch.randn(24, device="cuda", generator=g) * 0.5 for _ in range(2)]
    w = [torch.rand(D, device="cuda", generator=g) + 0.5 for _ in range(2)]
    stream = (torch.randn(t, 4, D, device="cuda", generator=g) * 3).to(torch.bfloat16)
    sub = torch.randn(t, D, device="cuda", generator=g)
    pre = torch.rand(t, 4, device="cuda", generator=g)
    return fn, scale, base, w, stream, sub, pre


def eager_layer(fn, scale, base, w, stream, sub, pre):
    a_pre, a_post, a_comb = hc_mixes(stream, fn[0], scale[0], base[0])
    rmsnorm_to_act(hc_pre(stream, pre), w[0])
    stream = hc_post(sub, stream, a_post, a_comb)
    f_pre, f_post, f_comb = hc_mixes(stream, fn[1], scale[1], base[1])
    rmsnorm_to_act(hc_pre(stream, a_pre), w[1])
    return hc_post(sub, stream, f_post, f_comb), f_pre


def fused_layer(fn, scale, base, w, stream, sub, pre, pending):
    stream, (a_pre, a_post, a_comb), _, _ = hc_fused_step(stream, sub_out=sub, post=pending[0], comb=pending[1],
                                                          mix=(fn[0], scale[0], base[0]), collapse_pre=pre,
                                                          norm_weight=w[0])
    stream, (f_pre, f_post, f_comb), _, _ = hc_fused_step(stream, sub_out=sub, post=a_post, comb=a_comb,
                                                          mix=(fn[1], scale[1], base[1]), collapse_pre=a_pre,
                                                          norm_weight=w[1])
    return stream, f_pre, (f_post, f_comb)


def _measure(step, iters: int = 200) -> tuple[float, float, int, float]:
    """(CPU issue us, wall GPU us, kernels, summed kernel time us) per layer. Kernel time (profiler, 20 layers)
    does not depend on host load; the two wall-clock numbers do."""
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            step()
        torch.cuda.synchronize()
    kev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    kernels = len(kev) // 20
    kernel_us = sum(e.device_time for e in kev) / 20
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        step()
    end.record()
    cpu_us = (time.perf_counter() - t0) / iters * 1e6
    torch.cuda.synchronize()
    gpu_us = start.elapsed_time(end) / iters * 1e3
    return cpu_us, gpu_us, kernels, kernel_us


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [1, 4, 64, 1024]
    print(f"{'T':>5} | {'eager kern':>10} {'kernel us':>10} {'CPU us':>9} {'wall us':>9} | "
          f"{'fused kern':>10} {'kernel us':>10} {'CPU us':>9} {'wall us':>9}")
    for t in sizes:
        fn, scale, base, w, stream, sub, pre = _setup(t)
        pending = (torch.rand(t, 4, device="cuda") * 2,
                   torch.full((t, 4, 4), 0.25, device="cuda"))
        with torch.inference_mode():
            e = _measure(lambda: eager_layer(fn, scale, base, w, stream, sub, pre))
            f = _measure(lambda: fused_layer(fn, scale, base, w, stream, sub, pre, pending))
        print(f"{t:>5} | {e[2]:>10} {e[3]:>10.1f} {e[0]:>9.1f} {e[1]:>9.1f} | "
              f"{f[2]:>10} {f[3]:>10.1f} {f[0]:>9.1f} {f[1]:>9.1f}")


if __name__ == "__main__":
    main()
