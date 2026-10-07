# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the custom-AR source overlay in this worktree without CUDA contexts.

This validates changes to custom_all_reduce.cuh using its production .cu and
existing complete communicator-lifecycle bindings. Integration still needs _C
rebuilt. Never installs anything into the shared Python environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "build_comm"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
os.environ.setdefault("MAX_JOBS", "4")
if not 1 <= int(os.environ["MAX_JOBS"]) <= 8:
    raise RuntimeError("MAX_JOBS must be in 1..8")
OUTPUT.mkdir(exist_ok=True)

from torch.utils.cpp_extension import load  # noqa: E402

sources = [ROOT / "benchmarks/kernels/sm70_qwen38_custom_ar_sidecar.cu"]
library = load(
    name="vllm_ds41_communication",
    sources=[str(s) for s in sources],
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_cuda_cflags=["-O3", "-DNDEBUG", "-std=c++17"],
    extra_include_paths=[str(ROOT / "csrc")]
    + [p for p in os.environ.get("CPATH", "").split(":") if p],
    extra_ldflags=["-lcuda"],
    build_directory=str(OUTPUT),
    is_python_module=False,
    verbose=True,
)
source_hashes: dict[str, str] = {}
manifest: dict[str, object] = {
    "library": str(library),
    "source_root": str(ROOT),
    "arch": "7.0",
    "sources": source_hashes,
}
for source in sources + [
    ROOT / "csrc/custom_all_reduce.cu",
    ROOT / "csrc/custom_all_reduce.cuh",
]:
    source_hashes[str(source.relative_to(ROOT))] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
manifest["sha256"] = hashlib.sha256(Path(library).read_bytes()).hexdigest()
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
