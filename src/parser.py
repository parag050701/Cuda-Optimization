"""
parser.py — CUDA kernel analyzer.

Extracts optimization-relevant parameters from a .cu source file using
regex-based heuristics (Phase 1 approach; libclang integration in Phase 2).

Outputs a KernelProfile used by generator.py to build the search space.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MemoryAccessPattern:
    has_global_load:    bool = False
    has_global_store:   bool = False
    has_shared_mem:     bool = False
    has_coalesced_hint: bool = False    # row-major consecutive access
    has_strided_access: bool = False    # e.g., B[k*N+col] column access
    has_reduction:      bool = False
    has_warp_shuffle:   bool = False


@dataclass
class KernelProfile:
    name:           str
    src_path:       Path
    block_dim:      int                 # current block dimension from source
    uses_shared:    bool
    loop_depth:     int                 # depth of inner loops (unroll candidates)
    reduction_ops:  list[str]           # "+=" patterns inside loops
    memory:         MemoryAccessPattern = field(default_factory=MemoryAccessPattern)
    raw_params:     dict = field(default_factory=dict)  # anything else extracted


# ── Regex patterns ──────────────────────────────────────────────────────────

_RE_KERNEL   = re.compile(r'__global__\s+void\s+(\w+)\s*\(')
_RE_SHARED   = re.compile(r'__shared__')
_RE_BLOCKDIM = re.compile(r'blockDim\.(x|y|z)')
_RE_THREADID = re.compile(r'threadIdx\.(x|y|z)')
_RE_REDUCTION= re.compile(r'\b\w+\s*\+=')
_RE_SYNCTH   = re.compile(r'__syncthreads\s*\(')
_RE_SHFL     = re.compile(r'__shfl')
_RE_STRIDE   = re.compile(r'\[\s*\w+\s*\*\s*\w+\s*\+\s*\w+\s*\]')  # [k*N+col]
_RE_ROWMAJ   = re.compile(r'\[\s*\w+\s*\*\s*\w+\s*\+\s*\w+\s*\]')  # same, col varies
_RE_FOR      = re.compile(r'\bfor\s*\(')
_RE_BLOCK_CONST = re.compile(r'blockDim\.x\s*\*\s*blockDim\.y|dim3\s+block\((\d+)')


def _count_loop_depth(src: str) -> int:
    """Estimate max loop nesting depth by counting nested for-loops."""
    max_depth = depth = 0
    i = 0
    while i < len(src):
        if src[i:i+3] == 'for':
            depth += 1
            max_depth = max(max_depth, depth)
        elif src[i] == '}':
            depth = max(0, depth - 1)
        i += 1
    return max_depth


def _extract_block_dim(src: str) -> int:
    """Try to read a literal block dimension from dim3 or #define."""
    m = re.search(r'dim3\s+block\(\s*(\d+)', src)
    if m:
        return int(m.group(1))
    m = re.search(r'#define\s+BLOCK_SIZE\s+(\d+)', src)
    if m:
        return int(m.group(1))
    return 16  # RTX 2070 safe default


def analyze_file(src_path: Path) -> list[KernelProfile]:
    """Parse a .cu file and return a profile for each __global__ kernel."""
    src = src_path.read_text(encoding="utf-8")
    profiles = []

    # Split on kernel boundaries
    kernel_starts = [(m.group(1), m.start()) for m in _RE_KERNEL.finditer(src)]
    for idx, (name, start) in enumerate(kernel_starts):
        end = kernel_starts[idx+1][1] if idx+1 < len(kernel_starts) else len(src)
        body = src[start:end]

        mem = MemoryAccessPattern(
            has_global_load    = bool(re.search(r'\bA\[|\bin\[|\binput\[', body)),
            has_global_store   = bool(re.search(r'\bC\[|\bout\[|\boutput\[', body)),
            has_shared_mem     = bool(_RE_SHARED.search(body)),
            has_strided_access = bool(_RE_STRIDE.search(body)),
            has_reduction      = bool(_RE_REDUCTION.search(body)),
            has_warp_shuffle   = bool(_RE_SHFL.search(body)),
        )

        reductions = _RE_REDUCTION.findall(body)
        profile = KernelProfile(
            name          = name,
            src_path      = src_path,
            block_dim     = _extract_block_dim(body),
            uses_shared   = mem.has_shared_mem,
            loop_depth    = _count_loop_depth(body),
            reduction_ops = reductions,
            memory        = mem,
        )
        profiles.append(profile)

    return profiles


def build_search_space(profile: KernelProfile) -> dict:
    """
    Given a kernel profile, return the candidate parameter grid.

    Heuristic pruning rules (from the proposal):
      - block_size must be a multiple of 32 (warp size)
      - tile sizes only matter if kernel is not already using shared memory
      - warp shuffle only makes sense for reduction kernels
    """
    space: dict[str, list] = {}

    # Block sizes — always tune these
    space["block_size"] = [64, 128, 192, 256]

    # Unroll factors
    if profile.loop_depth >= 1:
        space["unroll"] = [1, 2, 4, 8]
    else:
        space["unroll"] = [1]

    # Shared memory tile sizes (only if kernel doesn't already use smem)
    if not profile.uses_shared:
        space["tile_x"] = [16, 32]
        space["tile_y"] = [16, 32]
    else:
        space["tile_x"] = [16]
        space["tile_y"] = [16]

    # Memory layout: try transposed access for strided patterns
    if profile.memory.has_strided_access:
        space["transpose_b"] = [False, True]
    else:
        space["transpose_b"] = [False]

    # Warp shuffle for reductions
    if profile.memory.has_reduction:
        space["warp_shuffle"] = [False, True]
    else:
        space["warp_shuffle"] = [False]

    total = 1
    for v in space.values():
        total *= len(v)

    space["_total_variants"] = total
    space["_kernel"]         = profile.name
    return space


if __name__ == "__main__":
    import json, sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
           Path(__file__).parent / "kernels" / "baseline_kernels.cu"

    profiles = analyze_file(path)
    for p in profiles:
        space = build_search_space(p)
        print(f"\nKernel: {p.name}")
        print(f"  block_dim={p.block_dim}  loop_depth={p.loop_depth}")
        print(f"  shared={p.uses_shared}  strided={p.memory.has_strided_access}")
        print(f"  search space: {space['_total_variants']} variants")
        print(f"  params: { {k:v for k,v in space.items() if not k.startswith('_')} }")
