"""
parser.py — CUDA kernel analyzer.

Extracts optimization-relevant parameters from a .cu source file and
emits a KernelProfile used by generator.py to build the search space.

Two backends:
  - "libclang": AST-based walker (primary, Phase 2).
  - "regex":    pattern-matching heuristics (fallback, Phase 1).

The libclang path is tried first. It falls back to regex when:
  - the `libclang` Python binding is not importable,
  - the env var CUDA_AUTOTUNER_NO_LIBCLANG is set,
  - or the translation unit yields zero CUDA kernels (severe parse failure).

Both paths populate the same KernelProfile shape, with a `backend` tag
recording which analyzer produced the profile.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Public data classes ─────────────────────────────────────────────────────

@dataclass
class MemoryAccessPattern:
    has_global_load:    bool = False
    has_global_store:   bool = False
    has_shared_mem:     bool = False
    has_coalesced_hint: bool = False
    has_strided_access: bool = False
    has_reduction:      bool = False
    has_warp_shuffle:   bool = False


@dataclass
class KernelProfile:
    name:           str
    src_path:       Path
    block_dim:      int
    uses_shared:    bool
    loop_depth:     int
    reduction_ops:  list[str]
    memory:         MemoryAccessPattern = field(default_factory=MemoryAccessPattern)
    raw_params:     dict = field(default_factory=dict)
    backend:        str = "regex"          # "libclang" or "regex"


# ── Base names we treat as global-memory inputs / outputs ──────────────────
# These are heuristic — the compiler can't know from a name alone whether a
# pointer points at global memory, but within this project's baseline
# kernels the convention is consistent.
_LOAD_BASES  = {"A", "B", "in", "input", "ir", "gamma", "beta", "sdata"}
_STORE_BASES = {"C", "out", "output", "or_", "sdata"}


# ────────────────────────────────────────────────────────────────────────────
#  libclang AST backend
# ────────────────────────────────────────────────────────────────────────────

def _try_import_libclang():
    """Return the clang.cindex module, or None if the binding isn't usable."""
    if os.environ.get("CUDA_AUTOTUNER_NO_LIBCLANG"):
        return None
    try:
        from clang import cindex  # provided by the `libclang` PyPI package
    except Exception:
        return None

    lib = os.environ.get("LIBCLANG_LIBRARY_FILE")
    if lib:
        try:
            cindex.Config.set_library_file(lib)
        except Exception:
            pass
    return cindex


# Flags that let libclang's frontend digest a .cu file without needing the
# full CUDA header tree installed on the host running the tuner.
_CUDA_PARSE_ARGS = [
    "-x", "cuda",
    "-std=c++17",
    "--cuda-host-only",
    "-nocudainc",
    "-nocudalib",
    "-ferror-limit=0",
    "-Wno-everything",
]


def _is_cuda_kernel(cursor) -> bool:
    """A FUNCTION_DECL whose signature tokens contain `__global__`."""
    for tok in cursor.get_tokens():
        spell = tok.spelling
        if spell == "__global__":
            return True
        if spell == "(":                    # reached the parameter list
            return False
    return False


class _LibclangVisitor:
    """
    Walk a libclang FUNCTION_DECL subtree and accumulate the fields that
    feed KernelProfile / MemoryAccessPattern.
    """

    def __init__(self, cindex):
        self.cindex = cindex
        self.uses_shared       = False
        self.max_loop_depth    = 0
        self.reduction_count   = 0
        self.has_warp_shuffle  = False
        self.has_strided_access = False
        self.has_global_load    = False
        self.has_global_store   = False

    def visit(self, cursor, loop_depth: int = 0):
        CursorKind = self.cindex.CursorKind
        for child in cursor.get_children():
            new_depth = loop_depth
            if child.kind == CursorKind.FOR_STMT:
                new_depth = loop_depth + 1
                if new_depth > self.max_loop_depth:
                    self.max_loop_depth = new_depth
            self._inspect(child, CursorKind)
            self.visit(child, new_depth)

    def _inspect(self, cursor, CursorKind):
        kind = cursor.kind

        if kind == CursorKind.VAR_DECL:
            for tok in cursor.get_tokens():
                if tok.spelling == "__shared__":
                    self.uses_shared = True
                    break
            return

        if kind == CursorKind.CALL_EXPR:
            name = cursor.spelling or ""
            if name.startswith("__shfl"):
                self.has_warp_shuffle = True
            return

        if kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            children = list(cursor.get_children())
            if len(children) >= 2:
                idx_tokens = [t.spelling for t in children[1].get_tokens()]
                if "*" in idx_tokens and "+" in idx_tokens:
                    self.has_strided_access = True
            if children:
                base_tokens = [t.spelling for t in children[0].get_tokens()]
                if base_tokens:
                    base = base_tokens[0]
                    if base in _LOAD_BASES:
                        self.has_global_load = True
                    if base in _STORE_BASES:
                        self.has_global_store = True
            return


def _count_reductions_from_tokens(cursor) -> int:
    """Scan the token stream of a function body for `+=` occurrences."""
    count = 0
    for tok in cursor.get_tokens():
        if tok.spelling == "+=":
            count += 1
    return count


def _extract_block_dim_from_tu(tu_cursor, kernel_name: str) -> int:
    """
    Walk the whole translation unit for a `dim3 block(N, ...)` literal that
    is launched against this kernel. Falls back to 16 when none is found.

    This is deliberately simple: we look for any integer literal appearing
    immediately after `dim3 block(` in any host function in the file.
    """
    default = 16
    # Cheap: scan all tokens of the TU. The AST gives us ordering; a real
    # implementation would tie the dim3 to a specific <<<>>> launch, but the
    # baseline runner uses consistent block sizes per kernel family.
    tokens = [t.spelling for t in tu_cursor.get_tokens()]
    for i, tok in enumerate(tokens):
        if tok == "dim3" and i + 3 < len(tokens):
            # pattern: dim3 block ( N
            if tokens[i + 1] == "block" and tokens[i + 2] == "(":
                try:
                    return int(tokens[i + 3])
                except ValueError:
                    continue
    return default


def _libclang_analyze(cindex, src_path: Path) -> list[KernelProfile]:
    idx = cindex.Index.create()
    tu  = idx.parse(str(src_path), args=_CUDA_PARSE_ARGS)

    profiles: list[KernelProfile] = []
    CursorKind = cindex.CursorKind

    for cur in tu.cursor.walk_preorder():
        if cur.kind != CursorKind.FUNCTION_DECL:
            continue
        if not cur.is_definition():
            continue
        if not _is_cuda_kernel(cur):
            continue

        v = _LibclangVisitor(cindex)
        v.visit(cur)
        reductions = _count_reductions_from_tokens(cur)

        mem = MemoryAccessPattern(
            has_global_load    = v.has_global_load,
            has_global_store   = v.has_global_store,
            has_shared_mem     = v.uses_shared,
            has_strided_access = v.has_strided_access,
            has_reduction      = reductions > 0,
            has_warp_shuffle   = v.has_warp_shuffle,
        )
        profiles.append(KernelProfile(
            name          = cur.spelling,
            src_path      = src_path,
            block_dim     = _extract_block_dim_from_tu(tu.cursor, cur.spelling),
            uses_shared   = v.uses_shared,
            loop_depth    = v.max_loop_depth,
            reduction_ops = ["+="] * reductions,
            memory        = mem,
            backend       = "libclang",
        ))
    return profiles


# ────────────────────────────────────────────────────────────────────────────
#  Regex fallback backend
# ────────────────────────────────────────────────────────────────────────────

_RE_KERNEL    = re.compile(r'__global__\s+void\s+(\w+)\s*\(')
_RE_SHARED    = re.compile(r'__shared__')
# Match any `+=` occurrence — catches both `sum +=` and `sdata[tid] +=`.
_RE_REDUCTION = re.compile(r'\+=')
_RE_SHFL      = re.compile(r'__shfl')
_RE_STRIDE    = re.compile(r'\[\s*\w+\s*\*\s*\w+\s*\+\s*\w+\s*\]')
_RE_DIM3      = re.compile(r'dim3\s+block\(\s*(\d+)')
_RE_DEFINE    = re.compile(r'#define\s+BLOCK_SIZE\s+(\d+)')


def _regex_count_loop_depth(body: str) -> int:
    """
    Estimate max nesting depth of `for` loops via brace tracking.
    Word-boundary aware so substrings like "before" don't match.
    Imperfect on braceless loop bodies — acceptable for the fallback path.
    """
    max_depth = depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == "}":
            depth = max(0, depth - 1)
        prev_ok = (i == 0) or not (body[i - 1].isalnum() or body[i - 1] == "_")
        if (prev_ok and body[i:i + 3] == "for"
                and i + 3 < n and body[i + 3] in " ("):
            depth += 1
            if depth > max_depth:
                max_depth = depth
            i += 3
            continue
        i += 1
    return max_depth


def _regex_extract_block_dim(src: str) -> int:
    m = _RE_DIM3.search(src)
    if m:
        return int(m.group(1))
    m = _RE_DEFINE.search(src)
    if m:
        return int(m.group(1))
    return 16


def _regex_analyze(src_path: Path) -> list[KernelProfile]:
    src = src_path.read_text(encoding="utf-8")
    profiles: list[KernelProfile] = []

    kernel_starts = [(m.group(1), m.start()) for m in _RE_KERNEL.finditer(src)]
    for idx, (name, start) in enumerate(kernel_starts):
        end = kernel_starts[idx + 1][1] if idx + 1 < len(kernel_starts) else len(src)
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
        profiles.append(KernelProfile(
            name          = name,
            src_path      = src_path,
            block_dim     = _regex_extract_block_dim(src),
            uses_shared   = mem.has_shared_mem,
            loop_depth    = _regex_count_loop_depth(body),
            reduction_ops = ["+="] * len(reductions),
            memory        = mem,
            backend       = "regex",
        ))
    return profiles


# ────────────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────────────

def analyze_file(src_path: Path, verbose: bool = False) -> list[KernelProfile]:
    """Parse `src_path` and return one KernelProfile per __global__ kernel."""
    cindex = _try_import_libclang()
    if cindex is not None:
        try:
            profiles = _libclang_analyze(cindex, src_path)
            if profiles:
                if verbose:
                    print(f"[parser] libclang: {len(profiles)} kernel(s) "
                          f"from {src_path.name}")
                return profiles
            if verbose:
                print(f"[parser] libclang returned 0 kernels for "
                      f"{src_path.name}; falling back to regex.")
        except Exception as e:
            if verbose:
                print(f"[parser] libclang error on {src_path.name}: {e}; "
                      f"falling back to regex.")

    return _regex_analyze(src_path)


def find_kernel(src_path: Path, kernel: str,
                verbose: bool = False) -> Optional[KernelProfile]:
    """
    Return the first profile whose function name starts with `kernel`.
    Example: find_kernel(path, "matmul") → profile for `matmul_naive`.
    """
    for p in analyze_file(src_path, verbose=verbose):
        if p.name.startswith(kernel):
            return p
    return None


def build_search_space(profile: KernelProfile) -> dict:
    """
    Given a kernel profile, return the candidate parameter grid.

    Pruning rules:
      - tile sizes only matter if the kernel doesn't already use shared memory
      - transpose_b only helps when strided access was detected
      - warp_shuffle only helps when reductions are present
      - reg_tile (1xRT per-thread register blocking) is only added for matmul
    """
    space: dict[str, list] = {}
    is_matmul = profile.name.startswith("matmul")

    # block_size drives 1D thread block for softmax/reduction/layernorm.
    # Matmul derives its 2D block side from tile_x instead, so we skip it.
    if not is_matmul:
        space["block_size"] = [64, 128, 192, 256]

    space["unroll"] = [1, 2, 4, 8] if profile.loop_depth >= 1 else [1]

    # tile_x / tile_y are only consumed by the matmul templates (both
    # reg_tile=1 and the regblock variants derive BLOCK from tile_x).
    if is_matmul:
        space["tile_x"] = [16, 32]
        space["tile_y"] = [16, 32]

    # transpose_b: only matmul's template reads it. Keep True variant only
    # when we also detected strided access.
    if is_matmul and profile.memory.has_strided_access:
        space["transpose_b"] = [False, True]
    else:
        space["transpose_b"] = [False]

    # warp_shuffle: only the reduction template branches on it. For other
    # kernels the flag would just duplicate every variant.
    if profile.name.startswith("reduction") and profile.memory.has_reduction:
        space["warp_shuffle"] = [False, True]
    else:
        space["warp_shuffle"] = [False]

    # Register blocking: only makes sense for matmul-shaped kernels. We key
    # off the function name since the effect (RT outputs per thread) is
    # specific to how matmul_opt is templated.
    if is_matmul:
        space["reg_tile"] = [1, 2, 4]

    total = 1
    for v in space.values():
        total *= len(v)
    space["_total_variants"] = total
    space["_kernel"]         = profile.name
    return space


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
           Path(__file__).parent / "kernels" / "baseline_kernels.cu"

    profiles = analyze_file(path, verbose=True)
    if not profiles:
        print(f"No kernels found in {path}")
        sys.exit(1)

    for p in profiles:
        space = build_search_space(p)
        print(f"\nKernel: {p.name}   [backend={p.backend}]")
        print(f"  block_dim={p.block_dim}  loop_depth={p.loop_depth}")
        print(f"  shared={p.uses_shared}  strided={p.memory.has_strided_access}  "
              f"reduction={p.memory.has_reduction}  shfl={p.memory.has_warp_shuffle}")
        print(f"  reduction_ops={p.reduction_ops}")
        print(f"  search space: {space['_total_variants']} variants")
        print(f"  params: { {k:v for k,v in space.items() if not k.startswith('_')} }")
