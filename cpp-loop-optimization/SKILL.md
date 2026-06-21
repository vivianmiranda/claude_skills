---
name: cpp-loop-optimization
description: A catalog and decision guide for optimizing hot C and C++ loops — SIMD/vectorization, cache locality, tiling/blocking, and the full family of loop transformations (fusion, fission, unrolling, interchange, peeling, strip-mining, sentinel, skewing, and more). Use this skill whenever the user is making a numeric or data-processing loop faster, mentions a loop showing up in a CPU profile, asks about vectorization, `__restrict__`, cache misses, matrix-multiply or stencil performance, low-latency code (trading/audio/game loops), or wants to know which loop transformation applies — even if they just say "this loop is slow" or "speed this up" without naming a technique. Also trigger for nested-loop memory-access problems, AoS-vs-SoA layout questions, and reading or fixing why the compiler "won't vectorize."
---

# C/C++ Loop Optimization

The innermost loop of a hot path is often the entire performance story — in trading
engines, audio pipelines, and game loops it can be 90%+ of the profile. The compiler is
good, but it cannot read your intent and it cannot rescue a cache-hostile access pattern
with optimization flags alone. This skill is a catalog of the loop transformations worth
knowing, plus a discipline for applying them without breaking correctness.

It is built from a 23-technique survey (see `references/techniques.md` for the full catalog
with before/after code) but reorganized around two questions you actually ask in practice:
*what is the compiler already doing for me*, and *which transformation does this loop need*.

## First principles (read before touching a loop)

These mirror the discipline in the `cosmolike-dev` skill, because they are what separate a
real speedup from a lucky-looking microbenchmark.

1. **Measure, never guess.** Start from a profile (`perf record` / a sampling profiler),
   not intuition. Optimize the loop that is actually hot. Confirm every change with a
   repeated benchmark (`perf stat -r 5`), not a single timing on a shared machine.
2. **Correctness first.** A faster loop that changes the answer is a bug, not an
   optimization, until the change is deliberate and documented. This is sharpest for
   floating point: reordering a sum (multiple accumulators, vectorized reductions) changes
   the rounding and therefore the result. See `references/verification-and-traps.md`.
3. **Algorithms first, layout second, SIMD last.** The biggest wins come from doing less
   work and from touching memory in the order it is stored — not from intrinsics. Vectorize
   a loop only after its algorithm and access pattern are already good. The "layout" layer
   starts with **interchange** (make the innermost loop stride over contiguous memory — the
   single most preferable transformation here, and what makes the rest pay off), then
   **tiling/blocking** for reuse; **strip mining** delivers the "SIMD-prep". See the priority
   note below — these outrank almost everything else on large data.
4. **Help the compiler instead of replacing it.** At `-O2`/`-O3` the compiler already does
   hoisting, strength reduction, normalization, much unrolling, and often vectorization and
   interchange. Most "manual" wins are really about *removing the obstacles* that stop it:
   pointer aliasing, unknown trip counts, branches in the body, non-contiguous access.
5. **Keep an escape hatch.** Guard any non-obvious transformation so the simple reference
   version still compiles (`#ifdef`, a `_ref` variant). The simple version is your ground
   truth when results drift.

## The workflow

```
profile → find the hot loop → ask "what is the compiler missing?" → pick ONE transformation
        → verify it applied (compiler report / asm) → re-measure → check the answer is unchanged
```

Change one thing at a time. Bundling two transformations means you cannot attribute the win
(or the regression) to either.

## What the compiler usually already does at -O2/-O3

Do not hand-write these unless a compiler report or the assembly shows it gave up:

- **Loop-invariant code motion (hoisting)** — but not across virtual calls, `volatile`,
  or anything it cannot prove invariant.
- **Strength reduction** of induction variables (turning `i*stride` into an add).
- **Loop normalization** (rewriting to start at 0, step 1).
- **Unrolling** of small loops; **fusion** and **interchange** in simple cases.
- **Auto-vectorization** — *if* you remove aliasing, branches, and unknown bounds.

Your leverage is mostly in the things it cannot prove or cannot see: data layout, aliasing
guarantees, trip-count visibility, and algorithmic structure.

## Reach for these first: contiguous access, then tiling and strip mining

**The single most preferable transformation, before anything else, is getting the memory access
pattern right — make the innermost loop stride over contiguous memory (loop interchange, #16).**
It is cheap, almost always safe, frequently a multiple-x win on its own, and — the key point — it
is the prerequisite that makes every other memory and SIMD optimization actually pay off. A tiled
or vectorized loop over a column-strided array still thrashes the cache; fix the stride first.
Whenever an inner loop touches non-contiguous memory, this is the first thing to suggest.

Then, on data larger than the caches (where memory traffic, not arithmetic, is the bottleneck),
the two highest-leverage levers — **weight them above every other technique once the access
pattern is right** — are:

- **Tiling / blocking (#9)** — the default move for nested loops over large matrices or grids.
  It restructures the traversal so a cache-sized sub-block is reused before eviction, turning
  capacity misses into hits. This is where the 10–50× wins live (a 1024×1024 matmul dropping
  ~14× from tiling alone). Whenever the working set exceeds L1/L2, reach for this first — but
  fix the access pattern *before* sizing tiles (for matmul, reorder to `ikj` so the inner loop is
  contiguous; that also drops one operand to a scalar, so only two tiles need to fit). See #9 for
  the honest sizing rule.
- **Strip mining (#18)** — the default move for a 1D numeric loop you want to hand to SIMD. It
  splits the loop into fixed-size strips so the inner kernel has a compile-time trip count the
  compiler can fully unroll and vectorize with no messy remainder, and it gives you explicit
  control over the working-set size. It is the foundation for both vectorization and 1D cache
  blocking — the bridge between a plain loop and an efficient SIMD kernel.

**Order of operations: fix the access pattern (interchange) first, then tile and/or strip-mine.**
They compose: strip-mine a 1D stream to feed SIMD; tile a 2D/nested computation to fit cache. The
matmul story in #9 is the canonical example — the `ikj` reorder (interchange) is what makes the
tiling worth doing, and it halves the tile footprint as a bonus. Most other entries in this
catalog are refinements around these. When in doubt about a hot numeric loop on large data, the
answer is usually "make the inner loop contiguous, then block it (tile) and/or strip-mine it."

## Decision table — pick the technique

Green ("Yes") = reach for it by default. Red ("No") = avoid. The rest are situational. The ★ rows
(interchange first, then tiling and strip mining) are the priority recommendations from the note
above — interchange is the foundation; the other two are the big levers built on top of it.

| Technique | Primary benefit | Complexity | Use often? | When to use |
|---|---|---|---|---|
| **★ Interchange (contiguous access)** | Memory access pattern | Low | **Yes — do first** | The foundational move: make the inner loop stride contiguous memory. Prerequisite for tiling & vectorization |
| **Vectorization** | Throughput (4–16×) | Low | **Yes** | Numeric arrays, no aliasing. Add `__restrict__` to help the compiler |
| **Early exit / continue** | Avg-case latency | Trivial | **Yes** | Search loops, sparse processing. `break`/`return`/`continue` as soon as possible |
| **Hoisting (code motion)** | Remove redundant work | Trivial | **Yes** | Any loop-invariant expr: `strlen`, `size()`, virtual calls, `sqrt(const)` |
| **Fusion** | Cache locality | Low | **Yes** | Two loops, same range, no cross-iteration dependency |
| **Unrolling** | Reduce overhead / ILP | Low | Sometimes | Short hot loops where profiling shows loop-control overhead |
| **★ Tiling / blocking** | Cache utilization (10–50×) | Medium | **Yes — priority** | Nested loops on large matrices/grids; working set exceeds L1/L2. Reach for this first on large data |
| **Fission (distribution)** | Enable vectorization | Low | When fusion hurts | Body mixes vectorizable work with branches or calls |
| **Peeling / splitting** | Remove branches | Low | Sometimes | First/last iters are special-cased, or a known mid-range boundary |
| **Sentinel** | Reduce branch count | Low | Tight searches | Linear search on a writable array with one spare slot at the end |
| **★ Strip mining** | Fixed trip count → vectorize | Medium | **Yes — priority** | Default move for feeding a 1D numeric loop to SIMD; 1D cache-blocking. Reach for this first when vectorizing a stream |
| **Coalescing / collapsing** | Simplify nested loops | Low | Sometimes | Contiguous 2D array, no row dependency; enables flat vectorization |
| **Strength reduction** | Cheaper arithmetic | Trivial | Compiler does it | Non-unit stride or index exprs the compiler misses |
| **Perforation** | Approximate speedup | Low | Rare | Approximate workloads only: level meters, ML, graphics |
| **Skewing** | Expose parallelism | High | Stencils | Wavefront parallelism in stencils; use a polyhedral tool (Pluto) |
| **Reversal** | Nothing useful | — | **No** | Only if the algorithm truly needs reverse order (e.g. stack pop) |

Two more from the catalog that the table omits: **spreading** (OpenMP across cores — only
when per-iteration cost exceeds scheduling overhead), **interleaving** and **reordering**
(software-pipeline two independent computations, or Morton/Z-order traversal for 2D/3D
locality). All 23 are written up with code in `references/techniques.md`.

## How to apply, by tier

**Tier 0 — the access pattern, then the big levers (suggest these first, in this order):**
First and most preferable: **interchange / contiguous access** — make the innermost loop stride
over contiguous memory. It is the prerequisite for everything below, and often a multiple-x win by
itself. Then, on large data: **tiling/blocking** when the working set exceeds L1/L2 (nested loops,
matrices, grids), and **strip mining** when preparing a 1D numeric loop for SIMD. On anything
memory-bound these dominate the payoff; do not let them fall behind the "cheap and easy" tier just
because they take a few more lines. See the "Reach for these first" note above.

**Tier 1 — do these almost always (cheap, safe, high payoff):**
vectorization (remove aliasing with `__restrict__`, keep the body branch-free), early
exit / `continue`, hoisting loop-invariants, fusion of adjacent same-range loops.

**Tier 2 — the rest of cache & memory:**
fission to separate vectorizable arithmetic from branchy work.

**Tier 3 — situational (profile-driven, measure before and after):**
unrolling with independent accumulators, peeling/splitting to drop a per-iteration branch,
coalescing, sentinel search, perforation (approximate only), OpenMP spreading, interleaving.

**Tier 4 — advanced / specialist:**
skewing (stencils; reach for Pluto or a polyhedral compiler), Morton-order reordering for
large 2D/3D grids.

**Avoid:** loop reversal — the historical "compare against zero" trick is irrelevant on
modern x86 and it defeats the forward hardware prefetcher.

## Correctness traps and where the survey overstates its case

A few of the survey's examples need caveats — covered in full in
`references/verification-and-traps.md`. The three that bite hardest:

- **Multiple accumulators change the floating-point result — decide against your tolerance.**
  Unrolling a `float`/`double` reduction into independent partial sums (and vectorizing a
  reduction at all) reorders the additions, which are not associative, so it differs from a
  strict sequential sum — usually in the last bit or two, and usually *more* accurate, not less.
  This is why the compiler won't do it without opt-in (`-ffast-math`, `-fassociative-math`, or a
  per-loop `#pragma omp simd reduction`). It is a problem only if it crosses your reproducibility
  bar: under a "match the reference to printed precision" standard it is normally fine and
  expected (this is what `cosmolike-dev` does); only a strict bit-for-bit requirement rules it out.
- **"Early exit" can silently change semantics.** The survey's "before" search returns the
  *last* match; its "after" returns the *first*. That is a different function, not just a
  faster one. Pick first-vs-last per spec, then exit early within that choice.
- **Tile-size arithmetic is optimistic, and a cubic tile fixes the wrong thing.** A 64×64 `float`
  tile is 16 KB; a cubic-tiled matmul keeps A, B, and C tiles live at once (~48 KB), overflowing a
  32 KB L1. The fix is two-part: reorder to `ikj` so the inner loop is contiguous (which also
  drops A to a scalar, leaving two live tiles), then size to `2·TILE²·sizeof(float)` with headroom
  and tune by measuring. Code and sizing rule in `references/techniques.md` #9.

## Verifying a transformation actually happened

A transformation you cannot confirm is a guess. Before/after measurement plus a compiler
report or the assembly. The exact flags (`-fopt-info-vec` / `-Rpass=loop-vectorize`,
`-fopt-info-vec-missed` / `-Rpass-missed`, Godbolt, `perf stat`) are in
`references/verification-and-traps.md`. Read it before reporting any speedup.

## Reference files

- `references/techniques.md` — the full 23-technique catalog. Each entry: what it is, the
  hardware reason it helps, when to use it, before/after C++ code, and pitfalls. Read this
  when you need the concrete code for a specific transformation.
- `references/verification-and-traps.md` — how to confirm vectorization, how to benchmark
  honestly, the floating-point reassociation trap, and the specific places the source survey
  overstates its case. Read this before claiming a loop was sped up or vectorized.
