# Verifying optimizations & correctness traps

A loop optimization you cannot *confirm* is a guess, and several of the survey's techniques are
correctness hazards if applied mechanically. This file is the checklist to run before claiming a
loop was sped up or vectorized, and the catalog of places the source survey overstates its case.

## 1. Confirm the transformation actually happened

The compiler may ignore a `#pragma`, refuse to vectorize, or have already done the thing you
"added." Never report a speedup you have not verified two ways: a compiler/assembly report that
the code changed as intended, and a measurement that it got faster.

### Did it vectorize?

**GCC:**
```
-fopt-info-vec          # report loops that WERE vectorized
-fopt-info-vec-missed   # report loops the compiler gave up on, with the reason
```

**Clang:**
```
-Rpass=loop-vectorize          # report vectorized loops
-Rpass-missed=loop-vectorize   # report loops it declined to vectorize
-Rpass-analysis=loop-vectorize # the analysis behind the decision
```

The "missed" reports are the valuable ones — they name the obstacle (aliasing, unknown trip
count, unvectorizable reduction, function call in the body) so you know which technique to apply.

### Look at the assembly

The ground truth. Use [Compiler Explorer (godbolt.org)](https://godbolt.org) or
`g++ -O3 -S -masm=intel`. Vectorization shows up as packed instructions: `vaddps`/`vmulps`
(packed single-precision AVX), `vfmadd…` (fused multiply-add), `ymm`/`zmm` registers (256/512-bit).
Scalar code uses `addss`/`mulss` and `xmm` registers a slot at a time. If you see only scalar
ops in a loop you expected to vectorize, the transformation did not take.

### Enable the instruction set

Auto-vectorization only uses instructions the target allows. `-O3` alone often emits only SSE2
(the x86-64 baseline). To get AVX2/AVX-512 you must say so:
```
-O3 -march=native        # use everything THIS machine supports (not portable)
-O3 -mavx2 -mfma         # portable to any AVX2+FMA machine
```
A loop that "won't vectorize well" frequently just lacks `-march`/`-mavx2`.

## 2. Benchmark honestly

- **Repeat and aggregate.** `perf stat -r 5 ./bench` runs five times and reports mean ± stddev.
  A single run on a shared or laptop-throttled machine is noise; do not quote it.
- **Build with optimization.** Benchmarking a `-O0` build measures nothing real. Use the release
  flags you ship with.
- **Defeat dead-code elimination.** If the result is unused, the compiler may delete the whole
  loop and you will "measure" near-zero time. Consume the result (print it, write it through a
  `volatile`, or use a `DoNotOptimize` helper as in Google Benchmark).
- **Warm up and size realistically.** Cache effects dominate; a loop that fits in L1 in a
  microbenchmark may miss constantly in production. Benchmark at the real working-set size.
- **Profile to find the hot loop first.** `perf record` / `perf report`. Do not optimize a loop
  that is not in the profile — the survey's own framing (one loop becomes 90% of CPU) is exactly
  why you measure before choosing.

## 3. The floating-point reassociation trap (the big one)

Several techniques reorder floating-point additions: **unrolling with multiple accumulators**,
**vectorized reductions**, and the local-`sum` accumulation in **tiling**. Floating-point
addition is *not associative* — `(a + b) + c` and `a + (b + c)` can give different rounded
results. So these transformations change the numeric answer, usually in the last bit or two.

Why it matters:
- The compiler will *not* auto-vectorize or reassociate a `float`/`double` reduction at `-O2`/
  `-O3` precisely because it must preserve your result. It only does so under `-ffast-math`
  (or the narrower `-fassociative-math` / `#pragma omp simd reduction`). Turning on
  `-ffast-math` globally also drops NaN/Inf handling and signed-zero semantics — a big hammer.
- It matters only relative to *your reproducibility standard*. A "match the reference to the last
  printed digit" standard (the one the `cosmolike-dev` skill uses for chi2) tolerates this — the
  reassociation stays well below printed precision, and a fixed build is deterministic run to run,
  so it still passes. Only a strict *bit-for-bit* requirement is broken by it. Two caveats even
  when you opt in: the result can differ across vector widths (an AVX2 vs AVX-512 build reduces in
  4 vs 8 lanes), so re-validate when you change `-march`; and a *parallel* reduction whose
  partition varies with thread count is genuinely non-deterministic — different from the
  deterministic single-thread `omp simd` case.
- Note the direction: splitting one sequential accumulator into several partial sums is a form of
  pairwise summation and is typically *more* accurate than the serial sum, not less. The hazard is
  *difference from a sequentially-computed reference*, not loss of accuracy.

What to do: decide reassociation against your tolerance, not reflexively against bit-identity. If
you truly need bit-for-bit reproducibility, keep the serial reduction (or a compensated/Kahan
sum). Otherwise — the common scientific case, and what `cosmolike-dev` does — opt in locally
rather than flipping global `-ffast-math`:
```cpp
#pragma omp simd reduction(+:sum)   // opt in per-loop, leaves the rest strict
for (int i = 0; i < n; ++i) sum += data[i];
```
For integers none of this applies — integer addition is associative, so unrolled integer sums
are exact.

## 4. Where the source survey overstates its case

The survey is a solid, broad catalog and most of its advice is sound. These are the points worth
pushing back on or qualifying — surface them when someone applies the technique literally.

1. **The "early exit" example changes behavior, not just speed.** Its "before" returns the
   *last* matching index (it keeps overwriting `result`); its "after" returns the *first*. Those
   are different functions. The lesson (exit early) is right, but the two snippets are not
   equivalent — choose first-vs-last by spec, then exit early within that choice.

2. **"Reduces dependency chain" glosses over a numerical change.** The four-accumulator unrolled
   sum is presented as a pure win. It is faster and usually *as accurate or more* accurate, but it
   is not bit-identical to the serial sum (§3). That is fine — even expected — under a
   printed-precision tolerance; it is only a surprise if you assumed bit-identity. Worth a one-line
   note, not a warning label.

3. **The tile-size math is optimistic.** "A 64×64 tile of floats (16 KB) leaves room for both
   input tiles" understates the footprint: a tiled matmul keeps tiles of A, B, *and* C live at
   once (~48 KB), which does not fit a 32 KB L1. The technique is right and the ~14× number is
   plausible; the sizing rule should be "fit all live operands, then tune by measuring," not a
   single arithmetic estimate.

4. **The sentinel trick is mostly obsolete and is a concurrency hazard.** The survey already
   hedges ("modern branch predictors handle the two-condition case well, so profile"). Go further:
   it mutates a shared array, so it is a data race under any concurrent reader, and it breaks on
   `const` data. Treat it as a historical curiosity, not a go-to.

5. **Many "techniques" are things the compiler already does at -O2.** Hoisting, strength
   reduction, and normalization are listed as manual moves but are routine compiler passes. The
   honest framing (used in this skill) is: hand-write them only when a compiler report or the
   assembly shows the compiler did *not* — usually around virtual calls, `volatile`, aliasing, or
   unusual bounds. Presenting all 23 as equally "things you do" overstates the manual workload.

6. **`-O3` does not imply AVX.** The survey's vectorization advice omits `-march`/`-mavx2`.
   Without them the compiler often emits only SSE2, so a loop can be "vectorized" yet leave most
   of the available throughput on the table (§1).

None of these undermine the survey — they are the guardrails that keep its techniques from
turning into silent correctness bugs or unverified speed claims. The meta-rule holds throughout:
measure, confirm the transformation in the report or the assembly, and check the answer is still
right.
