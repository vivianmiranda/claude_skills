# The loop-optimization catalog (23 techniques)

Each entry follows the same shape: **what** it does, **why** it helps (the hardware reason),
**when** to use it, **code** (before → after), and **pitfalls**. Code is kept close to the
source survey so it is easy to recognize, with corrections noted where the original example
is misleading. Read `verification-and-traps.md` alongside this for the floating-point and
measurement caveats that several of these share.

## Table of contents

**Core — use these first (1–6)**
1. [Vectorization](#1-vectorization)
2. [Exit loops early — `break` / `return`](#2-exit-loops-early)
3. [Finish the body early — `continue`](#3-finish-the-body-early)
4. [Correct choice of loop](#4-correct-choice-of-loop)
5. [Unrolling](#5-unrolling)
6. [Fusion](#6-fusion)

**Cache & memory (7–10, 16)**
7. [Fission (distribution)](#7-fission-distribution)
8. [Perforation](#8-perforation)
9. [Tiling / blocking](#9-tiling--blocking)
10. [Reversal — do not use](#10-reversal--do-not-use)
16. [Interchange](#16-interchange)

**Compiler usually does these (11, 12, 20)**
11. [Code motion / hoisting](#11-code-motion--hoisting)
12. [Strength reduction](#12-strength-reduction)
20. [Normalization](#20-normalization)

**Restructuring for SIMD or branches (13–15, 17, 18)**
13. [Coalescing / collapsing](#13-coalescing--collapsing)
14. [Peeling](#14-peeling)
15. [Splitting](#15-splitting)
17. [Sentinel](#17-sentinel)
18. [Strip mining](#18-strip-mining)

**Parallel & advanced (19, 21–23)**
19. [Spreading (OpenMP)](#19-spreading-openmp)
21. [Skewing](#21-skewing)
22. [Interleaving](#22-interleaving)
23. [Reordering (Morton / Z-order)](#23-reordering-morton--z-order)

---

## Core — use these first

### 1. Vectorization

**What.** Get the compiler to emit SIMD instructions (SSE, AVX, AVX-512) so one instruction
processes several array elements at once, instead of one scalar element per instruction.

**Why.** A SIMD register is wide: 128-bit SSE holds 4 floats, 256-bit AVX2 holds 8 floats,
512-bit AVX-512 holds 16. A single `vaddps` adds 8 float pairs in roughly the time of one
scalar add, so throughput rises 4–16× for arithmetic-bound loops over contiguous data. This
is the single highest-payoff transformation for number-crunching loops.

**When.** Numeric loops over arrays with a simple body, no pointer aliasing, no data-dependent
branches, and a trip count the compiler can reason about.

**What helps the compiler vectorize:**
- Remove pointer-aliasing ambiguity with `__restrict__` (or `restrict` in C).
- Keep the loop body simple and branch-free.
- Make the trip count visible (a plain `int n`, not a hidden `size()` recomputed each step).
- Keep accesses unit-stride and, ideally, aligned.

```cpp
// Hard to vectorize — the compiler must assume a may overlap b or c,
// so it cannot prove the writes are independent.
void add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++)
        a[i] = b[i] + c[i];
}

// Easy to vectorize — __restrict__ promises the pointers do not alias,
// so the compiler is free to emit AVX/SSE.
void add(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int n) {
    for (int i = 0; i < n; i++)
        a[i] = b[i] + c[i];
}
```

You can also nudge or force it with hints:

```cpp
#pragma GCC ivdep                  // GCC: ignore assumed vector dependencies
#pragma clang loop vectorize(enable)
```

**Pitfalls.**
- `__restrict__` is a *promise*. If the pointers actually do overlap, you get silent wrong
  answers — only use it when you know the buffers are distinct.
- Reductions (summing into one variable) need floating-point reassociation to vectorize; the
  compiler will not do it without `-ffast-math`/`-fassociative-math`, and doing it changes the
  result. See `verification-and-traps.md`.
- A vectorized loop and its scalar remainder must produce the same answer — check the tail.
- Always confirm it actually vectorized (`-fopt-info-vec` / `-Rpass=loop-vectorize`); a hint
  is not a guarantee.

### 2. Exit loops early

**What.** Stop iterating the moment the work is done — `break` or `return` instead of
scanning to the end.

**Why.** In latency-sensitive code the average case matters as much as the worst case. A
linear search that returns on the first hit does, on average, half the work of one that
always scans to the end.

**When.** Search loops, validation loops, any "find the first X" pattern.

```cpp
// Bad — scans the whole array AND returns the LAST match, not the first.
int find_index_bad(const int* data, int n, int target) {
    int result = -1;
    for (int i = 0; i < n; ++i) {
        if (data[i] == target) {
            result = i;          // keeps going; overwrites with later matches
        }
    }
    return result;
}

// Good — returns the first match immediately.
int find_index(const int* data, int n, int target) {
    for (int i = 0; i < n; ++i) {
        if (data[i] == target) return i;
    }
    return -1;
}
```

**Pitfalls.** These two functions are not equivalent: the "bad" one returns the *last*
matching index, the "good" one the *first*. If duplicates exist and the spec wants the last
match, the early-exit rewrite is a behavior change, not just a speedup. Decide first-vs-last
by specification, then exit early within that choice.

### 3. Finish the body early

**What.** The mirror image of early exit: when a single iteration has nothing to do, `continue`
past the rest of its body instead of nesting the real work inside `if`s.

**Why.** It keeps the hot path tight and flat, and the branch predictor learns the "skip"
pattern quickly. It is also a readability win — guard clauses beat deep nesting.

**When.** Loops where many iterations are filtered out early (inactive records, already-handled
items, sparse data).

```cpp
void process_orders(const std::vector<Order>& orders) {
    for (const auto& order : orders) {
        if (!order.is_active()) continue;   // skip dead orders immediately
        if (order.is_filled())  continue;   // nothing to do

        // Real work — only for live, unfilled orders.
        match_order(order);
        update_book(order);
    }
}
```

**Pitfalls.** None structural. If the predicates themselves are expensive, the cost is the
predicate, not the loop shape — measure.

### 4. Correct choice of loop

**What.** Pick the loop construct that best expresses intent and best exposes structure to the
compiler: range-`for`, index-`for`, iterator-`for`, or an algorithm (`std::for_each`,
`std::accumulate`).

**Why.** In hot paths an index-based `for (int i = 0; i < n; ++i)` makes the trip count
obvious, which helps vectorization. Iterator-based loops sometimes obscure the count. Range-for
is cleanest for "touch every element" with no index arithmetic.

**When.** Default to range-for for clarity; switch to index-for in a hot numeric loop when it
aids vectorization or you need the index.

```cpp
for (int i = 0; i < n; ++i)        // index visible — good for vectorizer & when you need i
    out[i] = f(in[i]);

for (const auto& x : container)     // cleanest when you just need each element
    consume(x);
```

**Pitfalls.** Avoid signed/unsigned mismatch in the condition: `int i < v.size()` compares
`int` to `size_t`, which warns and can mask bugs. Use a matching type, or cache the bound. Note
the converse tradeoff: a signed `int` index can actually help the vectorizer (signed overflow is
undefined, so the compiler assumes no wraparound), but it overflows for arrays above ~2 billion
elements — use `size_t`/`ptrdiff_t` there.

### 5. Unrolling

**What.** Do several iterations' worth of work per loop step, so the loop-control overhead
(compare, increment, branch) is amortized and the CPU has more independent instructions to
schedule.

**Why.** Two distinct effects. First, less loop overhead per element. Second, and usually more
important, **breaking the dependency chain**: a single accumulator forces each add to wait for
the previous one (an FP add has ~3–5 cycle latency but the unit can start a new add every
cycle). Several independent accumulators keep that pipeline full.

**When.** Short, hot loops where a profile shows loop-control overhead, or a reduction whose
serial dependency chain is the bottleneck.

```cpp
// Manually unrolled by 4, with four independent accumulators.
void sum_unrolled(const float* data, int n, float& result) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;

    for (; i + 3 < n; i += 4) {     // main unrolled body
        s0 += data[i];
        s1 += data[i + 1];
        s2 += data[i + 2];
        s3 += data[i + 3];
    }
    for (; i < n; ++i)              // remainder
        s0 += data[i];

    result = s0 + s1 + s2 + s3;     // combine partial sums
}
```

Or let the compiler do it:

```cpp
void process(float* data, int n) {
    #pragma unroll 8               // Clang/GCC hint (GCC also: #pragma GCC unroll 8)
    for (int i = 0; i < n; ++i)
        data[i] = std::sqrt(data[i]);
}
```

**Pitfalls.**
- **The four-accumulator sum is not bit-identical to the serial sum.** Floating-point addition
  is not associative; regrouping changes the rounding. This is the same reason the compiler will
  not auto-vectorize a reduction without `-ffast-math`. For integers it is exact; for `float`/
  `double` in a correctness-sensitive domain, treat it as a deliberate numerical decision.
- Over-unrolling bloats the instruction cache and can slow things down. Let `-O2`/`-O3` unroll
  by default; force it only when profiling shows the overhead is real. `#pragma unroll` is a
  hint the compiler may ignore.

### 6. Fusion

**What.** Merge two loops over the same range into one.

**Why.** Each element is loaded once and reused while still in a register, instead of being
streamed through cache twice. Fewer passes means less memory traffic and better temporal
locality — often the dominant cost on large arrays.

**When.** Two adjacent loops over the same index range where the second does not depend on the
first having fully completed.

```cpp
// Before — two passes, double the cache traffic.
for (int i = 0; i < n; ++i) a[i] = b[i] * 2.0f;
for (int i = 0; i < n; ++i) c[i] = a[i] + 1.0f;

// After — one pass; a[i] is still in a register when c[i] uses it.
for (int i = 0; i < n; ++i) {
    a[i] = b[i] * 2.0f;
    c[i] = a[i] + 1.0f;
}
```

**Pitfalls.** Only valid when there is no cross-iteration dependency that needs the first loop
to finish entirely (e.g. the second loop reading `a[i+1]`). The compiler can fuse at `-O3` but
often will not when the bodies are complex or live in different functions, so this is a common
profitable manual move.

---

## Cache & memory

### 7. Fission (distribution)

**What.** The opposite of fusion: split one loop into several over the same range.

**Why.** Isolate the vectorizable arithmetic from the branchy or call-heavy work so the first
loop vectorizes cleanly. Also helps when one fat body overflows the instruction cache.

**When.** A loop body mixes simple SIMD-friendly math with a branch or function call that kills
vectorization.

```cpp
// Before — the branch + call in the same body blocks the vectorizer.
for (int i = 0; i < n; ++i) {
    a[i] = b[i] * c[i];
    if (a[i] > threshold) log(a[i]);
}

// After — the first loop now vectorizes; the second keeps the messy part.
for (int i = 0; i < n; ++i) a[i] = b[i] * c[i];
for (int i = 0; i < n; ++i) if (a[i] > threshold) log(a[i]);
```

**Pitfalls.** You give up fusion's single-pass locality, so you re-read `a`. The vectorization
win usually dominates, but it is a tradeoff — profile. ("Loop distribution" is the same thing
under a different name.)

### 8. Perforation

**What.** Skip iterations on purpose, trading accuracy for speed (approximate computing).

**Why.** If processing every Nth element gives an answer that is good enough, you do 1/N of the
work. Only legitimate where small errors are acceptable.

**When.** Approximate workloads only: audio level metering, rough analytics, some ML and
graphics. Not for anything where the result must be exact.

```cpp
// Rough energy estimate from every 4th sample.
float estimate_energy(const float* signal, int n) {
    float energy = 0.0f;
    for (int i = 0; i < n; i += 4)        // skip 3 of every 4
        energy += signal[i] * signal[i];
    return energy * 4.0f;                  // compensate for skipping
}
```

**Pitfalls.** Use with extreme care; in a trading or scientific pipeline it is almost never
acceptable. The `* 4.0f` correction assumes the skipped samples resemble the kept ones — false
for signals with structure at the skip stride (aliasing).

### 9. Tiling / blocking

> **Priority technique.** For nested loops over data larger than cache, this is the first thing
> to reach for — it is where the largest wins in this catalog live. Weight it above vectorization
> and the cheaper transformations when the working set exceeds L1/L2.

**What.** Restructure nested loops so they work on small sub-blocks ("tiles") that fit in cache,
finishing each tile before moving on, instead of streaming across the whole array repeatedly.

**Why.** When the working set exceeds cache, you miss on essentially every iteration. A tile
sized to fit in L1 is loaded once and reused many times before eviction — turning capacity
misses into hits. This is the classic, highest-impact cache optimization for matrix-heavy code.

**When.** Nested loops over large matrices where the naive traversal re-streams an operand
(e.g. column access to `B` in a matmul) and the working set exceeds L1.

```cpp
// Naive matmul — two separate problems at once.
void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i*N + j] += A[i*N + k] * B[k*N + j];  // B[k*N+j] strides by N (column access)
}
```

The naive loop has **two distinct problems that need two distinct fixes** — and the common
mistake (the one the source survey makes) is reaching for a cubic tile that only addresses one
of them while quietly overflowing L1:

1. **Bad access pattern.** The inner loop reads `B` down a column — stride `N`, one useful
   element per cache line. Fixed by *reordering the loops*, not by tiling.
2. **No reuse at scale.** Even with a good access pattern, each operand is streamed `N` times,
   so once the matrix exceeds cache you miss repeatedly. This is what tiling fixes.

**Fix step 1 — reorder `ijk` → `ikj` so the inner loop is contiguous and vectorizes.** Swapping
`j` and `k` makes `A[i*N+k]` a loop-invariant scalar and makes both `B[k*N+j]` and `C[i*N+j]`
stride-1: the inner loop is now a SAXPY.

```cpp
void matmul_ikj(const float* A, const float* B, float* C, int N) {   // C must be zeroed
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            const float a = A[i*N + k];          // invariant across j → scalar broadcast
            for (int j = 0; j < N; ++j)
                C[i*N + j] += a * B[k*N + j];     // both stride-1 → vectorizes to FMA
        }
}
```

This alone is usually a multiple-x speedup over naive, with two bonuses: the inner loop has no
reduction (each `j` lane is independent — nothing to reassociate), and each `C[i,j]` still
accumulates over `k` in increasing order, so the result is bit-identical to the naive loop under
matching FMA/contraction flags. It does not yet fix reuse, so for large `N` add tiling on top.

**Fix step 2 — tile the `ikj` loop, sized honestly.** Because `A` is now just a scalar in the
inner loop, only a **B tile and a C tile** must stay cache-resident — two tiles, not the three
of cubic `ijk` tiling. That halves the footprint before you even pick a size.

```cpp
constexpr int TILE = 32;   // sized to fit L1 with headroom — see sizing rule

void matmul_tiled_ikj(const float* A, const float* B, float* C, int N) {  // C must be zeroed
    for (int ii = 0; ii < N; ii += TILE)
      for (int kk = 0; kk < N; kk += TILE)
        for (int jj = 0; jj < N; jj += TILE)
          for (int i = ii; i < std::min(ii + TILE, N); ++i)
            for (int k = kk; k < std::min(kk + TILE, N); ++k) {
                const float a = A[i*N + k];
                for (int j = jj; j < std::min(jj + TILE, N); ++j)
                    C[i*N + j] += a * B[k*N + j];
            }
}
```

**Sizing rule.** Count the operands that must stay resident, then leave headroom. Here that is a
`TILE×TILE` block of B and of C: `2 · TILE² · sizeof(float)`. For a 32 KB L1 with headroom,
`TILE = 32` gives `2·32²·4 = 8 KB` (comfortable); `TILE = 64` would be 32 KB — no headroom, too
big. In practice you extract more reuse by targeting **L2** (256 KB–1 MB) with a larger block
while a small register micro-tile of C (e.g. 4×8 accumulators held in registers across the `k`
loop) does the innermost work — that is the structure tuned BLAS kernels use. Always confirm by
measuring across a few sizes (32, 48, 64, 96); the best tile depends on the exact cache hierarchy.

The survey reports a 1024×1024 matmul dropping ~1.9 s → ~130 ms (~14×) from tiling alone; `ikj`
+ honest tiling does better and is numerically cleaner.

**Pitfalls.**
- **Do not size one cubic `TILE` for all three operands and assume it fits L1.** The survey's
  64×64 cubic tile keeps A, B, and C tiles live (~48 KB) and overflows a 32 KB L1. Reorder to
  `ikj` first (drops A to a scalar → two live tiles), then size to fit with headroom.
- An alternative to `ikj` is to transpose `B` once so `ijk` reads both operands contiguously, but
  that needs a transpose pass and its inner loop is a reduction (reassociation applies) — `ikj`
  is usually the cleaner fix.
- For production, a hand-rolled kernel will not beat a tuned BLAS (OpenBLAS, BLIS, MKL). Hand
  tiling is for learning, for shapes a BLAS does not cover, or when a dependency is unwelcome.

### 10. Reversal — do not use

**What.** Iterating backwards (`for (int i = n-1; i >= 0; --i)`).

**Why it was suggested.** Comparing the counter against zero is marginally cheaper on some old
architectures.

**Why to avoid it.** On modern x86 that saving is gone, and reverse traversal fights the
hardware prefetcher, which is tuned for forward sequential access — so you get *more* cache
misses, not fewer.

**When.** Only when the algorithm genuinely requires reverse order (popping a stack, processing
in LIFO order). Never as a performance trick.

### 16. Interchange

> **The most preferable technique — do this first.** Getting the innermost loop to stride over
> contiguous memory is the foundational loop optimization: cheap, almost always safe, often a
> multiple-x win by itself, and the prerequisite that makes tiling (#9) and vectorization (#1)
> actually pay off. A tiled or vectorized loop over column-strided data still thrashes the cache.
> Whenever an inner loop touches non-contiguous memory, fix this before reaching for anything else.
> (For matmul this is the `ijk`→`ikj` reorder in #9.)

**What.** Swap the nesting order of two loops so the innermost loop strides over contiguous
memory.

**Why.** C/C++ arrays are row-major: `data[i*cols + j]` is contiguous in `j`. Looping with `j`
innermost walks memory sequentially, so each cache line is fully used and the prefetcher keeps
up. Looping with `i` innermost jumps `cols` elements each step, touching one element per cache
line — often the difference between fitting in cache and not.

**When.** Any nested loop whose inner index does not stride the contiguous dimension.

```cpp
// Bad — column-major access in a row-major array; cache-line thrashing.
for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i)
        data[i*cols + j] *= 2.0f;

// Good — inner loop walks contiguous memory.
for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
        data[i*cols + j] *= 2.0f;
```

**Pitfalls.** Interchange is only legal when it does not violate a data dependency. For the
common elementwise/reduction cases it is safe; for loop-carried dependencies, check first.

---

## Compiler usually does these

### 11. Code motion / hoisting

**What.** Move a computation that does not change between iterations out of the loop.

**Why.** A loop-invariant expression computed inside the loop is wasted work multiplied by the
trip count. The worst case is asymptotic: a function call in the loop *condition*.

**When.** Any invariant in the body; especially virtual calls, `volatile` reads, or expensive
math the compiler cannot prove invariant.

```cpp
// The famous O(n^2) trap — strlen() re-runs every iteration.
for (int i = 0; i < strlen(str); i++)   // recomputed each step!
    process(str[i]);

// Fixed — compute the length once.
int len = strlen(str);
for (int i = 0; i < len; i++)
    process(str[i]);
```

```cpp
// The compiler cannot hoist a virtual call or a volatile read — do it yourself.
const float scale = config.get_scale();   // virtual; not provably invariant
for (int i = 0; i < n; ++i)
    data[i] *= scale;
```

**Pitfalls.** Hoisting is only valid if the expression really is invariant. If `get_scale()`
has side effects, or the loop can change what it returns, you cannot hoist it. The compiler does
this automatically at `-O2`+ for cases it can prove; you only need to step in for virtual calls,
`volatile`, or aliasing it cannot see through.

### 12. Strength reduction

**What.** Replace an expensive operation on the loop variable (a multiply) with a cheaper
incremental update (an add).

**Why.** A multiply per iteration becomes one add per iteration when you carry a running offset
("induction variable"). Cheaper per step, and a friendlier pattern for the vectorizer.

**When.** Non-unit strides or 2D index arithmetic; mostly the compiler handles it, but it can
miss complex index expressions.

```cpp
// Before — i * stride is a multiply every iteration.
for (int i = 0; i < n; ++i)
    process(data[i * stride]);

// After — carry the offset and add.
int offset = 0;
for (int i = 0; i < n; ++i) {
    process(data[offset]);
    offset += stride;
}

// 2D: hoist the row pointer so the inner loop has no multiply.
for (int i = 0; i < rows; i++) {
    int* row = matrix + i * cols;   // one multiply per outer iteration
    for (int j = 0; j < cols; j++)
        sum += row[j];
}
```

**Pitfalls.** Mostly redundant with `-O2`; reach for it only when the assembly shows a multiply
in the inner loop the compiler did not eliminate.

### 20. Normalization

**What.** Rewrite a loop so it starts at 0, increments by 1, with an explicit trip count.

**Why.** A canonical loop is the form every other analysis pass (vectorization, unrolling,
dependence testing) reasons about most easily. Odd bounds or strides can block those passes.

**When.** Rarely by hand — it is mostly an internal compiler pass. Worth trying manually if the
compiler refuses to vectorize a loop with unusual bounds.

```cpp
// Before — non-zero start, non-unit step.
for (int i = start; i < end; i += step)
    process(data[i]);

// After — normalized.
int trip = (end - start + step - 1) / step;
for (int i = 0; i < trip; ++i)
    process(data[start + i * step]);
```

**Pitfalls.** The normalized form reintroduces a multiply (`i * step`) in the index — pair it
with strength reduction if that index is hot.

---

## Restructuring for SIMD or branches

### 13. Coalescing / collapsing

**What.** Flatten nested loops into a single loop over the total iteration count.

**Why.** One flat loop with a large trip count vectorizes and parallelizes more readily than a
short inner loop nested inside an outer one, and removes the inner loop's setup/teardown.

**When.** A 2D (or higher) traversal over contiguous storage with no row-to-row dependency.

```cpp
// Before — nested.
for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
        data[i*cols + j] *= 2.0f;

// After — one flat loop.
int total = rows * cols;
for (int i = 0; i < total; ++i)
    data[i] *= 2.0f;
```

With OpenMP, `collapse` expresses the same idea for parallelism:

```cpp
#pragma omp parallel for collapse(2)
for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
        data[i*cols + j] *= 2.0f;
```

**Pitfalls.** Only valid when the array is genuinely contiguous (a single allocation, not an
array of row pointers) and iterations are independent. `rows * cols` can overflow `int` for
large grids — use a wide enough type.

### 14. Peeling

**What.** Pull the special first (or last) iteration(s) out of the loop so the main body is
uniform.

**Why.** A per-iteration branch that is only true on the first step (e.g. "no previous element")
is paid on every step and blocks vectorization. Peeling that case out leaves a clean,
branch-free, vectorizable body. Also used to peel iterations until a pointer is SIMD-aligned.

**When.** The first/last iteration is a special case, or you need alignment before the main
vectorized run.

```cpp
// Before — the i == 0 branch is checked every iteration.
for (int i = 0; i < n; ++i) {
    if (i == 0) result[i] = data[i];
    else        result[i] = data[i] - data[i - 1];
}

// After — peel the first iteration; the rest is branch-free.
result[0] = data[0];
for (int i = 1; i < n; ++i)
    result[i] = data[i] - data[i - 1];
```

**Pitfalls.** Handle the `n == 0` edge case — the peeled `result[0] = data[0]` assumes at least
one element exists.

### 15. Splitting

**What.** Split the *iteration range* at a known boundary so each sub-loop has uniform behavior.
(Different from fission, which splits the *body*.)

**Why.** A condition that flips partway through the range forces a branch on every iteration.
Splitting the range removes the branch — each sub-loop is straight-line and independently
optimizable.

**When.** A loop whose behavior changes at a known index boundary.

```cpp
// Before — a branch on i every iteration.
for (int i = 0; i < n; i++) {
    if (i < boundary) a[i] = forward_formula(i);
    else              a[i] = backward_formula(i);
}

// After — split at the boundary; each loop is uniform.
for (int i = 0; i < boundary; i++) a[i] = forward_formula(i);
for (int i = boundary; i < n; i++) a[i] = backward_formula(i);
```

**Pitfalls.** Only helps when the boundary is loop-invariant. If `boundary` is recomputed or
data-dependent, splitting does not apply.

### 17. Sentinel

**What.** Write the search target into a spare slot at the end of the array so the inner loop
needs only one comparison per step (the equality test), not two (bounds and equality).

**Why.** The guaranteed match terminates the loop, so you drop the `i < n` bounds check from the
hot path, halving the branch count in a tight search.

**When.** Linear search over a writable array that has one extra slot, in a genuinely
search-bound loop.

```cpp
int find_sentinel(int* data, int n, int target) {
    int saved = data[n];           // needs a writable spare slot at index n
    data[n] = target;              // place the sentinel

    int i = 0;
    while (data[i] != target) ++i; // one comparison per iteration

    data[n] = saved;               // restore
    return (i < n) ? i : -1;       // i == n means only the sentinel matched
}
```

**Pitfalls.** This is a Knuth-era trick whose payoff has largely evaporated: modern branch
predictors handle the two-condition loop nearly for free, so profile before adopting it. It also
requires write access and a spare slot, breaks on `const` data, and is **not thread-safe** —
mutating a shared array under a reader is a data race. Often not worth the complexity.

### 18. Strip mining

> **Priority technique.** For a 1D numeric loop you want to vectorize, this is the first thing to
> reach for — it is the bridge between a plain loop and an efficient SIMD kernel, and the 1D
> counterpart of tiling. Weight it above the cheaper transformations when a hot stream needs SIMD.

**What.** Split a 1D loop into an outer loop over fixed-size "strips" and an inner loop within
each strip. Effectively 1D tiling, and the foundation for vectorization and cache blocking.

**Why.** The inner loop now has a compile-time-constant trip count (`STRIP`), so the compiler
can fully unroll and vectorize it with no messy remainder handling, and you get explicit control
over the working-set size handed to SIMD or a worker thread.

**When.** Preparing an inner kernel for SIMD, or cache-blocking a 1D data stream.

```cpp
constexpr int STRIP = 256;   // tuned for vector registers + L1

void process_stripmined(float* data, int n) {
    int i = 0;
    for (; i + STRIP <= n; i += STRIP)
        for (int j = 0; j < STRIP; ++j)        // fixed count → fully unrolled/vectorized
            data[i + j] = std::fma(data[i + j], 2.0f, 1.0f);
    for (; i < n; ++i)                          // remainder
        data[i] = std::fma(data[i], 2.0f, 1.0f);
}
```

**Pitfalls.** Choose `STRIP` as a multiple of the SIMD width and small enough to stay in cache.
The remainder loop still needs to match the main loop's results exactly.

---

## Parallel & advanced

### 19. Spreading (OpenMP)

**What.** Distribute iterations across cores (or SIMD lanes) — in practice, OpenMP or manual
thread partitioning.

**Why.** Independent iterations run concurrently on multiple cores, scaling throughput with core
count.

**When.** The per-iteration cost is high enough to dwarf the scheduling and synchronization
overhead.

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; ++i)
    result[i] = expensive_computation(input[i]);
```

**Pitfalls.** For short loops the cost of spawning/synchronizing the team can exceed the loop's
entire runtime — spreading then makes it slower. Watch for **false sharing** (different threads
writing into the same cache line) and shared-state races. In low-latency code, thread sync
overhead is often the thing you are trying to avoid in the first place.

### 21. Skewing

**What.** Reshape the iteration space of a nested loop with cross-iteration dependencies so that
independent iterations line up and can run in parallel. Common in stencil computations.

**Why.** A stencil like `a[i][j] = f(a[i-1][j-1], a[i-1][j], a[i-1][j+1])` has a dependency that
blocks naive parallelization. Skewing the index (`new_j = j + i`) realigns the iteration space
so a diagonal "wavefront" of independent points can be computed together.

**When.** Stencil / wavefront codes in scientific computing or GPU kernels.

```cpp
// 1D stencil; after skewing new_j = j + i exposes a parallel inner dimension.
for (int i = 1; i < N; ++i)
    for (int new_j = i + 1; new_j < M + i - 1; ++new_j) {
        int j = new_j - i;
        a[i][j] = 0.33f * (a[i-1][j-1] + a[i-1][j] + a[i-1][j+1]);
    }
}
```

**Pitfalls.** Advanced and error-prone — skewing alone usually has to be combined with
interchange to actually expose the parallel loop, and getting the bounds right is fiddly. If you
need this, reach for a polyhedral compiler (Pluto) rather than hand-transforming.

### 22. Interleaving

**What.** Mix iterations of two independent computations so their memory accesses overlap and
hide each other's latency — software pipelining across loops.

**Why.** An out-of-order CPU can have several loads in flight at once. Issuing a load for
computation A and then doing computation B's work while A's data arrives keeps the execution
units busy instead of stalling on memory.

**When.** Two independent streams of work over the same range; the compiler's scheduler is not
overlapping them on its own.

```cpp
for (int i = 0; i < n; ++i) {
    float a_val = a_input[i] * a_scale;   // start A's load
    float b_val = b_input[i] + b_offset;  // do B while A is in flight
    a_output[i] = a_val;
    b_output[i] = b_val;
}
```

**Pitfalls.** The real payoff is at coarser grain — process a chunk of A, then a chunk of B,
keeping both datasets warm in cache. This is a manual move for when the compiler's instruction
scheduler is leaving parallelism on the table; verify with a profile that it helped.

### 23. Reordering (Morton / Z-order)

**What.** Change the order in which iterations visit data — without changing results — to improve
locality. Interchange is the simplest form; this also covers space-filling-curve traversal of
2D/3D grids.

**Why.** Row-major traversal of a large 2D grid causes TLB and cache misses when the row stride
is large. A Z-order (Morton) curve keeps spatially-near points near in time, so a 2D
neighborhood stays resident in cache.

**When.** Large 2D/3D datasets where standard row-major traversal thrashes the TLB — image
processing, physics simulations.

```cpp
// Z-order traversal instead of row-by-row.
for (int z = 0; z < n * n; ++z) {
    int x = decode_morton_x(z);
    int y = decode_morton_y(z);
    if (x < width && y < height)
        process(grid[y * width + x]);
}
```

**Pitfalls.** Heavy machinery — the Morton encode/decode has its own cost, so it only pays off
when locality gains dominate. For independent loops X, Y, Z, their order also decides what is hot
in cache when the next one starts; profile rather than guess.
