---
name: pytorch-teaching-notebook-style
description: >-
  Use when writing, explaining, revising, or commenting PyTorch code meant for a
  teaching Jupyter notebook presented as slides (e.g. RISE). Enforces a house
  style: slide-narrow code, maximally didactic step-by-step comments that assume
  novice familiarity with every API, formal definitional comment tone, full-cell
  rewrites for any change, explicit-argument and single-responsibility design,
  strict only-what-was-asked scope, pure plain-text "raw blocks" for slide prose,
  no all-caps emphasis, and grounding abstractions in a concrete research example
  (a cosmic-shear data-vector emulator). Trigger on PyTorch lecture/teaching
  material, notebook cells destined for slides, or any request for a "raw block".
---

# PyTorch teaching-notebook style

How to produce and present PyTorch code for a teaching notebook shown as slides.
The audience is a learner who is an expert in their own scientific domain
(cosmology) but a novice in PyTorch and most programming APIs. Optimize for
clarity and didactic transparency, never for expert-to-expert brevity.

## Only what was asked (no unprompted additions)

- **Deliver exactly the requested change — nothing more.** Do not add
  features, options, refactors, defensive guards, or numerical/robustness
  embellishments the user did not ask for. When asked to make a quantity
  per-column, make it per-column; do not also bolt on a stability trick.
- **If an extra seems worth it, ask first, in one line,** and let the user
  decide. Never bundle it into the cell by default. Unrequested scope makes
  the code harder to read and forces the user to strip it back out.

## Output: always full cells

- **Deliver complete cells, never snippets.** For any change — even a one-line
  fix, a variable rename, or a comment tweak — give the entire cell, paste-ready.
  Never send a diff, only the changed lines, or a "replace X with Y" instruction.
  If a change spans multiple cells (e.g. a rename used across them), reprint every
  affected cell in full. No exceptions, even when the edit feels mechanical.
- **Point to a location by context, never by cell or line number.** The user reads
  the notebook as slides and cannot see cell indices or line numbers, so "cell 274"
  or "line 41" is meaningless to them. Anchor every reference to something visible
  in the content: the enclosing function or class name, a nearby comment or
  raw-block marker, or a short quoted line (e.g. "the `C_used = param_geom.encode(
  ...)` line in `_build_loaders_one`"). This holds for bug reports, change
  locations, and any place you direct the user's attention — not just the cells you
  hand back.

## No global variables in functions (flag the rare exception)

- **A function takes what it needs as parameters and reads no
  module-level data global.** Imports and module-level helper
  functions/classes are not "globals" in this sense — referencing `np`,
  `torch`, or a `make_X` helper is fine. The rule is about data and
  state: `device`, normalization stats, a fitted geometry, a config
  object, dataset arrays. Reaching out to those hides the function's real
  inputs and breaks the moment it is reused, moved, or called before the
  global exists. Pass them in, even when the signature gets longer.
- **A global read is an extreme exception, and never silent.** When
  threading a value through is genuinely unworkable and a global must be
  read, flag it in-code on the line that reads it with a visible marker —
  an emoji and the word WARNING — naming the global and why it is not a
  parameter, e.g.
  `# ⚠️ WARNING: reads module global device (not threaded as a param)`.
  A silent global read is a latent bug and a review red flag; the marker
  makes the dependency impossible to miss. (This WARNING is the one
  sanctioned all-caps token — see "No all-caps for emphasis" — a flag
  label, not emphasis.)
- **Check for silent globals mechanically, not by eye.** A function's
  free names should be only its parameters, locals, imports, and
  module-level helpers; anything else is a data-global read. A `symtable`
  pass (or a free-variable walk) over the cells lists, per function, the
  names it treats as global — filter out the imports and helper defs and
  what remains is the leak set. This catches what reading skips.
- **The rename hazard makes silent globals worse.** A global read by a
  now-stale name is the trap: rename the definition (`param_geom` ->
  `pgeom`) and a caller still passing `param_geometry=param_geom` either
  raises `NameError` or silently binds a stale value left in the kernel
  from an earlier run. Passing the parameter explicitly — and flagging
  any genuinely unavoidable global — removes both failure modes.

## Simplicity (factor shared logic)

- **Prefer the simplest design that works.** When two pieces of code do the
  same thing to different data (normalize the params, then the data
  vectors), write one small general function and call it twice — never
  duplicate the logic across parallel variables. Repetition is a design
  smell, and parallel near-copies drift out of sync.
- Favor a short single-purpose function over a longer one juggling several
  things at once. Clunky, repetitive code is harder to read on a slide and
  harder to trust.

## Code width (it must fit a slide)

- Keep code lines **<= ~60 characters, hard ceiling ~64.** Slides do not wrap
  code; an over-long line overflows the slide.
- Code often shares a slide with explanatory text in a two-column layout, so the
  code column, not the full slide, sets the budget. When code sits beside prose,
  aim for ~55.
- Break long statements with trailing operators, parenthesized continuations,
  split chained calls, or factored-out intermediates. For a wrapped call or
  signature, break after the opening `(` and use a **2-space hanging indent**
  for the continuation lines; never align the arguments under the opening
  paren (that pushes them far right and blows the width budget). Verify the
  width before sending.

## Comments and explanations (maximally didactic)

Assume the reader is a novice in the specific API. Domain expertise does not
transfer to terse code reasoning, so stay step-by-step even in their field.

- **Name every non-obvious mechanic.** Spell out: iterator vs list, lazy vs eager
  (generators / `yield`), in-place vs copy, views vs copies, device placement,
  scoping rules (`nonlocal`), hidden defaults, gradient accumulation, integer vs
  float division, and which parameter an anonymous value fills (for example, "the
  `lambda t: t` passed as the second argument is the `unpack` callback"). A novice
  would never infer these on their own.
- **Define jargon in plain words before using it** (e.g. pack/unpack = "store it
  / get it back").
- **Show the full chain of reasoning.** Never call a step "obvious"; when a token
  could confuse, pre-empt it.
- **Unpack clever code into clear steps.** Prefer numbered, named intermediate
  variables over a compact one-liner when teaching a mechanism (e.g. split a
  Hessian-vector product into "first backward", "dot product", "second backward").
- **No magic numbers.** Pull numeric controls into named variables and explain
  what each represents; prefer a principled stopping rule ("stop when the gradient
  is 1/1000 of its starting value") over a guessed iteration count. Do not write
  meta-commentary like "no magic number here" — just explain the value.
- **Teach the correct algorithm**, not a numerically-equivalent shortcut. (Example:
  a convergence check gates the optimizer step, so break before stepping, not
  after, even if the difference is negligible.)
- **Consistency over cleverness.** Reuse the APIs and idioms already introduced in
  the notebook; do not reach for a different-but-equivalent tool the reader has
  not seen. If a genuinely new API is warranted, introduce it explicitly (a short
  aside or raw block) and include its import. Never swap silently. Example failure:
  using `F.mse_loss(...)` after the notebook has only ever used `mse = nn.MSELoss()`
  — the consistent move is `mse_sum = nn.MSELoss(reduction="sum")`.
- **Write comments formal and definitional, not chatty.** Prefer
  "n_layers = number of dense layers between two skip points" to
  "n_layers is how many dense layers sit between the skips." Define terms
  with `name = ...` or a parameter block (`Arguments:`, one line each) and
  state mechanics as facts. Avoid conversational fillers ("we just",
  "now we", "let's", first-person "I").

## Recurring idioms — teach (and use) these correctly

These mechanics recur in this material; get them right and name them for
the reader.

- **Factories, not shared instances.** Pass a class/callable (e.g. a `norm`
  or `act`) and call it once per layer — `norm(size)` — so each layer gets
  its own module; a shared instance ties the layers' learnable parameters
  together. Wrap a size-less module in a lambda: `lambda s: nn.Tanh()`.
- **Spec dicts for constructible components.** Parameterize each
  constructible piece (model, optimizer, scheduler) with a dict whose `"cls"`
  key holds the class — the same first-class-value trick as the factories
  above — and whose other keys are its constructor kwargs. A small
  `make_X(spec, injected...)` helper pops `"cls"` and forwards the rest:
  `make_model(model_opts, in_dim, out_dim, device)`,
  `make_optimizer(model, opt_opts, lr, device)`,
  `make_scheduler(opt, sched_opts)`. Keep computed or device-dependent args
  out of the dict — the helper injects them: the `lr` (batch-scaled), `fused`
  / `torch.compile` (gated on `device.type`), in/out dims (from data and
  geometry), the optimizer handed to the scheduler. The driver then takes one
  spec dict per component (`model_opts`, `opt_opts`, `lr_opts`, `sched_opts`).
  Caveat: generalizing the scheduler *class* does not generalize *stepping* —
  `ReduceLROnPlateau` steps with a metric, others with none (branch on
  `isinstance`); per-batch schedulers (`OneCycleLR`) step inside the batch
  loop, not once per epoch.
- **`nn.ModuleList`, never a bare Python list,** for submodules — a plain
  list does not register parameters (missing from `.parameters()`,
  `.to(device)`, `state_dict`). A plain list is fine only as a temporary
  builder fed to `nn.Sequential(*layers)`.
- **Mutable default args:** `def f(x=None): ...; if x is None: x = {}`,
  never `def f(x={})` — one dict is created once and leaks across calls.
- **Alternative constructors are classmethods** (`from_cosmolike`,
  `from_state`), not a branching `__init__`; `__init__` just stores fields,
  and `cls(...)` (not a hardcoded class name) keeps subclasses correct.
- **Encapsulate data manipulation in the owning class.** Loaders read bare
  arrays; the class does squeeze / center / whiten / unsqueeze. Hand it the
  raw quantity (e.g. the full-block mean) and let it do the transform.
- **Device / dtype discipline.** Say where each tensor lives and its dtype;
  cast inputs to the model's dtype (`loadtxt` gives float64, the model is
  float32) and accumulate reductions / chi2 in float64.
- **An additive offset cancels in a difference.** Centering (subtract a
  mean or fiducial) never changes a residual-based loss — only the network
  targets' zero-point. Say so when it matters.
- **Weight decay belongs only on weight matrices.** L2 pulls every
  parameter it touches toward 0, which is meaningful only for connection
  weights. Do not decay learned-activation shape params
  (`activation_fcn.gamma/beta`), the `Affine`/norm `gain` (inits to 1 —
  decay drags it to 0 and attenuates the signal), or biases. Either set
  `weight_decay = 0` (simple; emulators with abundant data rarely need it)
  or split with the standard `ndim >= 2` rule — weight matrices are 2D,
  biases / `Affine` / `gamma`/`beta` are 1D:
  `for _, p in model.named_parameters(): (decay if p.ndim>=2 else
  no_decay).append(p)`, then two Adam param groups with `weight_decay` on
  `decay` and `0.0` on `no_decay`.

## Ground abstractions in the research case

When a comment introduces an abstract quantity or formula, add one short, symbolic
sentence connecting it to the reader's real research (here: emulating cosmic-shear
data vectors). Example: "in this toy L = 1, but for a cosmic-shear data vector
L = n_pairs * n_theta". Keep it symbolic and slide-tight; do not expand into a
wall of numbers (N_z=5 -> 15 pairs x 20 angles = 300) unless it clearly fits and
is asked for.

The running example is a cosmic-shear data-vector emulator: a ResMLP maps
cosmological parameters to the data vector. Targets are whitened in the covariance
eigenbasis (eigh of the unmasked block covariance), so the network predicts a
unit-variance, decorrelated vector; the loss un-whitens the residual and computes
the true full-3x2pt chi2 with cosmolike's masked inverse covariance. Only unmasked
entries are emulated ("squeeze"); the chi2 "unsqueezes" back to the full vector.
Training streams the data vectors in chunks (memmap -> device) as if too big for
GPU RAM. Params are z-scored with 5x std; the data vector is centered on the
training mean. The cov/mask/whitening geometry is saved with the model so inference
never re-reads cosmolike. cosmolike (`ci`) lives only on the user's workstation, so
code here is reviewed statically and the math is verified in isolation.

## Case study: the DataVectorGeometry class

A worked example the user singled out as the design to emulate. One class owns the
masked-data-vector geometry and every transform; a thin subclass adds the loss.

```python
class DataVectorGeometry:
  # Owns the masked-dv geometry + normalization for one
  # probe. Build from cosmolike (train) or saved tensors
  # (inference). Every dv transform lives here.
  PROBE_BLOCKS = {"xi": [0], "ggl": [1], "wtheta": [2],
                  "3x2pt": [0, 1, 2]}

  def __init__(self, device, total_size, dest_idx,
               evecs, sqrt_ev, Cinv, center):
    # plain: place fields on device. as_tensor accepts numpy
    # (from cosmolike) or cpu tensors (from a saved state).
    self.total_size = int(total_size)
    # dest_idx = global positions (in the full 3x2pt vector)
    # of the unmasked entries; squeeze/unsqueeze/center all
    # key off it -- never a block-local index (see below).
    self.dest_idx = torch.as_tensor(
      dest_idx, dtype=torch.long, device=device)
    self.evecs = torch.as_tensor(
      evecs, dtype=torch.float32, device=device)
    self.sqrt_ev = torch.as_tensor(
      sqrt_ev, dtype=torch.float32, device=device)
    self.Cinv = torch.as_tensor(
      Cinv, dtype=torch.float64, device=device)
    self.center = torch.as_tensor(
      center, dtype=torch.float32, device=device)

  @classmethod
  def from_state(cls, device, state):
    return cls(device, **state)     # keys match __init__

  @classmethod
  def from_cosmolike(cls, device, dv_center, ...):
    # read cov/inv_cov/mask/sizes; keep_cols = offsets of
    # the unmasked entries within the probe's block(s);
    # dest_idx = block_global[keep_cols], their global slots
    # in the full 3x2pt vector; center = dv_center[dest_idx]
    # (full-vector mean, indexed globally); eigh the unmasked
    # block cov; then build via __init__:
    return cls(device, total_size, dest_idx,
               U, sqrt_lam, Cinv, center)

  def state(self):                  # tensors to save
    return {"total_size": self.total_size,
            "dest_idx": self.dest_idx.cpu(),
            "evecs": self.evecs.cpu(),
            "sqrt_ev": self.sqrt_ev.cpu(),
            "Cinv": self.Cinv.cpu(),
            "center": self.center.cpu()}  # keys==__init__

  def squeeze(self, dv):   return dv[:, self.dest_idx]
  def unsqueeze(self, sq):
    full = torch.zeros(sq.shape[0], self.total_size,
                       dtype=sq.dtype, device=sq.device)
    full[:, self.dest_idx] = sq
    return full
  def whiten(self, x):   return (x @ self.evecs) / self.sqrt_ev
  def unwhiten(self, y): return (y * self.sqrt_ev) @ self.evecs.T
  def encode(self, dv):  # raw block -> network target
    return self.whiten(self.squeeze(dv) - self.center)
  def decode(self, y):   # network output -> physical dv
    return self.unwhiten(y) + self.center

class CosmolikeChi2(DataVectorGeometry):  # subclass = loss only
  def chi2(self, pred, target):
    r = self.unsqueeze(self.unwhiten(pred - target)).double()
    return torch.einsum("bi,ij,bj->b", r, self.Cinv, r)
```

Decisions that shaped it (the reasoning to reuse):

- **One class owns every transform; callers pass bare data.** Loaders only read
  rows; `encode` does squeeze -> center -> whiten. When a transform leaked into the
  driver, that was the smell to fix.
- **Split geometry from loss by subclassing.** `DataVectorGeometry` holds the
  transforms and covariance; `CosmolikeChi2` adds only `chi2`/`loss`. Single
  responsibility, and the chi2 reuses `unwhiten` + `unsqueeze`.
- **Two constructors as classmethods, not a branching `__init__`.**
  `from_cosmolike` reads the dataset; `from_state` rebuilds from saved tensors so
  inference never re-reads cosmolike (the dataset/probe can change — save the
  geometry with the weights). `__init__` just stores fields; `cls(...)` keeps the
  subclass correct.
- **`state()` <-> `from_state` round-trip with matching keys**, so `from_state` is
  `cls(device, **state)` and `__init__`'s `as_tensor` swallows numpy or cpu tensors.
- **Squeeze to emulate only unmasked entries; unsqueeze for the chi2.** The network
  predicts the smaller unmasked block; `chi2` unsqueezes to the full vector and
  applies the full masked precision — the chi2 is unchanged, only the model shrank.
  Index the data, the mean, and the precision by the *global* `dest_idx` (positions
  in the full 3x2pt vector), never a block-local index. `keep_cols` (offsets within
  the probe's block) is only the intermediate that builds
  `dest_idx = block_global[keep_cols]`.
- **Probe generalization is a review axis, not a freebie.** `xi` is block 0, so
  block-local offsets equal global indices and much "works by coincidence" — any
  step correct only because the block starts at 0 is a latent `ggl`/`wtheta` bug.
  On review, trace every probe-dependent quantity: index the data and the
  full-vector mean by the global `dest_idx`; assert the dataset width equals
  `total_size`, so a cosmic-shear-only file fails loudly instead of silently
  mis-indexing; confirm cosmolike's `get_mask` / `get_cov_masked` / `get_inv_cov`
  and the block sizes return full-3x2pt-length arrays even under a single-probe
  init (the front block survives a short return, later blocks do not); and do not
  reuse the `PROBE_BLOCKS` key as cosmolike's `possible_probes` name unless they
  match (galaxy-galaxy lensing may be `gammat`, not `ggl`). Take the full-dv length
  from `total_size`, not a hardcoded literal.
- **Whiten targets in the covariance eigenbasis** so every output is unit-variance
  and decorrelated (equally hard to fit); the loss un-whitens the residual, so the
  full chi2 is exact regardless. `unwhiten` exactly inverts `whiten` because `U` is
  orthonormal.
- **The center is a training quantity that cancels in the loss.** Use the training
  mean (not the dataset fiducial); pass the full-block mean and let the class
  squeeze it; it never touches the chi2 (cancels in the residual), only the targets'
  zero-point — so it travels in the saved geometry, not the loss.

The arc that produced this (each a one-correction step worth recognizing early):
monolithic loss class -> split base/subclass; centering in the driver -> moved into
the class; fiducial center -> training-mean center; treating `get_cov_masked` as raw
-> using the clean unmasked sub-block; a confusing `__init__(state=None)` switch ->
explicit `from_cosmolike` / `from_state`; block-local `keep_local` in `squeeze` ->
global `dest_idx` (an `xi`-only bug that silently breaks `ggl` / `wtheta`).

## No all-caps for emphasis

Do not capitalize ordinary words for stress, anywhere: code comments, raw blocks,
or prose. Emphasize through word choice and sentence structure. Genuinely
uppercase notation and acronyms are fine (`H` for the Hessian, `1D`, `SGD`, `MSE`,
`GPU`, `NLL`). The single sanctioned exception is the `WARNING` marker that flags
a function reading a module global (see "No global variables in functions"): there
`WARNING` is a required flag label, not emphasis.

## "Raw blocks" (slide prose) are pure plain text

When asked for a "raw block" (text destined for a Jupyter raw cell shown verbatim
on a slide), output pure plain text with no markup:

- No code fences or backticks, no bold, italics, or markdown headers.
- No leading `- ` or `N.` at line start — those render as bullets / numbered
  lists. Use plain prose; if you must enumerate, do it inline (`(1)`, `(2)`) or
  in-sentence.
- Plain title-case section labels are fine.
- Hard-wrap to ~60 characters (raw cells do not auto-wrap).
- Keep it succinct; a slide holds little text. When in doubt, cut.

## Viewing the slides

To check slide layout, render the deck PDF with PyMuPDF (`pip install pymupdf` in
a throwaway venv; the system may lack poppler):
`doc[i].get_pixmap(matrix=fitz.Matrix(150/72, 150/72))` -> save PNG -> read the PNG.

## Review before a long run

Before an expensive training run, review the whole pipeline for correctness, not
just style: undefined names, missing imports, and cell-execution order; shape /
dtype / device consistency end to end; nan paths (a zero normalization scale,
`sqrt` of a non-positive eigenvalue, `0/0`); that the loss / whitening math is
actually what it claims; and the two recurring bug classes in the next section
(the degenerate-case coincidence, and shared-resource accounting across sequential
calls). Prefer to verify the math numerically in isolation —
round-trip `decode(encode(x)) == x`, chi2 against a direct masked reference,
chi2 against cosmolike's own value — rather than trusting a read.

**After a mechanical refactor, check it mechanically — the eye misses crossed
names.** A wide find-and-replace (a rename, swapping `isinstance(x, T)` for a
capability flag, moving methods between classes) produces edits that *compile*
but break at runtime, and reading dozens of near-identical cells does not catch
them. Three cheap automated passes do, and they caught real bugs here:
(1) **compile every cell** (`compile(src, ...)` after blanking `%`/`!` magic
lines) — catches the stray or unbalanced paren the eye slides over. (2) **a
binding check** — for each call/`getattr` you touched, confirm the variable is
actually a parameter or local of its *enclosing function* (walk the function's
`symtable`/AST: args + assigned names). This catches a name pasted from the
wrong template — e.g. `getattr(lossfn, …)` dropped into a function whose
parameter is `chi2fn` (compiles, `NameError` at run), and the mirror image. Note
a global read shows as "unbound" too, which is the same scan surfacing the
"reads a data global" smell. (3) **a leftover scan** — grep for the *old*
pattern you were replacing (`isinstance(_, OldType)`, `obj.old_attr`,
`OldClass.from_x(`); zero hits is the done signal. Distinguish raw/markdown
cells from code cells first (slide-prose "raw" cells look like broken code to a
naive scan, and a code cell that is actually prose is a latent `SyntaxError`).
The discipline: a refactor is finished not when it reads right, but when
compile + binding + leftover-scan are all clean.

## Recurring bug classes (review for these, in any codebase)

Two failure modes recurred here and generalize well past this material; both are
easy to miss because the code runs — they need a reviewer's eye, not a stack
trace.

- **The degenerate-case coincidence — "works for the default, breaks for the
  rest."** When code is parameterized over a family of cases (probes, loss modes,
  data shards, tensor axes, config variants) but only one is ever exercised, an
  assumption true only for that case slips through — most dangerously when the
  exercised case has a degenerate property that makes two distinct things
  accidentally equal: a zero offset, the first or identity element, a single-item
  collection, a default name that happens to match another. The code is correct by
  coincidence. On review, trace every parameterized quantity and ask whether it is
  right for a general member of the family or only because the current one is
  special; assert the invariant that must hold for all of them; and reason through
  (or run) a non-degenerate member.
  Example: `xi` is block 0 of the 3x2pt data vector, so its block-local column
  indices equal the global ones — hiding a bug where `squeeze` / `center` used the
  block-local index instead of the global `dest_idx`. It works for `xi` and
  silently mis-indexes `ggl` / `wtheta` (blocks 1, 2). Same family: a config key
  reused as an external API string (the `PROBE_BLOCKS` key doubled as cosmolike's
  `possible_probes` — fine for `"xi"`, but galaxy-galaxy lensing is `"gammat"`,
  not `"ggl"`); and arrays assumed full-length that only the front block survives
  when the source returns a short one. Catches: an `assert width == total_size`
  invariant, decoupling names that only coincide for the default, and indexing by
  the global index everywhere. A close cousin is **inferring a capability from a
  type**: an `isinstance(obj, BaseClass)` test used to mean "this one needs the
  extra argument / takes the special branch" works only while *every* capable
  variant subclasses that base — add a sibling that doesn't, or a subclass that
  opts out, and it silently takes the wrong branch with no error. Prefer an
  explicit capability flag the object declares
  (`getattr(obj, "needs_params", False)`) over a type test standing in for a
  behavior; a new variant then opts in by setting the flag, and one that forgets
  fails visibly. A matching design move: when one class grows two
  responsibilities (a data-vector *geometry* and a *loss*), prefer composition —
  the loss HOLDS the geometry and forwards the few attributes callers read —
  over inheritance, so one geometry can be wrapped by several losses and a future
  variant isn't forced into the base type.

- **Shared-resource accounting across sequential calls.** When one operation that
  draws from a finite shared resource (memory, GPU VRAM, file handles, a
  rate / connection / token budget) is split or duplicated into several sequential
  calls, the accounting must thread the running remainder: call k plans against
  the budget minus what calls 1..k-1 consumed. A by-value `budget` / `limit`
  handed to a sequence of allocators is the smell — each sees the full pool, so the
  total overruns it. And when a budget meets a shared resource, finish the
  accounting in both directions — name what is over-counted and what is
  under-counted — never stop at the reassuring half.
  Example: `build_loaders` was split into a per-source builder called once for the
  train file then once for the val file; the val call planned against the full
  VRAM budget while the train set was already resident on the GPU — an
  over-estimate, an OOM on a tight card. Fix: the per-source builder returns the
  bytes it made resident, and the orchestrator passes `budget - used_train` to the
  next. The tell that was missed: a docstring already noted the shared model +
  `Cinv` were "counted in every call, so conservative" — but that stopped at the
  safe half and never asked what was un-counted (the resident train pool).

## Performance: diagnose first, then make a real run fast

The notebook teaches at toy scale (10k dvs), but real research is millions of
dvs and runs measured in days. The transferable lesson is a method, not a trick:
measure, fix the top bottleneck, re-measure — because the bottleneck moves.

- **Profile, do not guess.** `torch.profiler.profile(activities=[CPU, CUDA],
  schedule=schedule(wait,warmup,active))` over a few steady-state steps, tag
  sections with `record_function`; or `perf_counter` + `torch.cuda.synchronize()`
  for a coarse split. Sort by `self_cuda_time_total`; export a Chrome trace to
  see the idle gaps directly.
- **Read the meters correctly.** GPU-util = fraction of time busy (can be high on
  slow work); power = how much of the die is lit; the real metric is
  wall-clock/throughput. High util at low power = a few execution units pinned
  (e.g. FP64); low util = launch/dispatch-bound (work too cheap to keep the GPU
  fed). High util is not the goal; low wall-time is.
- **The bottleneck moves.** This session: FP64 chi2 dominated (90% util, low
  power) -> made it float32 (~64x) -> became launch-bound (30% util) -> bigger
  batch + compile. Each fix exposes the next.
- **FP64 is ~1/64 on consumer GPUs** (FP32 cores / FP64 units per SM, e.g. 128/2
  on a 3060 — a market-segmentation choice, not physics). Keep heavy math (the
  chi2 einsum) in float32; do the one-time `eigh` in float64 (numpy, at
  construction) but store the basis + `Cinv` at a chosen precision: make `dtype`
  a constructor arg defaulting to `torch.float32` (float64 only when a probe
  needs it). Verify the precision is enough against a float64 twin — build both,
  compare chi2; a ~1e-7 relative gap means float32 is safe. (The case study above
  stores `Cinv` float64 for clarity; in a real run make it the `dtype` arg.)
- **Tensor-core float32:** `torch.set_float32_matmul_precision("high")` once,
  globally -> TF32 matmuls (~10-bit mantissa, large speedup, negligible accuracy
  cost here).
- **AMP** (`with torch.autocast(device.type, dtype=torch.bfloat16): pred =
  model(x)`) wraps only the model; the loss stays outside. bf16 needs no
  `GradScaler` (it keeps float32's exponent range). Only helps once the model —
  not the loss — is a real fraction of per-batch time.
- **torch.compile** is useless while one big kernel dominates; it helps once
  launch-bound (fuses small ops, cuts launches). `mode="reduce-overhead"` (CUDA
  graphs) needs static shapes, so drop the ragged last minibatch:
  `n_full = (n // bs) * bs; for s in range(0, n_full, bs): ...` (the dropped rows
  rotate each epoch since the batch reshuffles, so nothing is permanently lost).
- **Feed the GPU.** Stage a big chunk on-device once, then minibatch by
  on-device slicing — no per-batch host->device copy. (The student anti-pattern:
  CPU-resident tensors + a `DataLoader(num_workers=0)` + per-batch `.to(device)`
  stalls the GPU on the unoverlapped copy.) And no per-batch host sync: accumulate
  the running loss on-GPU (`run_sum += loss.detach()*n`), `.item()` once per
  epoch — `loss.item()` every batch blocks the CPU on the GPU each step.
- **Scale past GPU RAM (the real case).** A regime ladder chosen at runtime from
  `psutil.virtual_memory().available`: fits GPU -> cache on GPU; fits RAM ->
  cache in pinned RAM, stream RAM->GPU; exceeds RAM -> memmap from disk. Two
  orthogonal gates: a RAM-vs-disk flag (host) and the chunk size (GPU; the
  `batches_per_load` budget — which must also subtract the resident `Cinv`).
  When the loaders are built once per source (a train file plus a separate val
  file at T/2), that budget is a shrinking pool, not a constant: each source
  must subtract what the previous one made resident before the next is sized — a
  by-value budget handed to a sequence of allocators against one GPU lets the
  total overrun it (the train set is already on the card when you size the val).
  Have the per-source builder return the bytes it made resident and pass
  `budget - used_prev` to the next.
  Double-buffer the next chunk on a side CUDA stream so I/O overlaps compute (it
  costs chunk size: two chunks resident). Pre-encode targets once to a smaller
  file (shrinks I/O, drops the per-chunk whiten; bakes the geometry, so a
  final-campaign step). Then big batch (+ LR scaling) and `Adam(..., fused=True)`.
- **Train and validate on different distributions cleanly.** When the val set
  must come from a separate file (e.g. a tempered T/2 draw that stays in the
  inference-relevant interior, not the broad training prior), build the geometry
  (whitening center, covmat, `Cinv`) once from the training source and apply it
  unchanged to the val file — never re-whiten val with its own statistics, or the
  model sees a transform it never trained on. Plumbing: a single-source loader
  builder called once per source, an orchestrator returning one `data` dict
  (`data["train"]` / `data["val"]`, same keys each), and a source-agnostic
  `eval_val` taking a source sub-dict. Derive the training stats from the
  canonical training object, not loose globals, so they cannot drift to the wrong
  file.
- **A precision perturbation diverges chaotically.** bf16/TF32 change the result
  from step 1; with a reactive `ReduceLROnPlateau` on a noisy metric the two runs
  then take different LR paths — so one run is not an A/B. Judge a precision/speed
  knob with a fixed schedule and a few seeds, not a single number.

## Robust losses and honest metrics

The chi2 loss is outlier-dominated (a sum of squares), so a few bad cosmologies
can hijack the gradient. Three levers, a metric discipline, and a ceiling.

- **Where the nonlinearity goes matters.** `sqrt(mean(chi2))` is a monotonic
  rescale of the mean — same gradient direction, outliers still dominate; it only
  changes the effective LR. To actually down-weight outliers, apply the transform
  per sample then average: `torch.sqrt(c).mean()` (`c` = per-sample chi2). Prefer
  the pseudo-Huber `(torch.sqrt(1 + 2*c) - 1).mean()` — quadratic for small c
  (stable; finite gradient at 0, where pure `sqrt` blows up) and linear for large
  c (robust).
- **Trimmed loss** drops the worst k% of the batch before averaging:
  `k = max(1, int(round((1-trim)*c.numel()))); c,_ = torch.topk(c, k,
  largest=False)`. `topk` picks which to keep (a non-differentiable choice);
  gradients still flow through the kept values. It is a hard reject (zero
  gradient), not a soft down-weight: powerful for contaminated data, a trap for
  hard-but-real regions (median plummets while the mean/tail freeze near init).
  Always diagnose which points are trimmed — pull the parameters of the worst val
  cosmologies. Clustered at parameter extremes (extreme IA, high H0) = real hard
  physics the trim is hiding; scattered = contamination the trim is right to drop.
- **Focal / hardness reweighting** up-weights the still-hard points so the loss
  keeps chasing the tail instead of being out-voted by the solved bulk: a weighted
  mean with `w = (c/(c+kappa))**gamma`, `w` detached (a priority, not a quantity
  the optimizer can game by shrinking it), normalized by `sum(w)`. Set `kappa` at
  the threshold you report (`frac>0.2` -> `kappa ~ 0.2`), not higher, or the
  pressure sits on the wrong band. Failure mode to watch: a monotone focal weight
  de-protects the points just below the threshold — they lose gradient and drift
  up across it — so it can make `frac>0.2` worse even while chasing the points
  above; a soft-threshold bump centered on the line (protect both sides) matches
  the metric better. Anneal `gamma` from 0 (warm up democratically), and clamp
  with `max(gamma, 0)` instead of an `if gamma <= 0` branch — at `gamma=0`, `w=1`
  collapses to the plain mean, so no fragile float-equality test is needed.
- **Loss-shaping has a ceiling: it cannot fit what training does not cover.** If
  the persistently-hard points sit where the training set is sparse (e.g. the
  edges of the sampled prior, which a tighter val draw then probes), no transform,
  trim schedule, or reweighting fits them — in this material trim-annealing and
  focal both made the reported metric worse, each only reshuffling which points
  are bad. Before spending loss-tuning compute, diagnose coverage: pull the
  parameters of the worst val points (extremes = under-covered, scattered =
  contamination), and evaluate on the distribution where the model is actually
  used, not the broad training prior — a "floor" is often an evaluation artifact
  of grading at the edges. Fix coverage (more data there, or eval where you
  infer); only then do the loss levers move it.
- **Report median and mean.** Emulator error is heavy-tailed: a median of 0.42
  beside a mean of ~1900 means a great bulk and an abandoned tail. The median
  alone oversells. The inference-relevant metric is the fraction over a threshold
  (`frac > 0.2`, `> 1`), and evaluation is never trimmed — eval must see the whole
  distribution, training-time tricks must not.
- **Verify numerics in isolation** (also in "Review before a long run"):
  round-trip `decode(encode(x))`; chi2 vs a direct masked sub-block; a float32
  path vs a float64 twin; the class path vs cosmolike's own value. Never trust a
  precision or refactor change without a reference comparison.

## Diagnosing an accuracy floor (loss vs optimization vs data vs capacity)

When a metric plateaus above target, find the cause before reaching for a fix:
the candidate causes (loss shape, optimization, data coverage, model capacity)
have mutually-exclusive cures, and most of the time the intuitive cause is wrong.
This ladder, worked end to end on the cosmic-shear emulator's `frac>0.2`, is the
transferable method.

- **Report a threshold ladder, not one number — tail or shoulder?** Count the
  fraction over several cutoffs at once (here 0.2, 0.5, 1, 10, 100). A heavy
  *tail* (a few catastrophic points) is a different problem from a *shoulder*
  piled just above the threshold. Fit a log-normal to the bulk (its median sets
  one parameter, the `frac>threshold` sets the other); if that single fit then
  predicts the *other* thresholds' fractions, the shoulder is just the upper tail
  of **one broad error distribution**, not a separable sub-population. The
  consequence is decisive: **you cannot narrow a spread by reweighting**, so any
  loss-shaping (focal, trim, threshold bump, per-element weight) is dead on
  arrival for a shoulder — and indeed all of them came back neutral.

- **Separate optimization noise from the floor.** Late-training per-epoch
  thrashing is a step-size effect (bounce amplitude ~ `lr × gradient_noise`), not
  the floor. Halve the LR: if the bounce tightens but the plateau does not move,
  the floor is underneath, and the reported "best" was the lucky bottom of a
  noisy, best-epoch-selected curve. Batch size is usually a *dead* lever here —
  under a correct `lr`-vs-`batch` coupling every batch size converges to the same
  result (the coupling holds the gradient-noise scale fixed); and those coupling
  rules (`lr ∝ sqrt(bs)` / `∝ bs`) are SGD heuristics — **AdamW needs little to no
  LR increase with batch**, so naive scaling overshoots and "bigger batch breaks."

- **Diagnose residuals in the metric's own coordinates, not a convenient
  marginal one.** A chi2 is a sum of squares in the *decorrelated* (whitened)
  space, so its true drivers are the whitened residual components — not the raw
  per-element residual in each element's own error bar. A marginal per-element
  lens will confidently flag a block (for us: the highest source-z bin at small
  theta) that the metric *barely charges for*, because neighbouring, strongly
  correlated elements share a cheap common-mode direction. The tell that you are
  looking at a ghost: **the network leaves large marginal residuals on that block
  even on the training set and the loss tolerates it** — if the loss cared, the
  optimizer would have crushed it. Also verify the target basis *is* the metric's
  basis (`chi2 == ||pred − target||²` to rounding); if not, that gap is a real
  conditioning mismatch worth fixing, not a physics floor.
  **Which errors the chi2 forgives is set by the covariance's eigen-spectrum, and
  for a correlation-function data vector the chi2 is a high-pass filter on the
  error.** Neighbouring points are strongly correlated, so the smooth
  (low-frequency) covariance eigenmodes carry the largest variance — hence the
  smallest precision — and the chi2 barely charges a smooth / common-mode error;
  the oscillatory (high-frequency) eigenmodes are small-variance, high-precision,
  and the chi2 crushes them. So the metric's blind spot is *smoothness*, not
  oscillation. Practical corollary that saves a wasted experiment: a smoothness /
  anti-oscillation regularizer (penalise the second derivative of the prediction)
  is **redundant** — it duplicates what the chi2 already does to oscillatory error
  and risks over-smoothing toward the very common-mode the chi2 ignores. The loose
  screw is the smooth coherent residual, which is physically real so you cannot
  just penalise it (and the one attempt to add pressure on the chi2's blind spot —
  a per-element weight — came back neutral).

- **The decisive test is the learning curve, not the train-vs-val gap.** A small
  `val − train` gap tells you variance is low (you are not overfitting), but it
  does *not* settle capacity vs data: a regularized model that fits its small
  training set only partway looks identical (`train ≈ val`, both high) whether the
  limit is capacity or data sparsity. The question is answered only by the
  *learning curve* — the metric versus training-set size. Still falling at the
  largest size you can afford → **data-limited** (more data helps); flattens →
  **capacity-limited**. It is cheap (a handful of trains) and definitive.
  Cautionary tale from this material, kept because the trap is common: `train ≈
  val` at one training size was read as a capacity limit and written down as
  "earned" — then the learning curve dropped the metric from 0.22 at 10k
  cosmologies to 0.10 (the target) at 46k. Data-limited the whole time. Do not let
  the train-vs-val gap stand in for the learning curve: the gap diagnoses
  overfitting, the curve diagnoses the floor.

- **When data is the binding constraint, the learning curve *is* the objective —
  "data-limited" does not always mean "add data."** When each training example is
  expensive (large simulations) or the parameter volume is huge (a high-D, hot
  prior), you can never afford the data the asymptotic floor would need, so the
  goal becomes **sample efficiency**: the *position* of the learning curve — the
  smallest training set that reaches target. Compare methods (preprocessing,
  architecture, input features, sampling) by N-to-target, not by the floor at one
  size: a lever that is *neutral at the floor* can still shift the curve far left,
  and that left-shift is the whole deliverable. Physics-informed preprocessing is
  the canonical case — it removes variance the network would otherwise learn from
  data, so its payoff is largest in the *low-data* regime and must be measured
  there, not at whatever single training size was convenient. (Worked example: a
  rescaling that "did nothing" at the 10%-of-pool size is exactly the kind of lever
  whose value only shows as a left-shift at small N.)

- **For a structured high-dimensional output, the lever is an axis-aware *head*,
  not a wider dense net — and the tell is a learning-curve *slope*, not a single
  point.** A dense MLP is output-permutation-invariant: reorder its outputs (and
  the targets and metric) and it trains identically, so it cannot use any
  structure along the output axis (smoothness/adjacency in ℓ, θ, λ…). It produces
  a decent prediction but leaves residuals *structured along that axis*, which it
  can't correct because every output is an independent readout. A head that *sees*
  the axis — a 1-D CNN or a transformer over the output sequence, **on top of a
  shared trunk** — corrects those structured residuals with shared weights, far
  more sample-efficiently. The data signature is a **slope** difference: the dense
  net's error-vs-`N_train` curve *flattens* (an architecture-limited floor — more
  data stops helping), while the axis-aware head's curve keeps descending. So
  compare architectures by the *shape* of `f`-vs-`N` on log–log (power law
  `f ~ N^(-α)` → a line of slope `-α`; the exponent is the efficiency rate), not
  the value at one N — a curve still falling where the baseline has plateaued is
  the win even before either reaches target. (Worked example: ResMLP+CNN/
  transformer dropped `f(Δχ²>0.2)` ~0.2→~0.06 vs a bare ResMLP, the bare one
  plateauing and the axis-aware head not. And the *failed* counter-move — splitting
  the whole network into independent per-output-group sub-nets — gains nothing: it
  discards the shared trunk that learns the (shared) input→latent map, and a dense
  sub-net still can't use the axis. Keep one shared trunk; make the *head*
  axis-aware.)

- **Watch the tempering confound.** If validation is drawn at a different
  sampling temperature than training (e.g. `T_val = T_train/2`), val has a smaller
  spread *by construction*, so `val < train` per element is just the temperature,
  not overfitting or its opposite. Always compare at the metric (`frac>0.2`),
  never at per-element rms across differently-tempered sets.

The arc to recognize: a stuck metric invites loss-knob roulette (LR, focal kappa,
bumps, per-element weights), and every turn here was neutral — but neutral on the
loss does not mean capacity. The progress was *elimination*; the resolution was
the one cheap experiment that directly varies the suspected cause — the learning
curve (error vs training-set size) — and it named the floor **data**, not
capacity. The number that ends the argument is the metric's learning curve, not
any single-split snapshot. The recorded trap: reading a small train-vs-val gap as
proof of capacity and asserting it before plotting error-versus-`N_train`.

## Quick checklist before sending notebook code

1. Whole cell, paste-ready (not a snippet)?
2. Every line within the width budget?
3. Non-obvious mechanics named; no unexplained jargon?
4. Numeric constants named and motivated?
5. Same APIs/idioms the notebook already uses?
6. No all-caps emphasis?
7. If a raw block: pure text, no markup, no bullet markers, hard-wrapped?
8. Only what was asked — no unprompted features, options, or scope?
9. No function silently reads a data global (only params, locals,
   imports, helpers)? Any unavoidable global read flagged in-code
   with the ⚠️ WARNING marker?
10. Shared logic in one function, called more than once — not duplicated?
11. Comments formal and definitional (`name = ...`), not chatty?
12. Heavy math in float32 unless float64 is needed (and verified vs a float64
    twin)? Precision a `dtype` arg, not hardcoded?
13. Robust loss applied per-sample (not `sqrt`-of-mean); eval never trimmed;
    median and mean both reported?
