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

## Prefer explicit arguments over globals

- **A function should take what it needs as parameters, not read
  module-level globals.** Reaching out to enclosing-scope names (stats
  tensors, data arrays, `device`) hides the function's real inputs and
  breaks the moment it is reused, moved, or called before those globals
  exist. Pass them in instead.
- Default to explicit arguments even when it means a longer signature.
  Only fall back to a global when threading the value through would be
  absurdly awkward.

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
  split chained calls, or factored-out intermediates. Use 2-space indentation.
  Verify the width before sending.

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

  def __init__(self, device, total_size, keep_local,
               dest_idx, evecs, sqrt_ev, Cinv, center):
    # plain: place fields on device. as_tensor accepts numpy
    # (from cosmolike) or cpu tensors (from a saved state).
    self.total_size = int(total_size)
    self.keep_local = torch.as_tensor(
      keep_local, dtype=torch.long, device=device)
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
    # read cov/inv_cov/mask/sizes; keep_local = unmasked
    # block positions (index dv0 cols); dest_idx = their
    # slots in the full dv; eigh the unmasked block cov;
    # squeeze dv_center; then build via __init__:
    return cls(device, total_size, keep_local, dest_idx,
               U, sqrt_lam, Cinv, center)

  def state(self):                  # tensors to save
    return {"total_size": self.total_size,
            "keep_local": self.keep_local.cpu(),
            "dest_idx": self.dest_idx.cpu(),
            "evecs": self.evecs.cpu(),
            "sqrt_ev": self.sqrt_ev.cpu(),
            "Cinv": self.Cinv.cpu(),
            "center": self.center.cpu()}  # keys==__init__

  def squeeze(self, dv):   return dv[:, self.keep_local]
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
  Keep `keep_local` (indexes raw block columns) and `dest_idx` (indexes the full
  vector) distinct.
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
explicit `from_cosmolike` / `from_state`.

## No all-caps for emphasis

Do not capitalize ordinary words for stress, anywhere: code comments, raw blocks,
or prose. Emphasize through word choice and sentence structure. Genuinely
uppercase notation and acronyms are fine (`H` for the Hessian, `1D`, `SGD`, `MSE`,
`GPU`, `NLL`).

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
`sqrt` of a non-positive eigenvalue, `0/0`); and that the loss / whitening math is
actually what it claims. Prefer to verify the math numerically in isolation —
round-trip `decode(encode(x)) == x`, chi2 against a direct masked reference,
chi2 against cosmolike's own value — rather than trusting a read.

## Quick checklist before sending notebook code

1. Whole cell, paste-ready (not a snippet)?
2. Every line within the width budget?
3. Non-obvious mechanics named; no unexplained jargon?
4. Numeric constants named and motivated?
5. Same APIs/idioms the notebook already uses?
6. No all-caps emphasis?
7. If a raw block: pure text, no markup, no bullet markers, hard-wrapped?
8. Only what was asked — no unprompted features, options, or scope?
9. Functions take needed values as arguments, not module globals?
10. Shared logic in one function, called more than once — not duplicated?
11. Comments formal and definitional (`name = ...`), not chatty?
