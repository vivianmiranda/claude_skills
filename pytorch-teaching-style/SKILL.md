---
name: pytorch-teaching-notebook-style
description: >-
  Use when writing, explaining, revising, or commenting PyTorch code meant for a
  teaching Jupyter notebook presented as slides (e.g. RISE). Enforces a house
  style: slide-narrow code, maximally didactic step-by-step comments that assume
  novice familiarity with every API, full-cell rewrites for any change, pure
  plain-text "raw blocks" for slide prose, no all-caps emphasis, and grounding
  abstractions in a concrete research example. Trigger on PyTorch lecture/teaching
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

## Ground abstractions in the research case

When a comment introduces an abstract quantity or formula, add one short, symbolic
sentence connecting it to the reader's real research (here: emulating cosmic-shear
data vectors). Example: "in this toy L = 1, but for a cosmic-shear data vector
L = n_pairs * n_theta". Keep it symbolic and slide-tight; do not expand into a
wall of numbers (N_z=5 -> 15 pairs x 20 angles = 300) unless it clearly fits and
is asked for.

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

## Quick checklist before sending notebook code

1. Whole cell, paste-ready (not a snippet)?
2. Every line within the width budget?
3. Non-obvious mechanics named; no unexplained jargon?
4. Numeric constants named and motivated?
5. Same APIs/idioms the notebook already uses?
6. No all-caps emphasis?
7. If a raw block: pure text, no markup, no bullet markers, hard-wrapped?
8. Only what was asked — no unprompted features, options, or scope?
