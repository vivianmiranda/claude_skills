# claude_skills

A small collection of Claude Skills for cosmology research, scientific
computing, and code work. They live as plain markdown so you can use
them in any Claude chat by pointing Claude at the raw file URL.

## What's a Claude Skill?

A skill is a folder containing a `SKILL.md` (the entry point) and
optional `references/` files (deeper material loaded only when needed).
The SKILL.md captures *how to do a specific kind of work well* —
conventions, traps, the shape of a good answer — so you don't re-explain
those things every chat. 

## Skills in this repo

- **camb-dev** — modifying CAMB (Fortran Boltzmann code) and integrating
  it with Cobaya inside Cocoa. Covers reionization basis, primordial
  P(k), thermo/lsampling, w(a) tables, the axion phantom-mirage model.
- **cosmolike-dev** — development, optimization, review, and debugging
  for the CosmoLike/Cocoa C codebase: `cosmo2D.c`, `pt_cfastpt.c`,
  OpenMP/SIMD, FAST-PT, TATT/NLA, `perf` benchmarking.
- **porting-legacy-physics-code** — porting physics modifications from
  a legacy fork onto a modern upstream codebase, with verbatim-numerics
  discipline and regime-complete ratio validation.
- **docker-skill** — building, debugging, and editing Docker images for
  ML and scientific computing (CUDA wheels, GPU passthrough, image
  bloat, bind-mount semantics).
- **explaining-code** — explaining code, APIs, libraries, and
  frameworks at a maximally didactic level, naming non-obvious mechanics
  (iterators vs lists, in-place vs copy, hidden defaults, `restrict`
  semantics, linkage of `inline`).

## How to use

At the start of a chat, paste a one-liner telling Claude to fetch the
skill and follow it:

> Fetch https://raw.githubusercontent.com/vivianmiranda/claude_skills/main/cosmolike-dev/SKILL.md
> and follow it for this chat; fetch `patterns.md` and `pitfalls.md`
> from the same `references/` folder when it tells you to.

Replace `cosmolike-dev` with whichever skill you need. The SKILL.md
itself tells Claude when to pull in the reference files, so you don't
have to enumerate them yourself — but naming them explicitly (as above)
is the most reliable way to make sure Claude grabs everything in long
sessions.

You can load several skills at once:

> Fetch and follow these for this chat:
> - https://raw.githubusercontent.com/vivianmiranda/claude_skills/main/camb-dev/SKILL.md (plus all files in its `references/`)
> - https://raw.githubusercontent.com/vivianmiranda/claude_skills/main/explaining-code/SKILL.md
