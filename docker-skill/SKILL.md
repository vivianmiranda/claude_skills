---
name: docker-ml
description: Methodology for building, debugging, and editing Docker images for ML and scientific computing — especially GPU-enabled containers running PyTorch / TensorFlow / JAX, dependency resolution between framework CUDA wheels, NVIDIA Container Toolkit setup, image-size diagnosis, and Linux-vs-Mac bind-mount behavior. Use this skill whenever the user mentions Dockerfile, docker build, docker run, GPU passthrough, --gpus all, nvidia-container-toolkit, base images like nvidia/cuda, container CUDA versions, image bloat, wheel conflicts (especially tensorflow[and-cuda] vs torch), Jupyter-in-Docker, read-only bind mounts, UID mismatches, or anything else container-related. Also trigger when the user asks to fix, modify, simplify, or trim a Dockerfile, even if "docker" isn't mentioned explicitly — for any block-level edit to a Dockerfile, this skill defines the output convention (always return the entire RUN block, never a snippet).
---

# Docker for ML / Scientific Computing

Conventions and methodology for editing Dockerfiles, debugging GPU container
issues, and resolving CUDA / framework dependency conflicts.

## When to read what

Before any non-trivial Docker work, read `references/docker-reference.md`.
That file contains:

- Editing conventions Claude must follow (drop-in RUN blocks, no
  pattern-matching to fake bugs, honest retractions)
- The GPU stack model (driver → toolkit → container → wheel)
- Installing the NVIDIA Container Toolkit
- Base image selection (`-runtime` vs `-cudnn8-devel`)
- Dependency resolution failures between TF and PyTorch CUDA wheels
- Image-size diagnostics and leverage points
- UID handling for bind mounts (and why it differs on macOS vs Linux)
- Linux-vs-Mac Docker behavior differences
- Inspection commands, verification scripts, antipatterns

`references/docker-reference.md` is the authoritative source for all of the
above. The body of this SKILL.md is a brief pointer — go to the reference
file for actual guidance.

## Mandatory rules (always apply)

These rules apply on every Dockerfile edit, regardless of context:

1. **Output entire RUN blocks, never patched snippets.** When fixing or
   modifying a `RUN` (or any multi-line shell instruction), output the
   complete block as a single drop-in replacement. The user wants to paste
   the whole block in one operation, not splice fragments by hand.

2. **Don't pattern-match to "common Dockerfile bugs."** Specifically: comment
   lines between backslash-continued lines do NOT break a `RUN`. BuildKit
   strips them before passing to the shell. Verify before flagging.

3. **Honest retractions.** If a prior claim turns out to be wrong, state the
   correction directly, name what was missed, and move on. No grovelling.

4. **⚠️ matplotlib means LaTeX, by default.** If a Dockerfile contains
   matplotlib, assume `rcParams['text.usetex'] = True` is on. The build
   MUST install `cm-super`, `dvipng`, `ghostscript`, and the
   `texlive-latex-base / texlive-latex-extra / texlive-latex-recommended /
   texlive-fonts-recommended / texlive-fonts-extra` set. **Do not silently
   trim these for image size.** When trimming a Dockerfile that includes
   matplotlib, explicitly flag:
   *"I'll assume `text.usetex=True` is on (default unless you tell me
   otherwise). LaTeX packages stay in. Confirm if you want them dropped."*
   See `references/docker-reference.md` for the full apt list.

Details and examples are in `references/docker-reference.md` under
"Editing conventions Claude must follow" and "LaTeX for matplotlib."

## Quick triage flow

When the user reports a Docker problem, walk this in order:

1. **Read the actual error.** Don't pattern-match. The error message points
   to the layer that failed.
2. **Identify the failing layer.** Host driver / toolkit / container CUDA /
   pip wheel — they fail differently. The stack model in the reference file
   names each.
3. **Propose the minimum diff.** Output the full block being changed; explain
   the diff from the previous version; estimate size/time impact when
   interesting.
4. **Verify.** End with a verification script the user can run to confirm
   the fix.

## Workflow expectations

- Ask before recommending changes that require host-level reboots or driver
  updates on a shared machine. If the user has flagged a host as
  "group depends on it," workarounds at the container/wheel layer are the
  default answer.
- Don't burn cycles on dependency resolver brute force. If TF and PyTorch
  CUDA wheels conflict, jump to the "drop `[and-cuda]` from TF" option after
  one TF version bump attempt.
- For image bloat, the highest-leverage changes are base-image choice
  (`-runtime` over `-devel`) and consolidating apt blocks. Mention these
  before suggesting fine-grained package pruning.

## Cross-platform reminders

- **Linux is the strict reference behavior.** macOS Docker Desktop hides
  several POSIX issues (UID translation on bind mounts, port binding, etc.)
  that surface immediately on Linux hosts.
- **NVIDIA Container Toolkit only works on Linux.** Docker Desktop on macOS
  cannot pass through Metal or CUDA. Containers run CPU-only on Mac
  regardless of what's installed in them.
