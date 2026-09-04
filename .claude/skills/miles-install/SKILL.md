---
name: miles-install
description: Install the radixark/miles training stack (Megatron-LM, TransformerEngine, sglang-miles, apex, wheels, miles itself) onto an existing GPU box — a devbox, a bare container, any machine that already looks like the lmsysorg/sglang base image — by replaying miles' docker/Dockerfile as bash. Use when asked to "install miles on this machine / devbox", "set up the miles env without rebuilding the image", "make this sglang box a miles box", "regenerate the miles install script", or when the Dockerfile changed and the box must catch up. Self-adapting: the bash is generated from the Dockerfile at run time, never hand-maintained.
user_invocable: true
---

# miles-install — replay the miles Dockerfile on a live box

`docker/Dockerfile` in radixark/miles is the only true spec of a miles environment and it
changes weekly. So this skill ships **no install recipe of its own**. It ships:

| file | role |
|---|---|
| `scripts/dockerfile2sh.py` | generic, stdlib-only Dockerfile → idempotent bash translator |
| `scripts/miles-install.sh` | entry point: clone miles at `--ref`, translate `docker/Dockerfile` with the variant from `docker/build.py`, run it, verify imports |

Every run re-reads the Dockerfile, so an upstream edit (new RUN, new ARG, reordered step) is
picked up automatically. Steps are content-hashed: re-running skips steps whose text already
completed, executes only new/changed ones, and resumes at a failure.

## Run it

Target: any box that resembles the recipe's base image (`FROM lmsysorg/sglang:<tag>` — the
generated script prints which). Typical: an rx devbox on a `lmsysorg/sglang` or
`radixark/miles_diffusion` image. Steps run as root, use system pip/apt, take **~35 min** on
an H200 box (measured 2026-09-04, cu12-x86: mamba-ssm source build 14 min, sglang editable
install 5 min, fast-hadamard 3 min, nccl-tests 2 min; everything else seconds).

Inside a miles checkout (this repo), replay the checkout itself — no clone, and your local
Dockerfile edits are what gets installed:

```bash
bash .claude/skills/miles-install/scripts/miles-install.sh --src "$(git rev-parse --show-toplevel)" --dry-run
bash .claude/skills/miles-install/scripts/miles-install.sh --src "$(git rev-parse --show-toplevel)"
```

On a remote box without the repo:

```bash
# 1. ship the two scripts (no scp on rx devboxes; COPYFILE_DISABLE avoids macOS ._ files)
S=.claude/skills/miles-install/scripts     # or ~/.claude/skills/miles-install/scripts
B64=$(COPYFILE_DISABLE=1 tar czf - -C "$S" dockerfile2sh.py miles-install.sh | base64)
rx devbox run <box> -- bash -c "mkdir -p /root/miles-install-skill && cd /root/miles-install-skill && echo '$B64' | base64 -d | tar xzf -"

# 2. preview (what would run, which variant, base image) — fast, safe
rx devbox run <box> -- bash -c 'cd /root/miles-install-skill && bash miles-install.sh --ref main --dry-run 2>&1 | grep -vE "^      \|"'

# 3. run detached; the log is the source of truth
rx devbox run <box> -- bash -c 'cd /root/miles-install-skill && nohup bash miles-install.sh --ref main > run.log 2>&1 & echo started'

# 4. poll (per-step ok/failed lines; strip ANSI)
rx devbox run <box> -- bash -c 'grep -aE "step [0-9]+/[0-9]+ ok|failed with rc|ERROR|all .* complete|  ok  |  FAIL " /root/miles-install-skill/run.log | sed "s/\x1b\[[0-9;]*m//g" | tail -20'
```

### Options that matter

- `--ref <branch|tag|sha>` — recipe *and* installed miles revision (MILES_COMMIT defaults to
  the resolved sha, so what you install is what the Dockerfile you replayed describes).
- `--src <dir>` — use an existing checkout instead of cloning. MILES_COMMIT then defaults to
  its HEAD, which the recipe clones from GitHub, so an unpushed tree needs
  `--build-arg MILES_COMMIT=main` (or pip-install the tree yourself afterwards).
- `--variant auto|cu13-x86|cu12-x86|...` — keys of `VARIANTS` in `docker/build.py`. `auto`
  picks the single-platform variant whose `ENABLE_CUDA_13` matches the host torch's CUDA
  major and whose platform matches `uname -m`. Pass it explicitly when the box has no torch.
- `--build-arg K=V` — any Dockerfile ARG (`SGLANG_COMMIT=…`, `MEGATRON_COMMIT=…`,
  `SGL_ROUTER_USE_WHEELS=0`, …). Wins over the variant. Undeclared args are ignored like
  `docker build` does.
- `--from N` / `--only N` / `--force` — resume / rerun control. `--list` maps step numbers
  to Dockerfile line numbers. Same knobs via env on the generated script:
  `DF_FROM_STEP`, `DF_ONLY_STEP`, `DF_FORCE`, `DF_DRY_RUN`, `DF_LIST`.
- `--workdir` — default `/root/.miles-install`: `miles-src/` (clone = docker build context),
  `install.sh` (generated), `install.meta.json` (resolved args, base image), `install.log`,
  `state/step-NNN-<hash>.done` markers, `state/env.sh` (persisted `ENV` lines, sourced from
  `~/.bashrc`).

## What the translator does and does not do

Faithful: Docker-style line continuation (comments inside a `RUN … \` block dropped, lines
joined so inline python one-liners keep working); `ARG` = build-arg > variant > Dockerfile
default, baked in and **never inherited from the host env** (the devbox itself exports
`SGLANG_IMAGE_TAG`, which must not leak into the recipe); `ENV` exported now and persisted
unexpanded (`PATH="/root/.cargo/bin:${PATH}"` stays composable); `WORKDIR`; `SHELL`; `RUN`
bodies executed verbatim via `/bin/sh -c` with args exported; `RUN --mount=type=bind`
emulated by copying the context file for the step; `type=cache` dropped; `COPY` globs
against the context with BuildKit's empty-wildcard no-op (`COPY wheel[s]/`).

Refuses loudly (exit 2) rather than guessing: multi-stage (`FROM` ×2, `COPY --from`),
`ADD <url>`, `ONBUILD`, unknown instructions. `USER`, `LABEL`, `EXPOSE`, `CMD`, … are noted
and skipped. If the Dockerfile starts using one of those, extend `translate()` in
`dockerfile2sh.py` — that is the only place that would need a change.

## Reading failures

A failed step prints `step N/M (L<line>) failed with rc=…`. `L<line>` is the Dockerfile
line — open it, fix the environment cause, re-run the same command; it resumes at N.
Common causes on a box that is *not* exactly the base image:

- **Base-image drift**: the recipe's `FROM lmsysorg/sglang:<tag>` vs the box's
  `SGLANG_IMAGE_TAG` env (both are printed at start). Torch ABI mismatch shows up in step
  "transformer_engine" (`verify_transformer_engine.py` fails to import) or as
  `undefined symbol` on first import of flash_attn / apex. Fix = pick a box on the right
  base image or `--build-arg WHEELS_TAG_X86=<release built for this torch>`.
- **Missing base-image tooling** (seen on `kangrui-sd3-verify`, a miles_diffusion box): the
  sglang-miles `setup.py` shells out to `cargo` to discover Rust extensions; the
  `lmsysorg/sglang` base ships a Rust toolchain in `/root/.cargo/bin`, other images don't.
  Step "sglang" fails with `FileNotFoundError: 'cargo'`. Fix on the box, then re-run:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  PATH=/root/.cargo/bin:$PATH bash miles-install.sh --ref <ref>   # resumes at the failed step
  ```
- **Another project owns the `miles` package**: `radixark/miles_diffusion` also installs an
  editable dist named `miles`. The recipe's `pip install -e /root/miles --no-deps` then
  competes with it; `python -c 'import miles; print(miles.__file__)'` decides. Uninstall the
  diffusion editable (`pip uninstall -y miles_diffusion miles`) if the LLM stack must win.
- **sglang checkout**: the recipe does `git checkout -f` of `sglang-miles` in
  `/sgl-workspace/sglang` — local edits there are discarded (that is Docker's behaviour too).
- **Rate limits**: wheels come from `api.github.com` unauthenticated (60 req/h/IP). If the
  fetch step returns 403 wait, or pre-drop the `.whl` files into `/tmp/wheels/`.

## Keeping it honest

- Never "fix" a broken step by editing `install.sh` — it is regenerated every run. Fix the
  environment, or change the Dockerfile upstream (then the fix reaches the image too).
- `install.meta.json` records the Dockerfile sha256, variant and resolved args of the last
  generation; quote it when reporting what was installed.
- Translator changes must keep `tests/test_dockerfile2sh.py` green
  (`python3 -m unittest` from that directory); it executes a generated script end to end.
