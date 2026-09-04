#!/usr/bin/env python3
"""Translate a single-stage Dockerfile into an idempotent, resumable bash script.

Stdlib only, so it runs on any box that has python3. The output script replays the
Dockerfile's RUN / COPY / ENV / ARG / WORKDIR / SHELL instructions against the current
machine, which is assumed to already look like the FROM image.

Why translate instead of hand-writing the install script: the Dockerfile is the
single source of truth for what an image contains, and it changes weekly. Anything
hand-derived from it rots. This translator is generic over the Dockerfile subset the
recipe uses, so a new RUN line, a new ARG, or a reordered step needs no change here.

Semantics kept faithful to Docker where it matters for correctness:
  * Line continuations are joined exactly like Docker (backslash+newline removed,
    comment lines inside a continuation dropped), so inline python one-liners that
    span lines still parse.
  * ARG values come from --build-arg > variant build_args > Dockerfile default. They
    are baked into the script, never inherited from the host environment (Docker
    does not inherit build args from the host either; the target box may carry an
    unrelated variable of the same name).
  * RUN bodies are executed verbatim through the Dockerfile's SHELL (default
    /bin/sh -c) inside the current WORKDIR, with ARG/ENV exported, so ${VAR}
    expansion happens at run time exactly as in the build.
  * RUN --mount=type=bind is emulated by copying the context file into place for
    the step; --mount=type=cache is dropped.
  * COPY resolves globs against the build context; an empty wildcard match is a
    no-op, matching BuildKit.

Each RUN step is content-hashed. A completed step leaves a marker, so re-running
after the Dockerfile changed executes only new or edited steps.

Usage:
    dockerfile2sh.py --context <repo> [--dockerfile docker/Dockerfile]
                     [--variant NAME|auto] [--build-py docker/build.py]
                     [--build-arg K=V]... [--name miles] -o install.sh
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

BUILDKIT_ARGS = {
    "TARGETPLATFORM",
    "TARGETOS",
    "TARGETARCH",
    "TARGETVARIANT",
    "BUILDPLATFORM",
    "BUILDOS",
    "BUILDARCH",
    "BUILDVARIANT",
}
# Instructions that only shape the image manifest; nothing to replay on a live box.
IGNORED = {"LABEL", "EXPOSE", "VOLUME", "STOPSIGNAL", "HEALTHCHECK", "MAINTAINER", "CMD", "ENTRYPOINT"}


class TranslateError(Exception):
    pass


# ----------------------------------------------------------------------------- parsing


def parse_dockerfile(text: str) -> list[tuple[str, str, int]]:
    """Return [(INSTRUCTION, argument_string, first_lineno)]."""
    out: list[tuple[str, str, int]] = []
    buf: str | None = None
    start = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        stripped = line.strip()
        if buf is not None:
            # Docker drops comment lines and blank lines that sit inside a continuation.
            if not stripped or stripped.startswith("#"):
                continue
            if line.endswith("\\"):
                buf += line[:-1]
                continue
            buf += line
            out.append(_split_instruction(buf, start))
            buf = None
            continue
        if not stripped or stripped.startswith("#"):
            # Parser directives (# syntax=, # escape=) are only honoured for the default escape.
            m = re.match(r"#\s*escape\s*=\s*(\S)", stripped, re.I)
            if m and m.group(1) != "\\":
                raise TranslateError(f"line {lineno}: only the default '\\' escape character is supported")
            continue
        if line.endswith("\\"):
            buf = line[:-1]
            start = lineno
            continue
        out.append(_split_instruction(line, lineno))
    if buf is not None:
        out.append(_split_instruction(buf, start))
    return out


def _split_instruction(line: str, lineno: int) -> tuple[str, str, int]:
    parts = line.strip().split(None, 1)
    return parts[0].upper(), (parts[1] if len(parts) > 1 else ""), lineno


def _strip_quotes(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def _parse_kv_list(rest: str, legacy_space_form: bool) -> list[tuple[str, str]]:
    """ENV/ARG/LABEL argument parsing: `K=V K2="v 2"` or the legacy `K value with spaces`."""
    if legacy_space_form and "=" not in rest.split(None, 1)[0]:
        k, _, v = rest.partition(" ")
        return [(k.strip(), _strip_quotes(v.strip()))]
    pairs = []
    for tok in shlex.split(rest, posix=True):
        k, eq, v = tok.partition("=")
        pairs.append((k, v if eq else None))
    return pairs


def _leading_flags(rest: str) -> tuple[list[str], str]:
    """Pull `--flag[=value]` tokens off the front of RUN/COPY/ADD arguments."""
    flags = []
    while True:
        m = re.match(r"\s*(--[A-Za-z][\w-]*(?:=(?:\"[^\"]*\"|'[^']*'|\S+))?)\s*", rest)
        if not m:
            return flags, rest.strip()
        flags.append(m.group(1))
        rest = rest[m.end() :]


def _exec_or_shell(rest: str) -> str:
    """RUN accepts JSON exec form; convert it to a shell line."""
    if rest.startswith("["):
        try:
            argv = json.loads(rest)
            if isinstance(argv, list):
                return shlex.join(str(a) for a in argv)
        except json.JSONDecodeError:
            pass
    return rest


# ----------------------------------------------------------------------------- variants


def load_variants(build_py: Path) -> dict:
    """Read the VARIANTS table out of build.py without importing it (it pulls in typer)."""
    tree = ast.parse(build_py.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise TranslateError(f"{build_py}: no top-level VARIANTS = {{...}} literal")


def detect_host() -> dict:
    """What the target box looks like: arch (docker naming) and CUDA major, if any."""
    machine = platform.machine()
    arch = {"x86_64": "amd64", "AMD64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    cuda_major = None
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
        if out:
            cuda_major = out.split(".")[0]
    except Exception:
        pass
    if cuda_major is None:
        try:
            out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=30).stdout
            m = re.search(r"release (\d+)\.", out)
            if m:
                cuda_major = m.group(1)
        except Exception:
            pass
    return {"machine": machine, "arch": arch, "cuda_major": cuda_major}


def pick_variant(variants: dict, host: dict, dockerfile_defaults: dict, default_dockerfile: str) -> str:
    """Choose the single-platform variant that matches this box.

    Match rule: the variant's platforms are exactly [linux/<our arch>], its Dockerfile is
    the one we're translating, and its ENABLE_CUDA_13 (or the Dockerfile default when the
    variant doesn't set it) agrees with the host's CUDA major.
    """
    want_platform = f"linux/{host['arch']}"
    want_cu13 = "1" if host["cuda_major"] == "13" else "0"
    matches = []
    for name, cfg in variants.items():
        if cfg.get("dockerfile", default_dockerfile) != default_dockerfile:
            continue
        if cfg.get("platforms") != [want_platform]:
            continue
        cu13 = str(cfg.get("build_args", {}).get("ENABLE_CUDA_13", dockerfile_defaults.get("ENABLE_CUDA_13", "1")))
        if cu13 == want_cu13:
            matches.append(name)
    if len(matches) != 1:
        raise TranslateError(
            f"cannot auto-pick a variant for host {host} (candidates: {matches or 'none'}; "
            f"all: {list(variants)}). Pass --variant explicitly."
        )
    return matches[0]


# ----------------------------------------------------------------------------- emit

RUNTIME_PRELUDE = r"""
# ---------------------------------------------------------------- runtime -----
DF_STATE_DIR="${DF_STATE_DIR:-$HOME/.cache/dockerfile2sh/__NAME__}"
DF_ENV_FILE="${DF_ENV_FILE:-$DF_STATE_DIR/env.sh}"
DF_FROM_STEP="${DF_FROM_STEP:-}"
DF_ONLY_STEP="${DF_ONLY_STEP:-}"
DF_FORCE="${DF_FORCE:-}"
DF_DRY_RUN="${DF_DRY_RUN:-}"
DF_WORKDIR="/"
DF_SHELL=(/bin/sh -c)
DF_STEP_TOTAL=__STEP_TOTAL__
mkdir -p "$DF_STATE_DIR" "$(dirname "$DF_ENV_FILE")"
touch "$DF_ENV_FILE"

df_log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
df_die() { printf '\033[1;31m[%s] ERROR:\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# ENV: export now and persist the *unexpanded* expression so later shells compose it
# the same way (PATH="/x:${PATH}" must not freeze today's PATH).
df_env() {
  local k="$1" v="$2"
  eval "export $k=\"$v\""
  local line="export $k=\"$v\""
  grep -qxF -- "$line" "$DF_ENV_FILE" 2>/dev/null || printf '%s\n' "$line" >> "$DF_ENV_FILE"
}

df_workdir() {
  case "$1" in /*) DF_WORKDIR="$1" ;; *) DF_WORKDIR="${DF_WORKDIR%/}/$1" ;; esac
  [ -n "$DF_DRY_RUN" ] || mkdir -p "$DF_WORKDIR"
}

# COPY src... dest — globs resolve against the build context, BuildKit-style: a
# directory source copies its contents; an empty wildcard match is a no-op.
df_copy() {
  local dest="${*: -1}"
  local -a srcs=("${@:1:$#-1}")
  case "$dest" in /*) ;; *) dest="${DF_WORKDIR%/}/$dest" ;; esac
  if [ -n "$DF_DRY_RUN" ]; then df_log "COPY ${srcs[*]} -> $dest"; return 0; fi
  local matched=0 s f
  local -a found
  for s in "${srcs[@]}"; do
    shopt -s nullglob dotglob
    found=("$DF_CTX"/$s)
    shopt -u nullglob dotglob
    for f in "${found[@]+"${found[@]}"}"; do
      matched=1
      if [ -d "$f" ]; then
        mkdir -p "$dest" && cp -a "$f"/. "$dest"/
      elif [[ "$dest" == */ ]] || [ -d "$dest" ] || [ "${#srcs[@]}" -gt 1 ]; then
        mkdir -p "$dest" && cp -a "$f" "$dest"/
      else
        mkdir -p "$(dirname "$dest")" && cp -a "$f" "$dest"
      fi
    done
  done
  [ "$matched" = 1 ] || df_log "COPY ${srcs[*]}: nothing matched in $DF_CTX (empty wildcard, no-op)"
}

# RUN. Reads the command body from stdin (a quoted heredoc, so nothing expands until
# the Dockerfile's shell runs it). Options: --bind src:target (emulated bind mount),
# --at "L<line>" (source location for logs).
df_run() {
  local n="$1"; shift
  local -a binds=()
  local where=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --bind) binds+=("$2"); shift 2 ;;
      --at) where="$2"; shift 2 ;;
      *) df_die "df_run: unknown option $1" ;;
    esac
  done
  local cmd; cmd="$(cat)"
  local key; key="$(printf '%s\n%s\n%s' "$DF_WORKDIR" "${DF_SHELL[*]}" "$cmd" | sha256sum | cut -c1-16)"
  local marker="$DF_STATE_DIR/step-$(printf '%03d' "$n")-$key.done"
  local head; head="$(printf '%s' "$cmd" | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-110 || true)"

  if [ -n "$DF_ONLY_STEP" ] && [ "$DF_ONLY_STEP" != "$n" ]; then return 0; fi
  if [ -n "$DF_FROM_STEP" ] && [ "$n" -lt "$DF_FROM_STEP" ]; then
    df_log "step $n/$DF_STEP_TOTAL ($where): skipped (--from $DF_FROM_STEP)"; return 0
  fi
  if [ -z "$DF_FORCE" ] && [ -e "$marker" ]; then
    df_log "step $n/$DF_STEP_TOTAL ($where): done earlier, skip  [$head]"; return 0
  fi
  df_log "step $n/$DF_STEP_TOTAL ($where) in $DF_WORKDIR: $head"
  if [ -n "$DF_DRY_RUN" ]; then printf '%s\n' "$cmd" | sed 's/^/      | /' >&2; return 0; fi

  local b src tgt
  local -a created=()
  for b in "${binds[@]+"${binds[@]}"}"; do  # empty-array-safe under set -u on bash 3.x
    src="${b%%:*}"; tgt="${b#*:}"
    [ -e "$tgt" ] || created+=("$tgt")
    mkdir -p "$(dirname "$tgt")" && cp -a "$DF_CTX/$src" "$tgt"
  done
  local t0=$SECONDS rc=0
  if ( cd "$DF_WORKDIR" && "${DF_SHELL[@]}" "$cmd" ); then rc=0; else rc=$?; fi
  for tgt in "${created[@]+"${created[@]}"}"; do rm -rf "$tgt"; done
  if [ "$rc" -ne 0 ]; then
    df_die "step $n/$DF_STEP_TOTAL ($where) failed with rc=$rc after $((SECONDS - t0))s. Fix the cause and re-run; completed steps are skipped, this one resumes."
  fi
  : > "$marker"
  df_log "step $n/$DF_STEP_TOTAL ok in $((SECONDS - t0))s"
}
# --------------------------------------------------------------------------------
"""


def sh_dq(s: str) -> str:
    """Escape for use inside a double-quoted bash string while keeping ${VAR} live."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")


def translate(
    dockerfile: Path,
    context: Path,
    build_args: dict[str, str],
    name: str,
    source_label: str,
) -> tuple[str, dict]:
    text = dockerfile.read_text()
    instrs = parse_dockerfile(text)
    body: list[str] = []
    step = 0
    from_count = 0
    arg_values: dict[str, str | None] = {}
    shell_set = False
    used_build_args: set[str] = set()
    base_image = None
    notes: list[str] = []

    def emit(s: str = "") -> None:
        body.append(s)

    for ins, rest, lineno in instrs:
        at = f"L{lineno}"
        if ins == "FROM":
            from_count += 1
            if from_count > 1:
                raise TranslateError(f"{at}: multi-stage Dockerfiles are not supported (second FROM)")
            toks = shlex.split(rest)
            flags = [t for t in toks if t.startswith("--")]
            image = [t for t in toks if not t.startswith("--")][0]
            base_image = image
            emit(f"# FROM {rest}")
            emit(f'DF_BASE_IMAGE="{sh_dq(image)}"')
            emit('df_log "recipe base image: $DF_BASE_IMAGE — this box is assumed to be equivalent to it"')
            if flags:
                notes.append(f"{at}: FROM flags ignored: {' '.join(flags)}")
            emit()
        elif ins == "ARG":
            for k, default in _parse_kv_list(rest, legacy_space_form=False):
                if k in build_args:
                    used_build_args.add(k)
                    arg_values[k] = build_args[k]
                    emit(f"export {k}={shlex.quote(build_args[k])}  # ARG {at} (build-arg)")
                elif k in BUILDKIT_ARGS and default is None:
                    detect = {
                        "TARGETARCH": "$(dpkg --print-architecture 2>/dev/null || uname -m | sed -e s/x86_64/amd64/ -e s/aarch64/arm64/)",
                        "TARGETOS": "linux",
                        "TARGETPLATFORM": "linux/$(dpkg --print-architecture 2>/dev/null || uname -m | sed -e s/x86_64/amd64/ -e s/aarch64/arm64/)",
                    }.get(k)
                    if detect is None:
                        raise TranslateError(f"{at}: BuildKit arg {k} has no host equivalent here")
                    emit(f'export {k}="{detect}"  # ARG {at} (BuildKit builtin, detected on host)')
                    arg_values[k] = None
                else:
                    val = "" if default is None else _strip_quotes(default)
                    arg_values[k] = val
                    emit(f'export {k}="{sh_dq(val)}"  # ARG {at} (Dockerfile default)')
            emit()
        elif ins == "ENV":
            for k, v in _parse_kv_list(rest, legacy_space_form=True):
                # Single-quoted so ${VAR} reaches df_env unexpanded; it expands at eval time
                # and is persisted as the expression, not today's value.
                emit(f"df_env {k} {shlex.quote(v or '')}  # ENV {at}")
            emit()
        elif ins == "WORKDIR":
            emit(f'df_workdir "{sh_dq(_strip_quotes(rest.strip()))}"  # WORKDIR {at}')
            emit()
        elif ins == "SHELL":
            argv = json.loads(rest)
            emit(f"DF_SHELL=({' '.join(shlex.quote(a) for a in argv)})  # SHELL {at}")
            shell_set = True
            emit()
        elif ins == "USER":
            notes.append(f"{at}: USER {rest} ignored; steps run as the invoking user")
            emit(f"# USER {rest}  (ignored)")
        elif ins in ("COPY", "ADD"):
            flags, args_s = _leading_flags(rest)
            for f in flags:
                if f.startswith("--from"):
                    raise TranslateError(f"{at}: COPY --from (multi-stage) is not supported")
                if not (f.startswith("--chown") or f.startswith("--chmod") or f == "--link"):
                    notes.append(f"{at}: {ins} flag {f} ignored")
            if args_s.startswith("["):
                argv = json.loads(args_s)
            else:
                argv = shlex.split(args_s)
            if len(argv) < 2:
                raise TranslateError(f"{at}: {ins} needs at least one source and a destination")
            if ins == "ADD" and any(re.match(r"https?://|git@", a) for a in argv[:-1]):
                raise TranslateError(f"{at}: ADD from URL is not supported; use RUN curl")
            emit(f"df_copy {' '.join(shlex.quote(a) for a in argv)}  # {ins} {at}")
            chmod = [f for f in flags if f.startswith("--chmod=")]
            if chmod:
                emit(f"chmod -R {shlex.quote(chmod[-1].split('=', 1)[1])} {shlex.quote(argv[-1])}")
            emit()
        elif ins == "RUN":
            flags, cmd = _leading_flags(rest)
            cmd = _exec_or_shell(cmd)
            binds = []
            for f in flags:
                if f.startswith("--mount="):
                    kv = dict(p.split("=", 1) if "=" in p else (p, "") for p in f[len("--mount=") :].split(","))
                    mtype = kv.get("type", "bind")
                    if mtype == "bind":
                        if kv.get("from"):
                            raise TranslateError(f"{at}: --mount from= another stage is not supported")
                        src = kv.get("source") or kv.get("src") or "."
                        tgt = kv.get("target") or kv.get("dst") or kv.get("destination")
                        if not tgt:
                            raise TranslateError(f"{at}: bind mount without target")
                        binds.append(f"{src}:{tgt}")
                    elif mtype in ("cache", "tmpfs"):
                        pass  # host filesystem stands in for both
                    elif mtype in ("secret", "ssh"):
                        notes.append(f"{at}: --mount type={mtype} dropped; provide it via the environment")
                    else:
                        raise TranslateError(f"{at}: unsupported mount type {mtype}")
                elif f.startswith("--network") or f.startswith("--security"):
                    pass
                else:
                    notes.append(f"{at}: RUN flag {f} ignored")
            step += 1
            opts = "".join(f" --bind {shlex.quote(b)}" for b in binds)
            emit(f"df_run {step} --at {at}{opts} <<'DF_EOF'")
            emit(cmd)
            emit("DF_EOF")
            emit()
        elif ins in IGNORED:
            emit(f"# {ins} {rest}  (image metadata, nothing to do)")
        elif ins == "ONBUILD":
            raise TranslateError(f"{at}: ONBUILD is not supported")
        else:
            raise TranslateError(f"{at}: unknown instruction {ins}")

    unused = set(build_args) - used_build_args
    if unused:
        notes.append(f"build-args not declared by the Dockerfile (ignored, like docker build): {sorted(unused)}")

    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    header = [
        "#!/usr/bin/env bash",
        "# Generated by dockerfile2sh.py — DO NOT EDIT; regenerate from the Dockerfile instead.",
        f"#   source : {source_label}",
        f"#   sha256 : {digest}",
        f"#   when   : {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"#   steps  : {step} RUN instructions",
        "#",
        "# Replays the Dockerfile on the current machine. Env knobs:",
        "#   DF_CTX=<dir>        build context (the repo checkout COPY/bind sources come from)",
        "#   DF_STATE_DIR=<dir>  step markers + env.sh (default ~/.cache/dockerfile2sh/<name>)",
        "#   DF_FROM_STEP=N      start at step N (earlier ones skipped)   DF_ONLY_STEP=N  run only step N",
        "#   DF_FORCE=1          ignore done-markers                     DF_DRY_RUN=1    print, don't run",
        "#   DF_LIST=1           list steps and exit",
        "set -euo pipefail",
        f'DF_CTX="${{DF_CTX:-{sh_dq(str(context))}}}"',
        '[ -d "$DF_CTX" ] || { echo "DF_CTX=$DF_CTX is not a directory (need the repo checkout as build context)" >&2; exit 2; }',
        'if [ -n "${DF_LIST:-}" ]; then grep -nE "^df_run [0-9]+ --at" "$0" | sed -E "s/^([0-9]+):df_run ([0-9]+) --at (L[0-9]+).*/step \\2  (Dockerfile \\3)/"; exit 0; fi',
    ]
    for n in notes:
        header.append(f"# note: {n}")
    prelude = RUNTIME_PRELUDE.replace("__NAME__", name).replace("__STEP_TOTAL__", str(step))
    footer = [
        "",
        'df_log "all $DF_STEP_TOTAL steps complete."',
        'df_log "persisted ENV lines live in $DF_ENV_FILE — source it in new shells (the wrapper wires it into ~/.bashrc)."',
    ]
    script = "\n".join(header) + "\n" + prelude + "\n" + "\n".join(body) + "\n".join(footer) + "\n"
    meta = {
        "base_image": base_image,
        "steps": step,
        "args": arg_values,
        "dockerfile_sha256": digest,
        "notes": notes,
        "shell_overridden": shell_set,
    }
    return script, meta


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True, type=Path, help="build context directory (repo checkout)")
    ap.add_argument(
        "--dockerfile",
        default=None,
        help="Dockerfile path relative to context (default: variant's or docker/Dockerfile)",
    )
    ap.add_argument(
        "--variant",
        default="auto",
        help="build.py VARIANTS key, 'auto' to detect, or 'none' for Dockerfile defaults only",
    )
    ap.add_argument("--build-py", default="docker/build.py", help="where VARIANTS live, relative to context")
    ap.add_argument(
        "--build-arg",
        action="append",
        default=[],
        metavar="K=V",
        help="override an ARG (repeatable, wins over variant)",
    )
    ap.add_argument("--name", default=None, help="state-dir name (default: context dir basename)")
    ap.add_argument("--source-label", default=None, help="human label for the header (e.g. repo@ref)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output script (default: stdout)")
    ap.add_argument("--meta", type=Path, default=None, help="also write resolved metadata as JSON here")
    a = ap.parse_args(argv)

    context = a.context.resolve()
    cli_args: dict[str, str] = {}
    for spec in a.build_arg:
        if "=" not in spec:
            ap.error(f"--build-arg expects K=V, got {spec!r}")
        k, v = spec.split("=", 1)
        cli_args[k] = v

    default_dockerfile = "docker/Dockerfile"
    variants: dict = {}
    build_py = context / a.build_py
    if a.variant != "none":
        if build_py.exists():
            variants = load_variants(build_py)
        elif a.variant != "auto":
            ap.error(f"--variant {a.variant} requested but {build_py} does not exist")
        else:
            print(f"warning: {build_py} missing; using Dockerfile defaults only", file=sys.stderr)

    # Dockerfile defaults are needed to auto-pick a variant (ENABLE_CUDA_13 may be unset there).
    probe_df = context / (a.dockerfile or default_dockerfile)
    df_defaults: dict[str, str] = {}
    if probe_df.exists():
        for ins, rest, _ in parse_dockerfile(probe_df.read_text()):
            if ins == "ARG":
                for k, v in _parse_kv_list(rest, legacy_space_form=False):
                    if v is not None:
                        df_defaults[k] = _strip_quotes(v)

    variant_name = a.variant
    variant_args: dict[str, str] = {}
    if variants:
        if variant_name == "auto":
            variant_name = pick_variant(variants, detect_host(), df_defaults, a.dockerfile or default_dockerfile)
            print(f"variant: {variant_name} (auto-detected)", file=sys.stderr)
        if variant_name not in variants:
            ap.error(f"unknown variant {variant_name!r}; known: {list(variants)}")
        cfg = variants[variant_name]
        variant_args = {k: str(v) for k, v in cfg.get("build_args", {}).items()}
        if a.dockerfile is None:
            a.dockerfile = cfg.get("dockerfile", default_dockerfile)
        plats = cfg.get("platforms") or []
        if len(plats) == 1 and "TARGETARCH" not in cli_args:
            variant_args.setdefault("TARGETARCH", plats[0].split("/")[1])
    elif variant_name == "auto":
        variant_name = "none"

    dockerfile = context / (a.dockerfile or default_dockerfile)
    if not dockerfile.exists():
        ap.error(f"Dockerfile not found: {dockerfile}")

    build_args = {**variant_args, **cli_args}
    name = a.name or context.name
    label = a.source_label or f"{dockerfile.relative_to(context)} in {context}"
    try:
        script, meta = translate(dockerfile, context, build_args, name, f"{label} (variant={variant_name})")
    except TranslateError as e:
        print(f"dockerfile2sh: {e}", file=sys.stderr)
        return 2
    meta["variant"] = variant_name
    meta["build_args"] = build_args
    meta["dockerfile"] = str(dockerfile.relative_to(context))

    if a.output:
        a.output.write_text(script)
        a.output.chmod(0o755)
        print(f"wrote {a.output} ({meta['steps']} steps, base {meta['base_image']})", file=sys.stderr)
    else:
        sys.stdout.write(script)
    if a.meta:
        a.meta.write_text(json.dumps(meta, indent=2) + "\n")
    for n in meta["notes"]:
        print(f"note: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
