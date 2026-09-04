#!/usr/bin/env bash
# Install the radixark/miles stack onto the current machine by replaying its Dockerfile.
#
# Nothing about the recipe is hard-coded here: every run clones miles at --ref, reads
# docker/Dockerfile + docker/build.py from that checkout, translates them with
# dockerfile2sh.py, and executes the result. When upstream edits the Dockerfile, the next
# run picks the change up; completed steps whose text is unchanged are skipped.
#
# Usage: miles-install.sh [options]
#   --ref <git ref>       miles ref to install (branch / tag / sha; default main)
#   --repo <url>          miles repo (default https://github.com/radixark/miles.git)
#   --src <dir>           use an existing miles checkout as the recipe/build context instead
#                         of cloning (e.g. the repo this script lives in). MILES_COMMIT then
#                         defaults to that checkout's HEAD, which must be reachable on GitHub —
#                         pass --build-arg MILES_COMMIT=main for an unpushed tree.
#   --variant <name>      docker/build.py variant; default auto (by host CUDA major + arch)
#   --build-arg K=V       override a Dockerfile ARG (repeatable; wins over the variant)
#   --workdir <dir>       clone + state dir (default /root/.miles-install or ~/.miles-install)
#   --from N | --only N   resume at / run only RUN step N
#   --force               re-run steps even if marked done
#   --dry-run             print the translated steps, run nothing
#   --generate-only       write install.sh and stop
#   --list                list steps with Dockerfile line numbers and exit
#   --no-bashrc           don't wire the persisted ENV file into ~/.bashrc
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="https://github.com/radixark/miles.git"
REF="main"
SRC_OVERRIDE=""
VARIANT="auto"
WORKDIR=""
BUILD_ARGS=()
MODE="run"
NO_BASHRC=""
export DF_FROM_STEP="${DF_FROM_STEP:-}" DF_ONLY_STEP="${DF_ONLY_STEP:-}" DF_FORCE="${DF_FORCE:-}" DF_DRY_RUN="${DF_DRY_RUN:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --src) SRC_OVERRIDE="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --build-arg) BUILD_ARGS+=("$2"); shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --from) DF_FROM_STEP="$2"; shift 2 ;;
    --only) DF_ONLY_STEP="$2"; shift 2 ;;
    --force) DF_FORCE=1; shift ;;
    --dry-run) DF_DRY_RUN=1; shift ;;
    --generate-only) MODE="generate"; shift ;;
    --list) MODE="list"; shift ;;
    --no-bashrc) NO_BASHRC=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;32m[miles-install]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[miles-install] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"
command -v python3 >/dev/null || die "python3 is required"
[ -f "$HERE/dockerfile2sh.py" ] || die "dockerfile2sh.py must sit next to this script ($HERE)"
[ "$(id -u)" = 0 ] || log "warning: not root — the recipe runs apt/pip system-wide and expects root"

if [ -z "$WORKDIR" ]; then
  if [ -w /root ] 2>/dev/null; then WORKDIR=/root/.miles-install; else WORKDIR="$HOME/.miles-install"; fi
fi
mkdir -p "$WORKDIR"
SRC="$WORKDIR/miles-src"
export DF_STATE_DIR="$WORKDIR/state"
export DF_CTX="$SRC"

# ---- 1. fetch the recipe (the miles checkout doubles as the docker build context) ----
if [ -n "$SRC_OVERRIDE" ]; then
  SRC="$(cd "$SRC_OVERRIDE" && pwd)"
  [ -f "$SRC/docker/Dockerfile" ] || die "--src $SRC has no docker/Dockerfile"
  export DF_CTX="$SRC"
  REF="$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || echo local)"
  [ -z "$(git -C "$SRC" status --porcelain -- docker requirements.txt 2>/dev/null)" ] || log "warning: $SRC has uncommitted changes under docker/ or requirements.txt; replaying them as-is"
else
  if [ ! -d "$SRC/.git" ]; then
    log "cloning $REPO -> $SRC"
    git clone --quiet --filter=blob:none "$REPO" "$SRC"
  fi
  git -C "$SRC" remote set-url origin "$REPO"
  log "fetching $REF"
  if git -C "$SRC" fetch --quiet origin "$REF"; then
    git -C "$SRC" checkout --quiet --force FETCH_HEAD
  else
    # A sha the server won't serve by name: fetch everything, then resolve it locally.
    git -C "$SRC" fetch --quiet origin
    git -C "$SRC" checkout --quiet --force "$REF"
  fi
fi
SHA="$(git -C "$SRC" rev-parse --short=9 HEAD)"
log "recipe at $REF = $SHA ($(git -C "$SRC" log -1 --format='%cs %s' | cut -c1-80)) in $SRC"

# ---- 2. translate Dockerfile -> bash for this box ----
GEN_ARGS=(--context "$SRC" --variant "$VARIANT" --name miles --source-label "$REPO@$REF ($SHA) docker/Dockerfile"
          -o "$WORKDIR/install.sh" --meta "$WORKDIR/install.meta.json")
# Install the same miles revision the recipe came from unless the caller pins MILES_COMMIT.
pinned_miles=""
for kv in "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"; do
  GEN_ARGS+=(--build-arg "$kv")
  case "$kv" in MILES_COMMIT=*) pinned_miles=1 ;; esac
done
[ -n "$pinned_miles" ] || GEN_ARGS+=(--build-arg "MILES_COMMIT=$SHA")
python3 "$HERE/dockerfile2sh.py" "${GEN_ARGS[@]}"

log "host: $(uname -m), python $(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:3])))'), $(python3 -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda)' 2>/dev/null || echo 'torch: not importable')"
[ -z "${SGLANG_IMAGE_TAG:-}" ] || log "host image tag env: SGLANG_IMAGE_TAG=$SGLANG_IMAGE_TAG (compare with the recipe base image printed below)"

case "$MODE" in
  list) DF_LIST=1 bash "$WORKDIR/install.sh"; exit 0 ;;
  generate) log "generated $WORKDIR/install.sh (not run)"; exit 0 ;;
esac

# ---- 3. replay ----
log "running $WORKDIR/install.sh  (state: $DF_STATE_DIR; log tee'd to $WORKDIR/install.log)"
set +e
bash "$WORKDIR/install.sh" 2>&1 | tee -a "$WORKDIR/install.log"
rc=${PIPESTATUS[0]}
set -e
[ "$rc" -eq 0 ] || die "install.sh exited $rc — see $WORKDIR/install.log; re-run the same command to resume"

# ---- 4. make persisted ENV visible to future shells ----
ENV_FILE="$DF_STATE_DIR/env.sh"
if [ -z "$NO_BASHRC" ] && [ -s "$ENV_FILE" ] && [ -z "${DF_DRY_RUN:-}" ]; then
  line="[ -f \"$ENV_FILE\" ] && . \"$ENV_FILE\"  # miles-install"
  grep -qF -- "# miles-install" "$HOME/.bashrc" 2>/dev/null || printf '\n%s\n' "$line" >> "$HOME/.bashrc"
  log "ENV from the Dockerfile persisted in $ENV_FILE and sourced from ~/.bashrc"
fi

[ -n "${DF_DRY_RUN:-}" ] && exit 0
log "verifying imports"
python3 - <<'PY' || die "verification failed — the environment differs from the recipe's base image; see install.log"
import importlib, sys
mods = ["miles", "megatron", "sglang", "transformer_engine", "flash_attn", "apex"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok  {m:20s} {getattr(mod, '__file__', '')}")
    except Exception as e:  # noqa: BLE001
        bad.append(m); print(f"  FAIL {m:20s} {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
PY
log "done. miles installed at /root/miles (editable); recipe $REPO@$REF ($SHA)"
