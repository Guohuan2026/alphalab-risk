#!/usr/bin/env bash
# One-shot setup + verification for AlphaLab on macOS.
# Usage (from anywhere):  bash ~/Documents/alphalab/scripts/setup_mac.sh
# Everything printed is also saved to logs/setup_mac.log so it can be reviewed later.
# Safe to re-run: each step skips work that is already done.

set -uo pipefail                       # -u: error on undefined vars; pipefail: a failing command in a pipe counts as failure
cd "$(dirname "$0")/.."                # run from the repo root no matter where you call it from
mkdir -p logs
LOG="logs/setup_mac.log"
exec > >(tee "$LOG") 2>&1              # copy all output (and errors) to the log file

step() { echo; echo "===== $1 ====="; }
fail() { echo "FAILED at: $1"; echo "Log saved to $(pwd)/$LOG"; exit 1; }

# The real environment lives OUTSIDE iCloud (default ~/.venvs/alphalab); .venv is a symlink.
# Why: with iCloud "Desktop & Documents" sync, iCloud marks every file in a .venv under
# ~/Documents as "hidden" within a second. Python 3.12 deliberately skips hidden .pth files,
# so the file pointing to src/ is ignored and `import alphalab` fails.
VENV_DIR="${ALPHALAB_VENV:-$HOME/.venvs/alphalab}"

# unhide: safety net for an old in-project .venv; shows flags (for diagnosis) and clears them.
unhide() {
  local sp=".venv/lib/python3.12/site-packages"
  [ -d "$sp" ] || return 0
  echo "[pth flags] $(ls -lO "$sp"/*.pth 2>/dev/null | awk '{print $5, $NF}' | tr '\n' ';')"
  chflags nohidden "$sp"/*.pth 2>/dev/null || true
}

step "0. System info"
date; sw_vers 2>/dev/null; uname -m

step "1. Install uv (Python installer) if missing"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv install"
fi
export PATH="$HOME/.local/bin:$PATH"   # make uv visible in this script
uv --version || fail "uv not found after install"

step "2. Create virtual environment with Python 3.12 (outside iCloud)"
if [ -d .venv ] && [ ! -L .venv ]; then        # a real folder from an older setup
  backup=".venv.old-$(date +%Y%m%d%H%M%S)"
  echo "Moving old in-project .venv to $backup (safe to delete once this run passes)"
  mv .venv "$backup" || fail "move old .venv"
fi
[ -x "$VENV_DIR/bin/python" ] || uv venv --python 3.12 "$VENV_DIR" || fail "venv"
[ -L .venv ] || ln -s "$VENV_DIR" .venv || fail "symlink .venv"   # -L: is it a symlink?
echo ".venv -> $(readlink .venv)"
source .venv/bin/activate
python --version

step "3. Install AlphaLab + dev tools (exact versions from requirements-lock.txt)"
# -c = constraints: install the project, but force every dependency to the locked version,
# so results reproduce even after libraries change defaults (quantile-forest 1.5 did).
uv pip install -e ".[dev]" -c requirements-lock.txt || fail "install"

step "4. Offline unit tests"
unhide
python -m pytest -q || fail "unit tests"

step "5. Lint"
ruff check . && ruff format --check . || echo "(lint issues - not fatal)"

step "6. Live Yahoo test"
unhide
python -m pytest -m network -q || fail "network test"

step "7. Yahoo ingest (download + snapshot)"
unhide
alphalab ingest --config configs/yahoo_us_core.yaml || fail "ingest"

step "8. Build returns + validation"
unhide
alphalab build --config configs/yahoo_us_core.yaml || fail "build"

step "DONE"
echo "Log saved to $(pwd)/$LOG"
