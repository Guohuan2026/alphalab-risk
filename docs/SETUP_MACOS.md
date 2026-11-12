# Setup on macOS with VS Code

Every command below is followed by what it does and why. Run them in the macOS
**Terminal** app or VS Code's built-in terminal (`` Ctrl+` ``).

## 0. One-time tools

```bash
xcode-select --install
```
Installs Apple's command-line developer tools (compilers, `git`, `make`). Many Python
packages and `git` need them. If it says "already installed", skip.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Installs **Homebrew**, the standard macOS package manager. Afterwards, follow the
"Next steps" it prints (on Apple Silicon it asks you to add `/opt/homebrew/bin` to your
PATH). Skip if `brew --version` already works.

```bash
brew install python@3.12 git
```
Installs Python 3.12 and git. Why 3.12 and not macOS's built-in `python3`: the system
Python is old and owned by macOS; a Homebrew Python is yours to manage. The project
needs Python ≥ 3.11 (it was tested here on 3.13; 3.12 is the most widely supported
version for the scientific stack right now).

```bash
brew install --cask visual-studio-code
```
Installs VS Code (skip if you have it). Then in VS Code: `Cmd+Shift+P` → "Shell Command:
Install 'code' command in PATH", so `code .` opens a folder from the terminal.

## 1. Get the project

```bash
mkdir -p ~/Projects && cd ~/Projects
```
`mkdir -p` creates the folder (and parents) if missing, without error if it exists;
`cd` moves into it.

```bash
unzip ~/Downloads/alphalab.zip -d ~/Projects
cd alphalab
```
Extracts the archive into `~/Projects/alphalab` and enters it. All later commands assume
you are in this folder (the configs use paths relative to it).

```bash
git init && git add . && git commit -m "Milestone 1: data pipeline and returns"
```
Starts version control. `git init` creates a repository, `git add .` stages every file
(`.gitignore` keeps data and the virtual environment out), `git commit` records a
snapshot you can always return to.

## 2. Virtual environment

```bash
python3.12 -m venv .venv
```
Creates an isolated Python installation in `.venv/`. Packages you install go there, not
into the global Python, so projects can't break each other.

```bash
source .venv/bin/activate
```
Activates it for this terminal session: `python` and `pip` now point into `.venv`. Your
prompt shows `(.venv)`. You must re-run this in every new terminal (VS Code does it
automatically, see step 4). `deactivate` turns it off.

```bash
python -m pip install --upgrade pip
```
Upgrades pip inside the venv. `python -m pip` guarantees you use the pip belonging to
the active Python, avoiding the classic "installed but can't import" problem.

```bash
pip install -e ".[dev]"
```
Installs AlphaLab and its dependencies from `pyproject.toml`.
- `-e` (editable): Python imports your source files directly, so code edits take effect
  without reinstalling.
- `.[dev]`: the current folder plus the optional `dev` extras (pytest, coverage, ruff).
  Quotes stop zsh from treating the brackets as a pattern.
It also creates the `alphalab` command.

```bash
python -c "import alphalab, pandas, yfinance; print(alphalab.__version__, pandas.__version__, yfinance.__version__)"
```
Smoke test: proves the package and key libraries import, and prints versions.

## 3. Verify (in this order)

```bash
python -m pytest -q
```
Runs the 41 offline unit tests in `tests/`. `-q` = quiet output. Expected:
`41 passed, 1 deselected` (the deselected one is the live-network test).

```bash
ruff check . && ruff format --check .
```
`ruff check` finds style issues and likely bugs; `ruff format --check` confirms formatting
without changing files. Expected: `All checks passed!` and no files to reformat.

```bash
alphalab ingest --config configs/synthetic.yaml
alphalab build  --config configs/synthetic.yaml
```
Runs the whole pipeline offline on synthetic data. `ingest` saves an immutable snapshot
under `data/raw/synthetic/<id>/`; `build` validates it and writes return files under
`data/processed/synthetic/<id>/`. Expected: `validation: no issues` and a `processed:` path.

```bash
python -m pytest -m network -q
```
Live test: downloads AAPL around its 4-for-1 split (2020-08-31) and checks the split
day's return is not a fake −75%. Needs internet. If Yahoo rate-limits you, wait a few
minutes and retry; yfinance is unofficial and occasionally breaks.

```bash
alphalab ingest --config configs/yahoo_us_core.yaml
alphalab build  --config configs/yahoo_us_core.yaml
```
Real data: 7 tickers, 2015-01-01 to 2026-10-30. Expect some **warnings** (not errors):
`vendor_adj_mismatch` on a few dividend days is normal (see METHODOLOGY §3). Inspect
`data/processed/yahoo/<id>/validation_report.csv`.

```bash
alphalab snapshots --config configs/yahoo_us_core.yaml
```
Lists the snapshot IDs you have. `build --snapshot <id>` rebuilds from a specific one.

## 4. VS Code

```bash
code .
```
Opens the folder. Install the recommended extensions when prompted (Python, Pylance,
Ruff — listed in `.vscode/extensions.json`). `.vscode/settings.json` already points VS
Code at `.venv`, enables pytest in the Testing panel (flask icon), and formats on save.
If the interpreter isn't picked up: `Cmd+Shift+P` → "Python: Select Interpreter" →
`./.venv/bin/python`.

## 5. Inspect results in Python

```python
import glob

import pandas as pd

out = sorted(glob.glob("data/processed/yahoo/*"))[-1]
m = pd.read_parquet(f"{out}/returns_monthly.parquet")
print(m.tail())
print(m.describe().T[["mean", "std"]] * [12, 12**0.5])  # rough annualised mean / vol
```

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `command not found: alphalab` | venv not active → `source .venv/bin/activate` |
| `ModuleNotFoundError: alphalab` | ran `pip install` outside the venv, or forgot `-e .` |
| `ModuleNotFoundError: alphalab` *inside* the venv, in `~/Documents` | iCloud marked the venv's `.pth` files hidden. Keep the venv outside iCloud: run `bash scripts/setup_mac.sh` (creates `~/.venvs/alphalab` and links `.venv` to it) |
| `zsh: no matches found: .[dev]` | missing quotes → `pip install -e ".[dev]"` |
| `end must be a fixed date in the past` | intended: set `end` to a past date in the YAML |
| Yahoo returns nothing / 429 | rate-limited or API change → wait; `pip install -U yfinance` |
| `content hash mismatch` | a raw file was edited; re-ingest instead of editing raw data |
