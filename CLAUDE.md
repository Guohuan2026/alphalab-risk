# AlphaLab - instructions for Claude Code

Educational, production-style quant research platform in Python (src layout, package `alphalab`).
Owner is learning: explain what each important command does and why, briefly.

## Environment
- macOS, Python 3.12 via uv, virtual env at `.venv/` (activate: `source .venv/bin/activate`).
- One-shot setup/verification: `bash scripts/setup_mac.sh` (log: `logs/setup_mac.log`).
- Known issue: macOS can mark `.venv/lib/python3.12/site-packages/*.pth` as hidden (iCloud-synced
  Documents), so Python skips them and `import alphalab` fails. Fix: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`.

## Commands
- Tests: `python -m pytest -q` (offline); live Yahoo: `python -m pytest -m network -q`
- Lint/format: `ruff check . && ruff format --check .`
- Pipeline: `alphalab ingest --config <yaml>` then `alphalab build --config <yaml>`

## Rules (non-negotiable)
- Point-in-time only: a value stamped at t uses data available at the close of t. Forward returns are
  labels, never signal inputs. Every new signal/factor gets an `assert_point_in_time` test.
- Never edit files in `data/raw/` (immutable, hash-verified snapshots). Re-ingest instead.
- Never use vendor backward-adjusted price LEVELS in signals.
- Do not claim something ran or passed unless it was actually executed in this session; show the output.
- Run tests + lint before calling a change done. Add/adjust tests with every code change.
- Document new limitations in `docs/LIMITATIONS.md`; methodology in `docs/METHODOLOGY.md`.
- Build incrementally, one milestone at a time (see `docs/ROADMAP.md`). Ask before large refactors.
