# AlphaLab - instructions for Claude Code

Educational, production-style quant research platform in Python (src layout, package `alphalab`).
Owner is learning: explain what each important command does and why, briefly.

## Environment
- macOS, Python 3.12 via uv. The virtual env lives at `~/.venvs/alphalab`, outside iCloud;
  `.venv` is a symlink to it (activate: `source .venv/bin/activate`).
- One-shot setup/verification: `bash scripts/setup_mac.sh` (log: `logs/setup_mac.log`).
- Why outside iCloud: `~/Documents` is iCloud-synced; iCloud marks every file in an in-project
  venv as hidden within a second, and Python 3.12 skips hidden `.pth` files, so `import alphalab`
  fails. Never recreate the venv inside the project; `scripts/setup_mac.sh` sets this up.

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

## Decisions: ask Codex before asking Guohuan
When you reach a decision you would normally ask Guohuan about (choosing between options, approving a fix, launching a job):
1. Send Codex a self-contained question via agent-hud (`mcp__agent-hud__send_message`) or `codex exec "<question + context + options>"`, asking for ONE recommendation and a reason.
2. Follow Codex's recommendation, unless it falls under "Always ask Guohuan" below.
3. Log every delegated decision in `docs/DECISIONS.md`: date, question, Codex's answer, what you did.

### Always ask Guohuan (never delegate)
- Deleting files or data, rewriting git history, force-pushing, or pushing to GitHub
- Opening or touching the lockbox / held-out test data
- Any job expected to run longer than 2 hours, or anything that costs money
- Changing the pre-registered research design or hypotheses
- When Codex's answer is unclear, contradicts the test results, or Codex doesn't reply within 30 minutes

## Autonomous run rules
- Do NOT ask the user questions or wait for user input. The user is away.
- For any decision you would normally ask the user about (design choices, ambiguous requirements, which model/method to use), send the question to the Codex session using Agent HUD's send_message, wait for its reply, and follow it.
- If Codex does not reply within 10 minutes, make the most reasonable choice yourself, note it in DECISIONS.md, and continue.
- At the start, run `date` and record the start time. Before each new step, run `date` again. Stop all work 2 hours after the start time.
- When stopping, write a summary to RUN_SUMMARY.md: what was done, decisions Codex made, open issues.
