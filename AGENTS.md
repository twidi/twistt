# Repository Guidelines

## Project Structure & Module Organization
- `twistt.py`: Main Python CLI script (single-file app).
- `twistt_osd.py`: OSD overlay daemon (runs under system Python with GTK4/gtk4-layer-shell). Spawned as a subprocess by `twistt.py`.
- `requirements.txt`: Runtime dependencies for `pip` users.
- `.env` (local) and `~/.config/twistt/config.env`: Optional configuration files loaded at runtime.
- Do not split into multiple files beyond the current structure. Keep main logic in `twistt.py` and OSD logic in `twistt_osd.py`. Only split further if a maintainer explicitly requests it.

## Build, Test, and Development Commands
- With uv (recommended): `./twistt.py --help` runs with deps resolved automatically.
- With pip:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `python twistt.py --help`
- System dependency: `ydotoold` must be running for paste simulation; pass custom socket via `--ydotool-socket` or `YDOTOOL_SOCKET`.

## Coding Style & Naming Conventions
- Python 3.11+, PEP 8, 4-space indents.
- Names: functions/vars `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- CLI flags use `--kebab-case`; env vars use `TWISTT_*` (e.g., `TWISTT_OPENAI_API_KEY`, `TWISTT_OUTPUT_MODE`).
- Keep functions small; prefer early returns; add docstrings/comments where behavior isn’t obvious.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (≤50 chars), body for rationale if needed. Example: `Replace subprocess ydotool calls with python-ydotool library`.
- Reference issues in bodies (e.g., `Fixes #12`). Group mechanical changes separately from behavior changes.
- PRs: clear description, reproduction steps, and testing notes. Include screenshots/terminal snippets when behavior changes. Update README/options tables when adding flags or env vars.

## Security & Configuration Tips
- Never commit secrets. Prefer `TWISTT_OPENAI_API_KEY` (or `OPENAI_API_KEY`). For provider-specific keys: `TWISTT_CEREBRAS_API_KEY` or `TWISTT_OPENROUTER_API_KEY`. `.env` is for local use only.
- Avoid recommending `sudo`; prefer adding the user to the `input` group for `/dev/input/*` access.
- If adding deps, update both the uv script header in `twistt.py` and `requirements.txt`.
- When adding new features with env vars, update README.md options table and CLAUDE.md dependencies list.

## Agent-Specific Instructions
- Keep changes minimal and focused. Do not reformat unrelated code.
- Validate both uv and pip flows. Reflect config surface changes in README and this file.
