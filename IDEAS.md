# Twistt – Feature Ideas

This document collects potential enhancements, with a focus on working reliably with the cursor in a terminal or any regular app (everything is done via simulated paste/keystrokes, no app-specific integrations).

## Ergonomics & Safety

- Quick undo: `--undo-hotkey` to remove the last paste (we record pasted length and send Backspace). Handy when a segment is wrong.
- Cancel current segment: while holding, press `Esc` (or `Alt`) to discard the segment (paste nothing).
- Auto newline: `--newline-on-release` to send Enter at the end; `--newline-with-shift` to add it only if Shift was pressed during recording (useful for terminals).
- Audio cues: short beeps at start/stop and when VAD detects speech (`--beep`, with `--beep-volume`).
- Quick history: triple‑tap the hotkey to re‑paste the last transcription (without speaking again).

## Input & Hotkeys

- More flexible hotkeys: allow `capslock`, `scrolllock`, `rightalt` for `--hotkey` (optional, documented). Still evdev‑based, so remains global.
- USB pedals: foot pedals now work out of the box since all input devices capable of emitting the hotkey are automatically detected. Use `--keyboard "pedal name"` to restrict to a specific device if needed.
- Temporary “mode” modifier: holding `Alt` while dictating toggles “Ctrl+Shift+V paste” or “no paste”, depending on config.

## Output & Pasting

- Output templates: `--prefix` and `--suffix` to automatically wrap dictated text (quotes, backticks, Markdown brackets, etc.).
- Mode presets for text/code/shell/email: `--mode text|code|shell|email` apply small local rules (no LLM), e.g. `shell` removes trailing punctuation and avoids capitals, `email` capitalizes sentences, `code` wraps in backticks or triple backticks.
- Always terminal‑safe paste: `--always-shift-paste` forces Ctrl+Shift+V even without Shift pressed (more reliable on Wayland/terminals).

## Post‑Processing & Dictation

- Opt‑in voice commands: `--voice-commands` with a prefix like “command:” (or a short hotword) for actions: “new paragraph”, “comma”, “delete last sentence”, “wrap in backticks”, “press enter”. Without the prefix, never interpret as commands (avoid false positives).
- Spoken punctuation: map words like “period”, “comma”, “new paragraph” → symbols; toggled by `--spoken-punct`.
- Dictionary/bias: `--bias-phrases foo,bar,baz` injected into the post‑processing prompt to normalize domain terms (without touching the STT model).
- Style memory: `--style-file` to keep a small style/glossary context reused across segments (especially helpful in `batch`).
- Live language toggle: a secondary hotkey switches language on the fly (e.g., `F10` forces `--language fr` while held; otherwise auto).

## Performance & Robustness

- Resilient ydotool: if the socket drops, auto‑reinit and fall back to “type” strategy as a safety net.
- Streaming granularity: expose `--stream-buffer-tokens` (current: 5) to balance latency vs stability.
- Key timing: `--key-delay-ms` to adapt keystroke sequence speed (slow terminals/VMs).
- Light audio filters: simple `--highpass`/`--lowcut` (NumPy) preprocessing for boomy/noisy mics (no heavy deps).
- Safety limits: `--max-paste-chars` prevents pasting huge blocks accidentally (log + chunk into slices).

## Development & Debug

- `--dry-run`: show exactly what would be pasted/edited without sending keys (clear logging). Great for testing modes.
- `--log-file` and `--verbose`: trace VAD (speech_started), correction sequencing, and latency metrics for STT/post‑processing.
- `--no-paste`: only copy to clipboard (useful for demos or paste‑hostile apps).
- `--seed-prompt-file`: inject an initial context for post‑processing (style, glossary) before any dictation.

## Suggested Priority (Quick Wins)

- `--preserve-clipboard`, `--newline-on-release`, `--always-shift-paste`
- `--undo-hotkey`
- `--mode text|code|shell` (local rules, no LLM)
- `--paste-strategy type` (clipboard‑free fallback)
