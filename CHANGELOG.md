# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Since this project does not use versioned releases, entries are organized by date.

## 2026-02-09

### Added

- Mistral/Voxtral real-time transcription provider (`voxtral-mini-transcribe-realtime-2602` model) with `-kmi/--mistral-api-key` CLI argument and `TWISTT_MISTRAL_API_KEY` env var
- Configurable toggle mode: choose between single-tap and double-tap activation via `--toggle-mode` (`single`/`double`). Default remains double-tap
- Configurable indicator text via `TWISTT_INDICATOR_TEXT` (`--indicator-text` / `-it`) and option to disable it via `TWISTT_INDICATOR_TEXT_DISABLED` (`--no-indicator` / `-ni`)
- Hotkey detection now listens to all input devices (mice, macropads, etc.), not just keyboards, enabling hotkeys remapped to mouse buttons
- Debug logging for hotkey event processing (active when `TWISTT_DEBUG=true`)

### Fixed

- Multi-second delay in full output mode between key release and text insertion

## 2026-01-09

### Fixed

- Last sentence lost when releasing hotkey too soon with OpenAI transcription

## 2025-10-25

### Added

- Configurable keyboard delay via `TWISTT_KEYBOARD_DELAY` (`-kd/--keyboard-delay`, default 20ms)
- Session logging to file via `--log` / `TWISTT_LOG` (default `~/.config/twistt/twistt.log`)
- Files section in configuration display showing loaded config files, prompt file, and log file path
- Support for multiple post-treatment prompts with `::` separator and multiple `-p` arguments
- Support for multiple `-c` arguments for config files with `::` separator
- `--check` argument to verify configuration without logging
- `--list-configs` option to list available configuration files
- Display silence duration in config panel

### Changed

- Unified `--post-prompt` argument: removed `--post-prompt-file` (`-pf`) and `TWISTT_POST_TREATMENT_PROMPT_FILE`; `--post-prompt` now auto-detects file path vs direct text
- `-p` without value removed (breaking change); use `-p ::` to include env var prompts
- Configuration logging delayed until first transcription

### Fixed

- Terminal output scrolling: completed sessions now print directly instead of accumulating in Rich Live area
- Post-treatment text duplication when toggled via Alt key

## 2025-10-08

### Changed

- `-p/--post-prompt` now accepts both text and file paths (auto-detection)
- `-p` flag without value enables post-treatment using `TWISTT_POST_TREATMENT_PROMPT` env variable

## 2025-09-26

### Added

- Configurable silence duration via `TWISTT_SILENCE_DURATION`
- Terminal user interface with Rich
- Output mode `none` for no buffer output except in the terminal
- Alt key toggle for post-treatment during recording (if configured)

### Fixed

- Deepgram problems due to empty transcripts
- Silence detection now respects configured duration for Deepgram

## 2025-09-25

### Added

- Short CLI flags: `-npc` (`--no-post-correct`), `-nt` (`--no-use-typing`), and other short flags
- `--no-post` flag to disable post-treatment even if configured

### Fixed

- `-h/--help` flag

## 2025-09-24

### Added

- `--config name` shorthand resolving to `~/.config/twistt/{name}.env`
- Debug mode activation via env var

### Fixed

- Saving config always applying unintentionally
- Buffer tasks now executed in sequence order instead of queue insertion order

## 2025-09-23

### Added

- Deepgram transcription provider
- `--keyboard` arg and matching env var to filter keyboard device
- `--microphone` arg and env var to select microphone
- Interactive device selection: use `--microphone` or `--keyboard` without argument to pick from a list
- `--save-config` option to persist current settings
- `TWISTT_CONFIG` env var to set config file path
- Config file inheritance (parent config support)

### Fixed

- WebSocket handling when closed while listening (Deepgram)
- Deepgram "done" event timeout handling
- Post-treatment error handling

## 2025-09-21

### Added

- `--use-typing` option: type characters instead of pasting for ASCII chars
- Clipboard restoration after each paste operation

### Fixed

- Output mode batch with post-correct enabled
- Indicator display in full mode with post-treatment

## 2025-09-20

### Added

- Activity indicator on output while processing
- Real-time delta output from transcription (when output mode/post-treatment allows)
- Configurable delay between keyboard actions
- Segments now output in numeric order, not readiness order

## 2025-09-08

### Added

- Post-correct mode with in-place corrections (`--post-correct` / `TWISTT_POST_CORRECT`)
- Toggle mode with double-tap activation (`--double-tap-window` / `TWISTT_DOUBLE_TAP_WINDOW`)
- Multiple hotkeys support via comma-separated values (`TWISTT_HOTKEYS`)

### Changed

- Shift key detection improved: state captured at any point during recording

## 2025-09-07

### Added

- Post-treatment feature for AI-powered text correction after transcription
- Multiple post-treatment providers: OpenAI, Cerebras, OpenRouter (`--post-provider`)
- Streaming support for post-treatment API
- Output mode option: `batch` (incremental) vs `full` (accumulate until key release) via `--output-mode`

## 2025-09-06

### Added

- Initial release: push-to-talk transcription tool using OpenAI API
- Hotkey-activated recording (F1-F12)
- Real-time speech-to-text with automatic text insertion at cursor
- Support for `YDOTOOL_SOCKET` env var and `--ydotool-socket` CLI argument
