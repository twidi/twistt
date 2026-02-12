# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Since this project does not use versioned releases, entries are organized by date.

## 2026-02-12

### Fixed

- Post-treatment not displaying in full output mode: the session was finalized before post-treatment had a chance to start, so the result was never shown in the console or OSD

### Changed

- OSD session end logic centralized: the main app now tells the OSD when a session is complete instead of the OSD duplicating the session completion logic internally

## 2026-02-10

### Added

- Live transcription OSD overlay for Wayland/Hyprland: a glass-morphism on-screen display showing a 60-bar spectrum analyzer with absolute dB scaling, a color-coded vertical dB level meter, real-time transcript text with fade-out scrolling, and pulsing state indicators. Runs as a separate daemon process (`twistt_osd.py`) under system Python with `gtk4-layer-shell` via `LD_PRELOAD`. Communicates via Unix socket IPC. Configurable size via `TWISTT_OSD_WIDTH`/`TWISTT_OSD_HEIGHT` (`--osd-width`/`--osd-height`). Disable with `TWISTT_OSD_DISABLED` / `--no-osd` / `-nosd`. Gracefully skipped when system dependencies are missing
- OSD monitor and position configuration: force OSD to a specific monitor via `TWISTT_OSD_MONITOR` (`--osd-monitor`, index 0/1/2...) and set position as percentage via `TWISTT_OSD_X`/`TWISTT_OSD_Y` (`--osd-x`/`--osd-y`, 0-100, center of window). Position requires a forced monitor to compute pixel offsets from the monitor geometry. Without `--osd-monitor`, the compositor chooses the screen (typically follows mouse) and the OSD is centered at the top

### Fixed

- evdev `ReadIterator` `InvalidStateError` on shutdown: devices are now closed before cancelling reader tasks to avoid race condition with evdev's internal Future handling

## 2026-02-09

### Added

- Audio ducking: automatically reduces system audio volume during recording to prevent microphone interference. Uses PulseAudio/PipeWire via `pulsectl`. Enabled by default with 50% reduction. Configurable via `TWISTT_DUCKING_PERCENT` (`--ducking-percent` / `-dp`). Disable with `TWISTT_DUCKING_DISABLED` / `--no-ducking` / `-nd`
- System tray icon displaying a microphone that changes color by state: cyan (idle), orange (recording), green (transcribing), violet (post-processing). Active states pulse to draw attention. Optional dependencies (`pystray`, `Pillow`, `PyGObject`) are auto-installed on first run via `uv pip`. Disable with `TWISTT_TRAY_ICON_DISABLED` / `--no-tray-icon` / `-nti`
- KDE Plasma 6 panel widget (`kde-widget/`) as an alternative to the system tray icon. Displays the same color-coded microphone with smooth pulsing animations. Can be placed anywhere in the panel, independently of the system tray. Communicates via a state file — no extra Python dependencies needed. Install with `kpackagetool6`
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
