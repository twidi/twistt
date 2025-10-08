# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Twistt is a Linux push-to-talk transcription tool that uses OpenAI's real-time API for speech-to-text conversion. The entire application is contained in a single Python script (`twistt.py`) that can be run with uv.

## Development Setup and Commands

### Running the Application

```bash
# Using uv (recommended - auto-handles dependencies)
./twistt.py --help

# Using Python directly (requires dependencies installed)
python twistt.py --help
```

### Dependencies

The project uses inline script dependencies (PEP 723) specified in `twistt.py`. Dependencies are also listed in `requirements.txt` for pip users:
- numpy, sounddevice (audio capture)
- soundcard (microphone discovery)
- websockets (OpenAI real-time API)
- pyperclipfix (clipboard operations)
- evdev (keyboard event monitoring)
- python-dotenv, platformdirs (configuration)
- python-ydotool (keyboard simulation for paste)
- openai (OpenAI SDK for post-treatment feature)

## Architecture

The application is a single-file Python script with the following key components:

1. **AudioTranscriber class**: Core logic for WebSocket connection to OpenAI, audio streaming, and transcription handling
2. **Keyboard monitoring**: Uses evdev to detect F-key presses for push-to-talk
3. **Audio capture**: Uses sounddevice to record from microphone in real-time (Pulse source pinned via soundcard)
4. **Auto-paste/typing**: Uses python-ydotool to paste text or optionally type ASCII characters directly
5. **Post-treatment**: Optional AI-powered correction using various providers (OpenAI, Cerebras, OpenRouter) to improve transcription accuracy

## Configuration

Configuration priority (highest to lowest):
1. Command-line arguments
2. User config: `~/.config/twistt/config.env`
3. Local `.env` file in script directory
4. Environment variables

Key environment variables use `TWISTT_` prefix (e.g., `TWISTT_OPENAI_API_KEY`, `TWISTT_HOTKEY` or `TWISTT_HOTKEYS`, `TWISTT_POST_TREATMENT_PROMPT` (can be text or file path), `TWISTT_POST_TREATMENT_PROVIDER`, `TWISTT_OUTPUT_MODE`, `TWISTT_POST_CORRECT`, `TWISTT_POST_TREATMENT_DISABLED`, `TWISTT_USE_TYPING`, `TWISTT_SILENCE_DURATION`).

`TWISTT_USE_TYPING` (or `--use-typing`) enables per-character typing for ASCII text, which is slower because of key delays; clipboard paste remains the fallback for non-ASCII characters.

Multiple hotkeys can be specified by separating them with commas (e.g., `TWISTT_HOTKEY=F8,F9,F10` or `--hotkey F8,F9,F10`).

Provider-specific API keys:
- `TWISTT_CEREBRAS_API_KEY` or `CEREBRAS_API_KEY` for Cerebras
- `TWISTT_OPENROUTER_API_KEY` or `OPENROUTER_API_KEY` for OpenRouter

## Output Modes

The application supports three output modes (`--output-mode` or `TWISTT_OUTPUT_MODE`):
- **batch** (default): Processes and pastes text incrementally as segments are detected by the API. Each segment maintains context from previous segments when using post-treatment.
- **full**: Accumulates all text while the key is held and processes/pastes only when released. Post-treatment runs without context between sessions.
- **none**: Disables all paste/typing output. Transcription and post-treatment still execute (BufferTask simply discards insert/correction commands), keeping terminal feedback available without touching the cursor.

Additionally, a post-correction mode (`--post-correct` or `TWISTT_POST_CORRECT`) pastes raw transcription immediately and, once the post-treatment result is fully available, edits the pasted text in-place using only arrow keys and backspace to minimize disruption.

## Testing

No formal test suite exists. Testing is manual:
1. Run the script with various hotkeys (including multiple hotkeys comma-separated)
2. Test transcription in different languages
3. Verify paste functionality works in different applications
4. Test post-treatment with different prompts, models, and providers
5. Verify post-treatment maintains transcription order
6. Test provider switching (OpenAI, Cerebras, OpenRouter) for post-treatment
7. Test output modes (batch vs full vs none) with and without post-treatment
8. Test toggle mode activation/deactivation with same hotkey when multiple hotkeys are configured
