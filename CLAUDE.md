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
- numpy, sounddevice (audio processing)
- websockets (OpenAI real-time API)
- pyperclipfix (clipboard operations)
- evdev (keyboard event monitoring)
- python-dotenv, platformdirs (configuration)
- python-ydotool (keyboard simulation for paste)

## Architecture

The application is a single-file Python script with the following key components:

1. **AudioTranscriber class**: Core logic for WebSocket connection to OpenAI, audio streaming, and transcription handling
2. **Keyboard monitoring**: Uses evdev to detect F-key presses for push-to-talk
3. **Audio capture**: Uses sounddevice to record from microphone in real-time
4. **Auto-paste**: Uses python-ydotool to simulate Ctrl+V for automatic text insertion

## Configuration

Configuration priority (highest to lowest):
1. Command-line arguments
2. User config: `~/.config/twistt/config.env`
3. Local `.env` file in script directory
4. Environment variables

Key environment variables use `TWISTT_` prefix (e.g., `TWISTT_OPENAI_API_KEY`, `TWISTT_HOTKEY`).

## Testing

No formal test suite exists. Testing is manual:
1. Run the script with various hotkeys
2. Test transcription in different languages
3. Verify paste functionality works in different applications