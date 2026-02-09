# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Twistt is a Linux push-to-talk transcription tool that uses OpenAI, Deepgram, or Mistral real-time APIs for speech-to-text conversion. The entire application is contained in a single Python script (`twistt.py`) that can be run with uv.

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
- websockets (OpenAI/Deepgram real-time API)
- pyperclipfix (clipboard operations)
- evdev (input device event monitoring)
- python-dotenv, platformdirs (configuration)
- python-ydotool (keyboard simulation for paste)
- openai (OpenAI SDK for post-treatment feature)
- mistralai[realtime] (Mistral real-time transcription API)

## Architecture

The application is a single-file Python script with the following key components:

1. **Transcription task classes**: Provider-specific classes (OpenAI, Deepgram, Mistral) inheriting from `BaseTranscriptionTask` for WebSocket-based real-time transcription
2. **Input device monitoring**: Uses evdev to detect F-key presses for push-to-talk across all input devices (keyboards, mice with remapped buttons, macropads, etc.)
3. **Audio capture**: Uses sounddevice to record from microphone in real-time (Pulse source pinned via soundcard)
4. **Auto-paste/typing**: Uses python-ydotool to paste text or optionally type ASCII characters directly
5. **Post-treatment**: Optional AI-powered correction using various providers (OpenAI, Cerebras, OpenRouter) to improve transcription accuracy

## Configuration

Configuration priority (highest to lowest):
1. Command-line arguments
2. User config: `~/.config/twistt/config.env`
3. Local `.env` file in script directory
4. Environment variables

Key environment variables use `TWISTT_` prefix (e.g., `TWISTT_OPENAI_API_KEY`, `TWISTT_HOTKEY` or `TWISTT_HOTKEYS`, `TWISTT_POST_TREATMENT_PROMPT`, `TWISTT_POST_TREATMENT_PROVIDER`, `TWISTT_OUTPUT_MODE`, `TWISTT_POST_CORRECT`, `TWISTT_POST_TREATMENT_DISABLED`, `TWISTT_USE_TYPING`, `TWISTT_KEYBOARD_DELAY`, `TWISTT_SILENCE_DURATION`, `TWISTT_INDICATOR_TEXT`, `TWISTT_INDICATOR_TEXT_DISABLED`).

**Configuration files (`TWISTT_CONFIG` and `-c`/`--config`)**:
- Environment variable: can specify multiple config files separated by `::` delimiter (e.g., `TWISTT_CONFIG="base.env::local.env"`)
- `-c` argument can be specified multiple times: `-c base.env -c local.env`
- Each `-c` value can contain `::` separators for multiple files: `-c "base.env::local.env"`
- Config files are loaded in reverse order (last specified file has highest priority)
- Example: `-c file1.env -c file2.env` → file2.env values override file1.env values
- Files can use relative paths (resolved in config dir `~/.config/twistt/`) or absolute paths
- Each config file can define `TWISTT_PARENT_CONFIG` to inherit from another config file
- **Include default config**: Prefix any `-c` value with `::` to include the default config (`~/.config/twistt/config.env`) first:
  - `-c ::` → uses only default config
  - `-c ::fr.env` → combines default config + fr.env (fr.env as modifier)
  - `-c base.env -c ::local.env` → combines default config + base.env + local.env
  - Without `::` prefix: `-c` replaces default config entirely

**Post-treatment prompts (`TWISTT_POST_TREATMENT_PROMPT` and `-p`/`--post-prompt`)**:
- Environment variable: can specify multiple prompts separated by `::` delimiter (e.g., `TWISTT_POST_TREATMENT_PROMPT="prompt1.txt::Fix grammar::prompt2.txt"`)
- Each part is resolved as a file path (if exists) or literal text
- All prompts are concatenated with double newlines (`\n\n`) between them
- `-p` argument can be specified multiple times: `-p file1.txt -p "Fix grammar" -p file2.txt`
- Each `-p` value can contain `::` separators for multiple prompts: `-p "prompt1.txt::Fix grammar"`
- If ANY `-p` value starts with `::`, the environment variable is included first:
  - `-p ::` → uses only `TWISTT_POST_TREATMENT_PROMPT`
  - `-p "::extra.txt"` → combines env var + extra.txt
  - `-p file1.txt -p "::file2.txt"` → combines env var + file1.txt + file2.txt
- Without `::` prefix: `-p` replaces `TWISTT_POST_TREATMENT_PROMPT` entirely
- Order of concatenation: env var (if requested) → all `-p` values in order (with `::` prefix removed)

`TWISTT_USE_TYPING` (or `--use-typing`) enables per-character typing for ASCII text, which is slower because of key delays; clipboard paste remains the fallback for non-ASCII characters.

`TWISTT_KEYBOARD_DELAY` (or `--keyboard-delay` / `-kd`) sets the delay in milliseconds between all keyboard actions (typing, paste, navigation keys). Default is 20ms. Increase this value (e.g., 25-50ms) if you experience character ordering issues or missing characters in terminal emulators.

Multiple hotkeys can be specified by separating them with commas (e.g., `TWISTT_HOTKEY=F8,F9,F10` or `--hotkey F8,F9,F10`).

`TWISTT_LOG` (or `--log`) specifies the path to the log file. Default is `~/.config/twistt/twistt.log`. All transcription sessions (configuration panel and finalized transcriptions) are logged to this file.

`TWISTT_INDICATOR_TEXT` (or `--indicator-text` / `-it`) customizes the text shown at the cursor position while recording/processing. Default is `" (Twistting...)"`. `TWISTT_INDICATOR_TEXT_DISABLED` (or `--no-indicator` / `-ni`) disables the indicator entirely when set to `true`.

Provider-specific API keys:
- `TWISTT_MISTRAL_API_KEY` or `MISTRAL_API_KEY` for Mistral transcription
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
9. Test Mistral transcription provider (no server-side VAD, client-side timeout for finalization)
7. Test output modes (batch vs full vs none) with and without post-treatment
8. Test toggle mode activation/deactivation with same hotkey when multiple hotkeys are configured
- never care about retro-compatibility, it's a personal project
- pas besoin d'etre precis apres chaque update quand tu mets a jour CLAUDE.md. par contre il est important de mettre a jour le readme si les modifications ont un impact sur l'utilisateur (notamment tout ce qui est arguments/variables d'environement)