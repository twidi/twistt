# Twistt - Push-to-Talk Transcription Tool

A Linux speech-to-text transcription tool using OpenAI or Deepgram for STT with push-to-talk functionality.

## Features

- **Push-to-Talk**: Hold a function key (F1-F12) to record and transcribe
- **Toggle mode**: Double-tap the key to start recording, press again to stop
- **Smart transcription**: Text appears when you pause or stop speaking
- **Auto-output**: Automatically outputs transcribed text at cursor position
- **Multi-language support**: Transcribe in any language supported by the provider
- **Configurable audio gain**: Amplify microphone input if needed
- **Multiple model support**: Choose between `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`
- **Post-treatment**: Optional AI-powered correction of transcribed text for improved accuracy

## Requirements

- Linux (tested on X11 and Wayland)
- Python 3.11+
- `ydotool` for simulating keyboard input (by pasting or typing + pasting)
- OpenAI or Deepgram API key for transcription (depending on provider)
- OpenAI, Cerebras, or OpenRouter API key for post-treatment (if used)
- Microphone access

## Installation

### Using uv (Recommended)

The script is designed to run with [uv](https://github.com/astral-sh/uv), which handles dependencies automatically:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the script (dependencies will be auto-installed)
./twistt.py --help
```

### Using pip

If you prefer using pip:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python twistt.py --help
```

### System Dependencies

**ydotool** is required for output. It's a replacement for xdotool that works on both X11 and Wayland, used here to simulate typing and pasting.

**Important**: The versions available in Debian/Ubuntu repositories are too old. You'll need to build from source.

For installation instructions, see: https://docs.o-x-l.com/automation/ydotool.html

Here's a simplified systemd service for single-user setup:

```ini
# /etc/systemd/system/ydotoold.service
[Unit]
Description=ydotoold (root) for user 1000
# Ensure /run/user/1000 exists
Requires=user-runtime-dir@1000.service
After=user-runtime-dir@1000.service
# Start after display/user session
After=display-manager.service user@1000.service
BindsTo=user@1000.service

[Service]
Type=simple
# Avoid stale socket -> "Connection refused"
ExecStartPre=/usr/bin/rm -f /run/user/1000/.ydotool_socket
ExecStart=/usr/local/sbin/ydotoold --socket-path=/run/user/1000/.ydotool_socket --socket-own=1000:0
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

**Note**: If you use a custom socket path (as shown above with `/run/user/1000/.ydotool_socket`), you'll need to specify it when running twistt:
- Via environment variable: `YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket ./twistt.py`
- Or via argument: `./twistt.py --ydotool-socket /run/user/1000/.ydotool_socket`

## Configuration

### API Key Setup

Set your OpenAI API key(s) using one of these methods (in order of priority):

1. **Command line argument**: `--api-key YOUR_KEY`
2. **User config file**: `~/.config/twistt/config.env`
3. **Local .env file**: Create `.env` in the script directory
4. **Environment variable**: Export in your shell

Example `.env` or `config.env` file:
```env
# OpenAI API key (required if model from OpenAI, by default)
TWISTT_OPENAI_API_KEY=sk-...
# or
OPENAI_API_KEY=sk-...

# Deepgram API key (required if model from Deepgram)
TWISTT_DEEPGRAM_API_KEY=dg_...
# or
DEEPGRAM_API_KEY=dg_...

# Optional settings
TWISTT_HOTKEY=F9           # Single hotkey
TWISTT_HOTKEYS=F8,F9,F10   # Multiple hotkeys (comma-separated)
TWISTT_MODEL=gpt-4o-transcribe   # For OpenAI; for Deepgram use e.g. nova-2-general
TWISTT_LANGUAGE=en  # Leave empty or omit for auto-detect
TWISTT_GAIN=1.0
TWISTT_MICROPHONE=Elgato Wave 3  # Optional text filter to auto-select a microphone
TWISTT_DOUBLE_TAP_WINDOW=0.5  # Time window for double-tap detection
TWISTT_KEYBOARD=keychron  # Optional text filter to auto-select matching keyboard
TWISTT_YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket  # Optional, auto-detected by default

# Output mode
TWISTT_OUTPUT_MODE=batch  # batch (default) or full
TWISTT_USE_TYPING=false  # Type ASCII characters via ydotool instead of copy/paste (slower)

# Post-treatment settings (optional)
TWISTT_POST_TREATMENT_PROMPT="Fix grammar and punctuation"
TWISTT_POST_TREATMENT_PROMPT_FILE=/path/to/prompt.txt  # Alternative to direct prompt
TWISTT_POST_TREATMENT_MODEL=gpt-4o-mini  # Model for post-treatment
TWISTT_POST_TREATMENT_PROVIDER=openai  # Provider: openai, cerebras, or openrouter
# Post-treatment correct mode (apply corrections in-place with keyboard; requires batch output mode)
TWISTT_POST_TREATMENT_CORRECT=false

# Provider-specific API keys (for post-treatment)
TWISTT_CEREBRAS_API_KEY=csk-...  # Required if using cerebras provider
TWISTT_OPENROUTER_API_KEY=sk-or-...  # Required if using openrouter provider
```

### Available Options

| Option                 | Environment Variable                                | Default                       | Description                                                                                                                                       |
|------------------------|-----------------------------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `--hotkey`             | `TWISTT_HOTKEY` or `TWISTT_HOTKEYS`                 | F9                            | Push-to-talk key(s) (F1-F12), comma-separated for multiple                                                                                        |
| `--keyboard`           | `TWISTT_KEYBOARD`                                   | -                             | Filter text for automatically selecting the keyboard input device<br/>Pass without a value to force interactive selection and ignore env defaults |
| `--double-tap-window`  | `TWISTT_DOUBLE_TAP_WINDOW`                          | 0.5                           | Time window in seconds for double-tap detection                                                                                                   |
| `--model`              | `TWISTT_MODEL`                                      | gpt-4o-transcribe             | Transcription model (for OpenAI or Deepgram)                                                                                                      |
| `--language`           | `TWISTT_LANGUAGE`                                   | Auto-detect                   | Transcription language (ISO 639-1)                                                                                                                |
| `--gain`               | `TWISTT_GAIN`                                       | 1.0                           | Microphone amplification                                                                                                                          |
| `--microphone`         | `TWISTT_MICROPHONE`                                 | Default input                 | Text filter or ID to select the microphone<br/>Pass without a value to force interactive selection and ignore env defaults                        |
| `--openai-api-key`     | `TWISTT_OPENAI_API_KEY` or `OPENAI_API_KEY`         | -                             | OpenAI API key                                                                                                                                    |
| `--deepgram-api-key`   | `TWISTT_DEEPGRAM_API_KEY` or `DEEPGRAM_API_KEY`     | -                             | Deepgram API key                                                                                                                                  |
| `--ydotool-socket`     | `TWISTT_YDOTOOL_SOCKET` or `YDOTOOL_SOCKET`         | Auto-detect                   | Path to ydotool socket                                                                                                                            |
| `--post-prompt`        | `TWISTT_POST_TREATMENT_PROMPT`                      | -                             | Post-treatment instructions                                                                                                                       |
| `--post-prompt-file`   | `TWISTT_POST_TREATMENT_PROMPT_FILE`                 | -                             | File containing post-treatment prompt                                                                                                             |
| `--post-model`         | `TWISTT_POST_TREATMENT_MODEL`                       | gpt-4o-mini                   | Model for post-treatment                                                                                                                          |
| `--post-provider`      | `TWISTT_POST_TREATMENT_PROVIDER`                    | openai                        | Provider for post-treatment (openai, cerebras, openrouter)                                                                                        |
| `--post-correct`       | `TWISTT_POST_TREATMENT_CORRECT`                     | false                         | Apply post-treatment by correcting already-output text in-place (only in batch output mode)                                                       |
| `--cerebras-api-key`   | `TWISTT_CEREBRAS_API_KEY` or `CEREBRAS_API_KEY`     | -                             | Cerebras API key                                                                                                                                  |
| `--openrouter-api-key` | `TWISTT_OPENROUTER_API_KEY` or `OPENROUTER_API_KEY` | -                             | OpenRouter API key                                                                                                                                |
| `--output-mode`        | `TWISTT_OUTPUT_MODE`                                | batch                         | Output mode: batch (incremental) or full (complete on release)                                                                                    |
| `--use-typing`         | `TWISTT_USE_TYPING`                                 | false                         | Type ASCII characters directly (slower); clipboard still handles non-ASCII                                                                        |
| `--config PATH`        | `TWISTT_CONFIG`                                     | `~/.config/twistt/config.env` | Load configuration overrides from the specified file instead of the default user config                                                           |
| `--save-config [PATH]` | `TWISTT_CONFIG`                                     | false                         | Persist provided command-line values to a config file (defaults to `~/.config/twistt/config.env` or `TWISTT_CONFIG` if set)                       |

Selecting a microphone sets the `PULSE_SOURCE` environment variable for Twistt only, so your system default input stays untouched. Run `./twistt.py --microphone` without a value to pick from the list even if an environment variable is set.

Use `--config` (or `TWISTT_CONFIG`) to load settings from a specific file while leaving the default user config untouched. Use `--save-config` to capture only the options you explicitly pass on the command line; existing keys in the config file are preserved. Provide a path (or set `TWISTT_CONFIG`) to control which file gets written. `TWISTT_CONFIG` is read only from the process environment—do not place it in `.env` files or `config.env`.

### Config Inheritance

Config files can define `TWISTT_PARENT_CONFIG` to inherit from another config file. Values in the child file take precedence. This allows creating presets that only specify what differs from a base configuration:

```bash
# base.env - shared settings
TWISTT_OPENAI_API_KEY=sk-...
TWISTT_OUTPUT_MODE=batch
TWISTT_POST_TREATMENT_PROMPT=Please correct any errors

# work.env - inherits base, changes hotkey
TWISTT_PARENT_CONFIG=base.env
TWISTT_HOTKEY=F8

# gaming.env - inherits base, different settings
TWISTT_PARENT_CONFIG=base.env
TWISTT_HOTKEY=F9
TWISTT_OUTPUT_MODE=full
```

Parent paths can be relative (resolved from the child config's directory) or absolute. Circular references are detected and will cause an error.

## Usage

### Basic Usage

```bash
# Start with default settings (F9 key, auto-detect language)
./twistt.py

# Use F5 key with English transcription
./twistt.py --hotkey F5 --language en

# Use multiple hotkeys
./twistt.py --hotkey F8,F9,F10

# Force French language
./twistt.py --language fr

# Increase microphone sensitivity
./twistt.py --gain 2.0

# Enable post-treatment to fix grammar and punctuation
./twistt.py --post-prompt "Fix grammar, punctuation, and obvious errors"

# Use a file for more complex post-treatment instructions
./twistt.py --post-prompt-file instructions.txt

# Specify a different model for post-treatment
./twistt.py --post-prompt "Make the text more formal" --post-model gpt-4o

# Use Cerebras for post-treatment (faster inference)
./twistt.py --post-prompt "Fix errors" --post-provider cerebras --post-model llama3-8b

# Use OpenRouter for post-treatment (access to many models)
./twistt.py --post-prompt "Fix errors" --post-provider openrouter --post-model meta-llama/llama-3.2-3b-instruct

# Post-treatment correct mode: output raw immediately then update in place via post-treatment
./twistt.py --post-prompt "Fix grammar" --post-correct

# Use full output mode (wait for hotkey release to output/process)
./twistt.py --output-mode full

# Type ASCII characters directly (slower; non-ASCII characters are still handled via clipboard)
./twistt.py --use-typing

# Use Deepgram as provider
TWISTT_PROVIDER=deepgram TWISTT_DEEPGRAM_API_KEY=dg_xxx ./twistt.py --model nova-2-general --language fr

# Save your preferred options for next time
./twistt.py --language fr --gain 2.0 --microphone "Elgato Wave 3" --save-config

# Save to a custom config file
./twistt.py --language fr --gain 2.0 --save-config ~/.config/twistt/presets/french.env

# Load a custom preset before launching
./twistt.py --config ~/.config/twistt/presets/french.env
```

### How It Works

Twistt supports two recording modes:

#### Push-to-Talk Mode (Hold)
1. **Start the script**: Run `./twistt.py`
2. **Position cursor**: Click where you want text to appear
3. **Hold to record**: Press and hold one of your configured hotkeys (default: F9)
4. **Speak**: Talk while holding the key
5. **Release to transcribe**: Let go of the key
6. **Auto-output**: Text is automatically output at cursor position

#### Toggle Mode (Double-Tap)
1. **Start the script**: Run `./twistt.py`
2. **Position cursor**: Click where you want text to appear
3. **Double-tap to start**: Press-release-press the same hotkey quickly (within 0.5s)
4. **Speak freely**: Recording continues without holding any key
5. **Press to stop**: Press the same hotkey once to stop recording (only the hotkey that started toggle mode can stop it)
6. **Auto-output**: Text is automatically output at cursor position

The transcription appears where the cursor is located.

An indicator ("(Twisting...)" text) is shown at the cursor position when recording is active, or text is being output or post-treatment is running. 

### Output Modes

Twistt supports two output modes that control when text is processed and output:

- **batch mode** (default): Text is processed and can be output incrementally as you speak. Each pause triggers processing of that segment. With post-treatment enabled, each segment maintains context from previous segments.

- **full mode**: All text is accumulated while you hold the key and only processed/output when you release it. With post-treatment, the entire text is processed at once without maintaining context between sessions. This mode is useful when you want to speak a complete thought before any processing occurs.

### Tips

- **Shift mode**: Press Shift at any time while recording to use Ctrl+Shift+V instead of Ctrl+V to paste (useful for terminals). Shift can be pressed:
  - When starting recording (together with the hotkey)
  - At any moment while holding the hotkey
  - The earliest Shift press is remembered for the entire recording session
- **Multiple sentences**: Keep holding the key to transcribe continuously
- **Pause support**: Brief pauses are handled automatically
- **Live feedback**: Watch the terminal to see transcription as it processes
- **Output mode choice**: Use `--output-mode full` when you want to complete your entire thought before processing
- **Post-treatment**: Enable for improved accuracy, especially useful for:
  - Fixing punctuation and capitalization
  - Correcting common speech-to-text errors
  - Adapting text style (formal, informal, technical)
  - Language-specific corrections

## Keyboard Detection

The script automatically detects your physical keyboard. If multiple keyboards are found, you'll be prompted to select one. Virtual keyboards are automatically filtered out. Set `--keyboard "partial name"` or `TWISTT_KEYBOARD=partial name` to pre-filter devices and auto-select when only one match remains. Pass `--keyboard` with no value to always display the selection menu and ignore any configured default.

## Post-Treatment (Optional)

Post-treatment uses AI to improve transcription accuracy by correcting errors, fixing punctuation, and applying custom transformations. It's activated automatically when you provide a prompt.

### Supported Providers

#### Transcription

You can choose between different AI providers for transcription:

- **OpenAI**: Uses OpenAI's GPT transcribe models (`gpt-4o-transcribe` (default) `gpt-4o-mini-transcribe`). Better to not use `--use-typing`.
- **Deepgram**: Uses Deepgram's Nova models (`nova-2`, `nova-3`). Really real time but more expensive. Great with `--use-typing`

#### Post-Treatment

You can choose between different AI providers for post-treatment:

- **OpenAI** (default): Uses OpenAI's GPT models
- **Cerebras**: Fast inference with open-source models ([docs](https://inference-docs.cerebras.ai/)). Models can be free!
- **OpenRouter**: Access to many different AI models ([docs](https://openrouter.ai/)). Provides paid cerebras models like GPT-OSS.

Each provider requires its own API key, which can be set via environment variables or command-line arguments.

### Creating a Post-Treatment Prompt

You can provide instructions directly via command line or use a file for more complex prompts:

**Example prompt file** (`corrections.txt`):
```
Fix any grammar and punctuation errors.
Ensure proper capitalization.
Expand common abbreviations.
Remove filler words like "um" and "uh".
Keep the conversational tone.
```

Then use it with:
```bash
./twistt.py --post-prompt-file corrections.txt
```

### Post-Treatment Examples

```bash
# Simple corrections
./twistt.py --post-prompt "Fix grammar and punctuation"

# Technical writing
./twistt.py --post-prompt "Use technical vocabulary, expand acronyms on first use"

# Formal style
./twistt.py --post-prompt "Make the text more formal and professional"

# Use a more powerful model for complex corrections
./twistt.py --post-prompt-file complex_rules.txt --post-model gpt-4o

# Use Cerebras for faster processing
export CEREBRAS_API_KEY=csk-...
./twistt.py --post-prompt "Fix errors" --post-provider cerebras --post-model llama3-70b

# Use OpenRouter for access to various models
export OPENROUTER_API_KEY=sk-or-...
./twistt.py --post-prompt "Improve clarity" --post-provider openrouter --post-model anthropic/claude-3-haiku
```

**Note**: Post-treatment adds a small delay (typically under 1 second) as it processes the text through the AI model.

## Language Support

By default, the tool auto-detects the language you're speaking. You can also specify a language using ISO 639-1 codes:
- `en` - English
- `fr` - French  
- `es` - Spanish
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese
- `zh` - Chinese
- And many more...

Leave the language parameter empty to use auto-detection.

## Troubleshooting

### "No physical keyboard detected"
- The script needs to monitor keyboard events
- Run with appropriate permissions if needed
- Select your keyboard manually from the list

### "ydotool error"
- Ensure ydotool daemon is running: `sudo ydotoold &`
- If using a custom socket path, set it via `YDOTOOL_SOCKET` environment variable or `--ydotool-socket` argument

### "Permission denied on /dev/input/eventX"
- Add your user to the `input` group: `sudo usermod -a -G input $USER`
- Log out and back in for changes to take effect
- Or run with sudo (not recommended for regular use)

### Audio issues
- Check microphone permissions
- Adjust `--gain` if audio is too quiet/loud
- Ensure no other application is using the microphone

## Security Notes

- The API key is sent only to OpenAI's servers
- Audio is processed in real-time and not stored locally
- Transcriptions are only kept in memory during the session

## Ideas

We maintain a curated list of potential enhancements in IDEAS.md. If you have suggestions or want to pick something up, check it out and open an issue or PR.

## Author

Stephane "Twidi" Angel, with the help of @claude and @codex

## License

MIT License - See LICENSE file for details
