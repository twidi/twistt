# Twistt - Push-to-Talk Transcription Tool

A Linux speech-to-text transcription tool using OpenAI, Deepgram, or Mistral for STT with push-to-talk functionality.

## Features

- **Push-to-Talk**: Hold a function key (F1-F12) to record and transcribe
- **Toggle mode**: Tap (or double-tap) the key to start recording, press again to stop
- **Smart transcription**: Text appears when you pause or stop speaking
- **Auto-output**: Automatically outputs transcribed text at cursor position
- **Multi-language support**: Transcribe in any language supported by the provider
- **Configurable audio gain**: Amplify microphone input if needed
- **Multiple model support**: Choose between `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, Deepgram Nova, or Mistral Voxtral models
- **Post-treatment**: Optional AI-powered correction of transcribed text for improved accuracy

## Requirements

- Linux (tested on X11 and Wayland)
- Python 3.11+
- `ydotool` for simulating keyboard input (by pasting or typing + pasting)
- OpenAI, Deepgram, or Mistral API key for transcription (depending on provider)
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

# Mistral API key (required if model from Mistral)
TWISTT_MISTRAL_API_KEY=...
# or
MISTRAL_API_KEY=...

# Optional settings
TWISTT_HOTKEY=F9           # Single hotkey
TWISTT_HOTKEYS=F8,F9,F10   # Multiple hotkeys (comma-separated)
TWISTT_MODEL=gpt-4o-transcribe   # For OpenAI; for Deepgram use e.g. nova-2-general; for Mistral use voxtral-mini-transcribe-realtime-2602
TWISTT_LANGUAGE=en  # Leave empty or omit for auto-detect
TWISTT_SILENCE_DURATION=500  # Milliseconds of silence before ending the current segment
TWISTT_GAIN=1.0
TWISTT_MICROPHONE=Elgato Wave 3  # Optional text filter to auto-select a microphone
TWISTT_DOUBLE_TAP_WINDOW=0.5  # Time window for double-tap detection (and single-tap threshold)
TWISTT_TOGGLE_MODE=double  # Toggle activation: single (one tap) or double (double-tap)
TWISTT_KEYBOARD=keychron  # Optional text filter to auto-select matching keyboard
TWISTT_YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket  # Optional, auto-detected by default

# Output mode
TWISTT_OUTPUT_MODE=batch  # batch (default) or full
TWISTT_USE_TYPING=false  # Type ASCII characters via ydotool instead of copy/paste (slower)
TWISTT_KEYBOARD_DELAY=20  # Delay in milliseconds between keyboard actions (default: 20ms)

# Indicator text (shown at cursor position while recording/processing)
TWISTT_INDICATOR_TEXT=" (Twistting...)"  # Customize the indicator text (default: " (Twistting...)")
TWISTT_INDICATOR_TEXT_DISABLED=false  # Set to true to disable the indicator entirely

# System tray icon (shows a microphone icon in the system tray, turns red when active)
TWISTT_TRAY_ICON_DISABLED=false  # Set to true to disable the system tray icon (enabled by default)

# Audio ducking (automatically reduces system audio during recording)
TWISTT_DUCKING_DISABLED=false  # Set to true to disable audio ducking (enabled by default)
TWISTT_DUCKING_PERCENT=50  # How much to reduce system volume BY during recording (0-100, default: 50)

# Logging
TWISTT_LOG=/path/to/custom/twistt.log  # Optional, defaults to ~/.config/twistt/twistt.log

# Post-treatment settings (optional)
TWISTT_POST_TREATMENT_PROMPT="Fix grammar and punctuation"  # Can be text, file path, or multiple separated by '::'
TWISTT_POST_TREATMENT_MODEL=gpt-4o-mini  # Model for post-treatment
TWISTT_POST_TREATMENT_PROVIDER=openai  # Provider: openai, cerebras, or openrouter
# Post-treatment correct mode (apply corrections in-place with keyboard; requires batch output mode)
TWISTT_POST_TREATMENT_CORRECT=false
# Disable post-treatment entirely (ignores prompts/files)
TWISTT_POST_TREATMENT_DISABLED=false

# Provider-specific API keys (for post-treatment)
TWISTT_CEREBRAS_API_KEY=csk-...  # Required if using cerebras provider
TWISTT_OPENROUTER_API_KEY=sk-or-...  # Required if using openrouter provider
```

### Available Options

| Option                                         | Environment Variable                                | Default                       | Description                                                                                                                                                                                                             |
|------------------------------------------------|-----------------------------------------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-k, --hotkey`                                 | `TWISTT_HOTKEY` or `TWISTT_HOTKEYS`                 | F9                            | Push-to-talk key(s) (F1-F12), comma-separated for multiple                                                                                                                                                              |
| `-kb, --keyboard`                              | `TWISTT_KEYBOARD`                                   | -                             | Filter text for selecting input device(s) for hotkey detection (keyboards, mice with remapped buttons, etc.)<br/>Pass without a value to force interactive selection and ignore env defaults                             |
| `-dtw, --double-tap-window`                    | `TWISTT_DOUBLE_TAP_WINDOW`                          | 0.5                           | Time window in seconds for double-tap detection (and single-tap threshold)                                                                                                                                              |
| `-tm, --toggle-mode`                           | `TWISTT_TOGGLE_MODE`                                | double                        | Toggle activation mode: `single` (one tap) or `double` (double-tap)                                                                                                                                                    |
| `-m, --model`                                  | `TWISTT_MODEL`                                      | gpt-4o-transcribe             | Transcription model (for OpenAI, Deepgram, or Mistral)                                                                                                                                                                  |
| `-l, --language`                               | `TWISTT_LANGUAGE`                                   | Auto-detect                   | Transcription language (ISO 639-1)                                                                                                                                                                                      |
| `-sd, --silence-duration`                      | `TWISTT_SILENCE_DURATION`                           | 500                           | Silence duration in milliseconds before the transcription service ends the current segment                                                                                                                              |
| `-g, --gain`                                   | `TWISTT_GAIN`                                       | 1.0                           | Microphone amplification                                                                                                                                                                                                |
| `-mic, --microphone`                           | `TWISTT_MICROPHONE`                                 | Default input                 | Text filter or ID to select the microphone<br/>Pass without a value to force interactive selection and ignore env defaults                                                                                              |
| `-koa, --openai-api-key`                       | `TWISTT_OPENAI_API_KEY` or `OPENAI_API_KEY`         | -                             | OpenAI API key                                                                                                                                                                                                          |
| `-kdg, --deepgram-api-key`                     | `TWISTT_DEEPGRAM_API_KEY` or `DEEPGRAM_API_KEY`     | -                             | Deepgram API key                                                                                                                                                                                                        |
| `-kmi, --mistral-api-key`                      | `TWISTT_MISTRAL_API_KEY` or `MISTRAL_API_KEY`       | -                             | Mistral API key                                                                                                                                                                                                         |
| `-ys, --ydotool-socket`                        | `TWISTT_YDOTOOL_SOCKET` or `YDOTOOL_SOCKET`         | Auto-detect                   | Path to ydotool socket                                                                                                                                                                                                  |
| `-p, --post-prompt`                            | `TWISTT_POST_TREATMENT_PROMPT`                      | -                             | Post-treatment prompt (text/file). Can be specified multiple times. Within a value, use `::` to separate multiple prompts. Prefix any `-p` value with `::` to include env/config variable. Example: `-p :: -p file.txt` |
| `-pm, --post-model`                            | `TWISTT_POST_TREATMENT_MODEL`                       | gpt-4o-mini                   | Model for post-treatment                                                                                                                                                                                                |
| `-pp, --post-provider`                         | `TWISTT_POST_TREATMENT_PROVIDER`                    | openai                        | Provider for post-treatment (openai, cerebras, openrouter)                                                                                                                                                              |
| `-pc, --post-correct, -npc, --no-post-correct` | `TWISTT_POST_TREATMENT_CORRECT`                     | false                         | Apply post-treatment by correcting already-output text in-place (only in batch output mode)                                                                                                                             |
| `-np, --no-post`                               | `TWISTT_POST_TREATMENT_DISABLED`                    | false                         | Disable post-treatment regardless of prompts or files                                                                                                                                                                   |
| `-kcb, --cerebras-api-key`                     | `TWISTT_CEREBRAS_API_KEY` or `CEREBRAS_API_KEY`     | -                             | Cerebras API key                                                                                                                                                                                                        |
| `-kor, --openrouter-api-key`                   | `TWISTT_OPENROUTER_API_KEY` or `OPENROUTER_API_KEY` | -                             | OpenRouter API key                                                                                                                                                                                                      |
| `-o, --output-mode, -no, --no-output-mode`     | `TWISTT_OUTPUT_MODE`                                | batch                         | Output mode: batch (incremental), full (complete on release), or none (disabled)                                                                                                                                        |
| `-t, --use-typing, -nt, --no-use-typing`       | `TWISTT_USE_TYPING`                                 | false                         | Type ASCII characters directly (slower); clipboard still handles non-ASCII. Use `-t`/`--use-typing` to enable, `-nt`/`--no-use-typing` to disable                                                                       |
| `-kd, --keyboard-delay`                        | `TWISTT_KEYBOARD_DELAY`                             | 20                            | Delay in milliseconds between keyboard actions (typing, paste, navigation keys). Increase if you experience character ordering issues                                                                                   |
| `-it, --indicator-text`                        | `TWISTT_INDICATOR_TEXT`                             | ` (Twistting...)`             | Text shown at cursor position while recording/processing                                                                                                                                                                |
| `-ni, --no-indicator`                          | `TWISTT_INDICATOR_TEXT_DISABLED`                    | false                         | Disable the indicator text shown at cursor position while recording/processing                                                                                                                                          |
| `-nti, --no-tray-icon`                         | `TWISTT_TRAY_ICON_DISABLED`                        | false                         | Disable the system tray icon (microphone icon that turns red when active). Requires optional packages: `pystray`, `Pillow`, `PyGObject` (see System Tray Icon section)                                                           |
| `-nd, --no-ducking`                            | `TWISTT_DUCKING_DISABLED`                          | false                         | Disable audio ducking (automatic volume reduction of system audio during recording). Requires `pulsectl` package                                                                                                        |
| `-dp, --ducking-percent`                       | `TWISTT_DUCKING_PERCENT`                           | 50                            | How much to reduce system volume BY during recording (0-100). 50 means reduce to 50% of original volume                                                                                                                |
| `--log`                                        | `TWISTT_LOG`                                        | `~/.config/twistt/twistt.log` | Path to log file where transcription sessions are saved                                                                                                                                                                 |
| `--check`                                      | -                                                   | -                             | Display configuration and exit without logging anything to file. Useful for verifying settings before running.                                                                                                          |
| `--list-configs [DIR]`                         | -                                                   | -                             | List all configuration files found in `~/.config/twistt/` (or DIR if specified) with their variables and exit. API keys are masked, all values are limited to 100 characters.                                            |
| `-c, --config PATH`                            | `TWISTT_CONFIG`                                     | `~/.config/twistt/config.env` | Load configuration from file(s). Can be specified multiple times or use `::` separator. Later files override earlier ones. Prefix with `::` to include default config. Example: `-c ::fr.env` (default + modifier)      |
| `-sc, --save-config [PATH]`                    | `TWISTT_CONFIG`                                     | false                         | Persist provided command-line values to a config file (defaults to `~/.config/twistt/config.env` or `TWISTT_CONFIG` if set)                                                                                             |

Selecting a microphone sets the `PULSE_SOURCE` environment variable for Twistt only, so your system default input stays untouched. Run `./twistt.py --microphone` without a value to pick from the list even if an environment variable is set.

Use `--config` (or `TWISTT_CONFIG`) to load settings from one or more files. You can specify multiple config files either by using `-c` multiple times or by separating paths with `::` in a single argument or environment variable. Later files override values from earlier ones.

**Including the default config**: Prefix any `-c` value with `::` to include the default config (`~/.config/twistt/config.env`) as the base, allowing you to use modifier files that only specify what differs. For example, `-c ::fr.env` combines the default config with `fr.env` (where `fr.env` might only set `TWISTT_LANGUAGE=fr`). Without the `::` prefix, `-c` replaces the default config entirely.

If you provide a relative path that doesn't exist in the current directory, and a file with that name (plus `.env`) exists in `~/.config/twistt/`, it will be used automatically. For example, `--config work` will use `~/.config/twistt/work.env` if `work` doesn't exist locally. Use `--save-config` to capture only the options you explicitly pass on the command line; existing keys in the config file are preserved. Provide a path (or set `TWISTT_CONFIG`) to control which file gets written. `TWISTT_CONFIG` is read only from the process environment—do not place it in `.env` files or `config.env`.

### Config Inheritance and Multiple Config Files

Twistt supports two complementary ways to combine configuration files:

**1. Multiple config files via `-c` or `TWISTT_CONFIG`:**

You can specify multiple config files that are loaded in sequence, with later files overriding values from earlier ones:

```bash
# Load multiple configs via command line
./twistt.py -c base.env -c project.env -c local.env

# Or using :: separator
./twistt.py -c "base.env::project.env::local.env"

# Or via environment variable
TWISTT_CONFIG="base.env::project.env" ./twistt.py
```

In these examples:
- `base.env` is loaded first (lowest priority)
- `project.env` overrides values from `base.env`
- `local.env` overrides values from both `base.env` and `project.env` (highest priority)

**Using modifier files with the default config:**

Create small config files that only specify what differs from your default configuration, then use the `::` prefix to combine them:

```bash
# Create a French language modifier
echo "TWISTT_LANGUAGE=fr" > ~/.config/twistt/fr.env

# Create a high-gain modifier for quiet microphones
echo "TWISTT_GAIN=3.0" > ~/.config/twistt/loud.env

# Use modifiers with default config
./twistt.py -c ::fr.env  # French language + all default settings
./twistt.py -c ::loud.env  # High gain + all default settings
./twistt.py -c ::fr.env -c ::loud.env  # French + high gain + all defaults
```

This is particularly useful when you have a well-configured default setup and only want to temporarily change one or two settings.

**2. Parent config inheritance via `TWISTT_PARENT_CONFIG`:**

Individual config files can define `TWISTT_PARENT_CONFIG` to inherit from another config file. Values in the child file take precedence over the parent:

```bash
# ~/.config/twistt/config.env - shared settings
TWISTT_OPENAI_API_KEY=sk-...
...

# ~/.config/twistt/gpt.env - inherits base and use open ai model without typing mode (because not recommended)
TWISTT_PARENT_CONFIG=config.env
TWISTT_MODEL=gpt-4o-transcribe
TWISTT_USE_TYPING=false

# ~/.config/twistt/nova.env - inherits base and use nova-2 model with typing mode (because it fits well)
TWISTT_PARENT_CONFIG=config.env
TWISTT_MODEL=nova-2
TWISTT_USE_TYPING=true
```

In those examples, `nova.env` and `gpt.env` being in `~/.config/twistt/`, they can be used like that: `twistt.py --config nova` or `./twistt.py --config gpt` (without passing the full path and the `.env` extension to the config argument)

Parent paths can be relative (resolved from the child config's directory) or absolute. Circular references are detected and will cause an error.

**Combining both approaches:**

You can mix multiple config files and parent inheritance. For example:

```bash
# Load base config with its parent, then override with local settings
./twistt.py -c gpt.env -c local.env
```

This will:
1. Load `config.env` (parent of `gpt.env`)
2. Load `gpt.env` (overrides `config.env`)
3. Load `local.env` (overrides both `config.env` and `gpt.env`)

### Listing Available Configurations

Use `--list-configs` to see all configuration files in `~/.config/twistt/` and their variables:

```bash
./twistt.py --list-configs

# Or list configs from a specific directory
./twistt.py --list-configs /path/to/configs
```

This displays:
- All `.env` files in the config directory, sorted alphabetically
- For each file:
  - Filename with parent config shown in parentheses if defined
  - All variables in alphabetical order
  - API keys are masked (only first 3 characters + "...")
  - All values are limited to 100 characters with newlines replaced by spaces
  - "..." is appended only if the value exceeds 100 characters

Example output:
```
Configuration files found in: /home/user/.config/twistt

config.env
  TWISTT_HOTKEY = F8,F9
  TWISTT_LANGUAGE = fr
  TWISTT_OPENAI_API_KEY = sk-...
  TWISTT_POST_TREATMENT_PROMPT = Fix grammar and punctuation. Remove filler words like "um" and "uh". Keep the conversational...

fr.env
  TWISTT_LANGUAGE = fr

gpt.env (parent config: ~/.config/twistt/config.env)
  TWISTT_MODEL = gpt-4o-transcribe
  TWISTT_USE_TYPING = false
```

This is useful for:
- Discovering what config files you have
- Understanding config inheritance relationships
- Verifying variable values without opening files
- Security: checking API keys are set without revealing full values

### Logging

All transcription sessions are automatically logged to a file. By default, logs are saved to `~/.config/twistt/twistt.log`. You can customize the log file location using:

- Command-line argument: `--log /path/to/logfile.log`
- Environment variable: `TWISTT_LOG=/path/to/logfile.log`

The log file contains:
- Configuration panel (displayed at startup)
- Completed transcription sessions with timestamps
- Both raw transcription and post-treatment results (if enabled)

Note: Live updates during recording are **not** logged, only finalized sessions are saved.

To disable logging, point the log file to `/dev/null`:
```bash
./twistt.py --log /dev/null
```

### Post-Treatment Prompt

The `--post-prompt` argument and `TWISTT_POST_TREATMENT_PROMPT` environment variable support multiple prompts that can be combined.

**Multiple prompts with `::` separator:**

You can specify multiple prompts separated by `::`. Each part is resolved independently as either a file (if it exists) or literal text, then all parts are combined with double newlines between them:

```bash
# Environment variable examples
TWISTT_POST_TREATMENT_PROMPT="prompt1.txt::Fix grammar::prompt2.txt"
TWISTT_POST_TREATMENT_PROMPT="corrections.txt::Make it formal"
```

**File resolution for each part:**
- Absolute paths are checked directly
- Relative paths are searched in: current directory → script directory → `~/.config/twistt/`
- Shell expansion such as `~` is supported
- When the filename has no extension, Twistt tries with no extension, then `.txt` and `.prompt` variants
- If a file is found, its content is used; otherwise the value is treated as direct text
- Empty files are rejected

**Using `-p` / `--post-prompt` argument:**

The `-p` flag can be specified multiple times and supports two modes:

1. **Replace mode** (default) - ignores environment variable:
   ```bash
   ./twistt.py -p "Fix grammar"                    # Uses only this prompt
   ./twistt.py -p "prompt1.txt::Make it formal"    # Combines these two
   ./twistt.py -p file1.txt -p "Fix grammar"       # Multiple -p: file1.txt + literal text
   ```

2. **Append mode** (prefix ANY `-p` value with `::`) - includes environment variable:
   ```bash
   # If TWISTT_POST_TREATMENT_PROMPT="base.txt"
   ./twistt.py -p "::"                       # Uses only base.txt (env var)
   ./twistt.py -p "::extra.txt"              # Combines: base.txt + extra.txt
   ./twistt.py -p :: -p file1.txt            # Combines: base.txt + file1.txt
   ./twistt.py -p file1.txt -p "::file2.txt" # Combines: base.txt + file1.txt + file2.txt
   ```

**Key points:**
- You can use `-p` multiple times: `-p file1.txt -p file2.txt -p "Fix grammar"`
- If ANY `-p` value starts with `::`, the environment variable is included **first**
- Order: env var (if requested) → all `-p` values in order (with `::` prefix removed)
- Each `-p` value can contain `::` separators for multiple prompts within one argument

**Examples:**

```bash
# Single prompts
./twistt.py -p translate            # Uses translate.txt if exists, else literal text
./twistt.py -p "Fix grammar"        # Direct text
./twistt.py -p ./prompts/formal.txt # Explicit file path

# Multiple prompts via environment
TWISTT_POST_TREATMENT_PROMPT="base.txt::Fix grammar"
./twistt.py  # Uses both prompts combined

# Multiple -p arguments
./twistt.py -p file1.txt -p "Fix grammar" -p file2.txt

# Mixing :: separator and multiple -p
./twistt.py -p "prompt1.txt::Make formal" -p prompt2.txt

# Including environment variable
TWISTT_POST_TREATMENT_PROMPT="base.txt"
./twistt.py -p ::               # Uses only base.txt
./twistt.py -p "::extra.txt"    # Uses base.txt + extra.txt
./twistt.py -p :: -p custom.txt # Uses base.txt + custom.txt
./twistt.py -p file1.txt -p "::file2.txt"  # Uses base.txt + file1.txt + file2.txt

# Disable post-treatment
./twistt.py --no-post
```

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
./twistt.py --post-prompt instructions.txt

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
TWISTT_DEEPGRAM_API_KEY=dg_xxx ./twistt.py --model nova-2-general --language fr

# Use Mistral/Voxtral as provider
TWISTT_MISTRAL_API_KEY=xxx ./twistt.py --model voxtral-mini-transcribe-realtime-2602

# Save your preferred options for next time
./twistt.py --language fr --gain 2.0 --microphone "Elgato Wave 3" --save-config

# Save to a custom config file
./twistt.py --language fr --gain 2.0 --save-config ~/.config/twistt/presets/french.env

# Load a custom preset
./twistt.py --config ~/.config/twistt/french.env
./twistt.py --config french  # equivalent to the one above
./twistt.py --config /path/to/gaming.env

# Load multiple config files (later files override earlier ones)
./twistt.py --config base.env --config local.env
./twistt.py -c "base.env::project.env::local.env"

# Use modifier files with default config (:: prefix includes default)
./twistt.py -c ::fr.env  # Combines default config + fr.env modifier
./twistt.py -c :: -c local.env  # Combines default config + local.env
./twistt.py -c ::  # Uses only default config explicitly

# Specify a custom log file
./twistt.py --log /tmp/twistt-debug.log

# Disable logging (output to /dev/null)
./twistt.py --log /dev/null

# Check configuration without starting (useful to verify settings)
./twistt.py --check
./twistt.py --config french --check  # Verify a specific config

# List all available config files and their variables
./twistt.py --list-configs
./twistt.py --list-configs /path/to/configs  # List from custom directory
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

#### Toggle Mode (Single-Tap or Double-Tap)

Toggle mode can be activated with either a single tap or a double tap, depending on the `--toggle-mode` setting:

- **Single-tap mode** (`--toggle-mode single`): A quick tap (shorter than `--double-tap-window`) activates toggle mode. A longer press works as push-to-talk.
- **Double-tap mode** (`--toggle-mode double`, default): Press-release-press the same hotkey quickly (within `--double-tap-window`) to activate toggle mode.

Once in toggle mode:

1. **Speak freely**: Recording continues without holding any key
2. **Press to stop**: Press the same hotkey once to stop recording (only the hotkey that started toggle mode can stop it)
3. **Auto-output**: Text is automatically output at cursor position

The transcription appears where the cursor is located.

An indicator ("(Twistting...)" text by default) is shown at the cursor position when recording is active, or text is being output or post-treatment is running. The indicator text can be customized via `TWISTT_INDICATOR_TEXT` or disabled entirely via `TWISTT_INDICATOR_TEXT_DISABLED=true`.

### System Tray Icon

A system tray icon (microphone) is displayed when Twistt is running. It stays grey when idle and turns red when recording, transcribing, or post-processing. The tray icon is enabled by default and can be disabled via `TWISTT_TRAY_ICON_DISABLED=true` or `--no-tray-icon`.

The tray icon requires optional Python packages (`pystray`, `Pillow`, `PyGObject`) which in turn need system libraries to build. Twistt **auto-installs** these packages on first run (via `uv pip`) into `~/.local/share/twistt/optional-deps/`. This requires `uv` to be available on the system (which is the case when running with `uv run`). If the system libraries are missing, the install fails silently and the tray icon is skipped without affecting the rest of the application.

To enable the tray icon, install the required system libraries before running Twistt:

```bash
# Debian/Ubuntu
sudo apt install libgirepository1.0-dev libcairo2-dev pkg-config python3-dev gir1.2-ayatanaappindicator3-0.1
```

Then just run Twistt normally — the Python packages will be installed automatically on first launch:

```bash
uv run ./twistt.py
```

### KDE Plasma Widget

For KDE Plasma users, Twistt provides a standalone panel widget (plasmoid) as an alternative to the system tray icon. The widget displays the same microphone icon with the same color-coded states, but can be placed **anywhere** in your panel — independently of the system tray area.

**States:**
- **Cyan** (static): idle
- **Orange** (pulsing): recording
- **Green** (pulsing): transcribing
- **Violet** (pulsing): post-processing
- **Grey** (static): application not running

The widget communicates with Twistt via a lightweight state file (`~/.local/share/twistt/plasma-widget-state`), which is written automatically by Twistt and cleaned up on exit. No extra Python dependencies are needed — the state file is always written, whether the pystray-based tray icon is enabled or not.

**Installation (requires KDE Plasma 6):**

```bash
kpackagetool6 --type Plasma/Applet --install kde-widget/com.github.twidi.twistt-indicator/
```

Then right-click your panel → *Add Widgets* → search for **Twistt Indicator** → drag it to your panel.

**Upgrade after update:**

```bash
kpackagetool6 --type Plasma/Applet --upgrade kde-widget/com.github.twidi.twistt-indicator/
# Restart Plasma shell to reload the widget
kquitapp6 plasmashell && kstart plasmashell
```

**Uninstall:**

```bash
kpackagetool6 --type Plasma/Applet --remove com.github.twidi.twistt-indicator
```

The widget and the system tray icon can be used simultaneously, or independently. If you only use the Plasma widget, you can disable the tray icon with `--no-tray-icon`.

### Audio Ducking

Audio ducking automatically reduces the volume of all system audio outputs (music, videos, notifications, etc.) while recording, to prevent them from being picked up by the microphone and interfering with transcription. Volume is restored as soon as recording stops (key released or toggle off).

Ducking is **enabled by default** with a 50% reduction (volume reduced to 50% of original). It uses PulseAudio/PipeWire via the `pulsectl` library, which is included as a dependency.

Configuration:
- Disable ducking: `TWISTT_DUCKING_DISABLED=true` or `--no-ducking` / `-nd`
- Adjust reduction: `TWISTT_DUCKING_PERCENT=50` or `--ducking-percent 50` / `-dp 50` (50 means reduce to 50% of original volume)

If the user manually changes the system volume during ducking, the original volume (before ducking) will be restored when recording ends.

### Output Modes

Twistt supports three output modes that control when text is processed and output:

- **batch mode** (default): Text is processed and can be output incrementally as you speak. Each pause triggers processing of that segment. With post-treatment enabled, each segment maintains context from previous segments.

- **full mode**: All text is accumulated while you hold the key and only processed/output when you release it. With post-treatment, the entire text is processed at once without maintaining context between sessions. This mode is useful when you want to speak a complete thought before any processing occurs.

- **none**: Twistt skips all output entirely. Transcription and post-treatment still run (just like batch mode), but nothing is pasted or typed at the cursor position. Use when you only want live feedback in the terminal or plan to copy results manually later.

### Tips

- **Shift mode**: Press Shift at any time while recording to use Ctrl+Shift+V instead of Ctrl+V to paste (useful for terminals). Shift can be pressed:
  - When starting recording (together with the hotkey)
  - At any moment while holding the hotkey
  - The earliest Shift press is remembered for the entire recording session
- **Alt to toggle post-treatment**: Press Alt at any time while recording to toggle post-treatment on/off for the current session. This is useful when you have post-treatment configured but want to temporarily disable it for certain inputs (or the reverse).
- **Multiple sentences**: Keep holding the key to transcribe continuously
- **Pause support**: Brief pauses are handled automatically
- **Live feedback**: Watch the terminal to see transcription as it processes
- **Output mode choice**: Use `--output-mode full` when you want to complete your entire thought before processing, or `--no-output-mode` to disable output entirely
- **Post-treatment**: Enable for improved accuracy, especially useful for:
  - Fixing punctuation and capitalization
  - Correcting common speech-to-text errors
  - Adapting text style (formal, informal, technical)
  - Language-specific corrections

## Input Device Detection

The script automatically detects all input devices capable of emitting the configured hotkey (keyboards, mice with remapped buttons, macropads, etc.). All matching devices are monitored simultaneously, so a hotkey press is detected regardless of which device it comes from. Virtual input devices (ydotool, uinput, etc.) are automatically filtered out. Set `--keyboard "partial name"` or `TWISTT_KEYBOARD=partial name` to restrict to devices matching the filter. Pass `--keyboard` with no value to display an interactive selection menu.

## Post-Treatment (Optional)

Post-treatment uses AI to improve transcription accuracy by correcting errors, fixing punctuation, and applying custom transformations. It's activated automatically when you provide a prompt.

### Supported Providers

#### Transcription

You can choose between different AI providers for transcription:

- **OpenAI**: Uses OpenAI's GPT transcribe models (`gpt-4o-transcribe` (default), `gpt-4o-mini-transcribe`). Better to not use `--use-typing`.
- **Deepgram**: Uses Deepgram's Nova models (`nova-2`, `nova-3`). Really real time but more expensive. Great with `--use-typing`.
- **Mistral**: Uses Mistral's Voxtral model (`voxtral-mini-transcribe-realtime-2602`). No server-side VAD (silence detection is handled client-side).

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
./twistt.py --post-prompt corrections.txt
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
./twistt.py --post-prompt complex_rules.txt --post-model gpt-4o

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

- API keys are sent only to their respective provider's servers (OpenAI, Deepgram, or Mistral)
- Audio is processed in real-time and not stored locally
- Transcriptions are only kept in memory during the session

## Ideas

We maintain a curated list of potential enhancements in IDEAS.md. If you have suggestions or want to pick something up, check it out and open an issue or PR.

## Author

Stephane "Twidi" Angel, with the help of @claude and @codex

## License

MIT License - See LICENSE file for details
