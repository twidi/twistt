#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "sounddevice",
#     "websockets",
#     "pyperclipfix",
#     "evdev",
#     "python-dotenv",
#     "platformdirs",
# ]
# ///

import os
import sys
import json
import asyncio
import base64
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
import pyperclipfix as pyperclip
import evdev
from evdev import ecodes
from dotenv import load_dotenv
from platformdirs import user_config_dir

# Load .env from script's directory first
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Then load and override with user config file
config_dir = Path(user_config_dir("twistt", ensure_exists=False))
user_config_path = config_dir / "config.env"
if user_config_path.exists():
    load_dotenv(dotenv_path=user_config_path, override=True)

ENV_PREFIX = "TWISTT_"
RT_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SR = 24_000
BLOCK_MS = 40
BLOCK_SIZE = int(SR * BLOCK_MS / 1000)

EV_SPEECH_STARTED = "input_audio_buffer.speech_started"
EV_DELTA = "conversation.item.input_audio_transcription.delta"
EV_DONE = "conversation.item.input_audio_transcription.completed"

# F key mapping for evdev
F_KEY_CODES = {
    'f1': ecodes.KEY_F1,
    'f2': ecodes.KEY_F2,
    'f3': ecodes.KEY_F3,
    'f4': ecodes.KEY_F4,
    'f5': ecodes.KEY_F5,
    'f6': ecodes.KEY_F6,
    'f7': ecodes.KEY_F7,
    'f8': ecodes.KEY_F8,
    'f9': ecodes.KEY_F9,
    'f10': ecodes.KEY_F10,
    'f11': ecodes.KEY_F11,
    'f12': ecodes.KEY_F12,
}

class AudioTranscriber:
    def __init__(self, openai_api_key, language, model, gain, keyboard):
        self.openai_api_key = openai_api_key
        self.language = language
        self.model = model
        self.gain = gain
        self.keyboard = keyboard
        self.recording = False
        self.stream_task = None
        self.current_transcription = []
        self.speech_started = False

        self.headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        session_config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "input_audio_transcription": {
                    "model": self.model,
                },
                "input_audio_noise_reduction": {
                    "type": "near_field"  # "near_field" or "far_field" or null
                },
                "include": [
                    "item.input_audio_transcription.logprobs",
                ],
            },
        }
        
        # Only add language if specified
        if self.language:
            session_config["session"]["input_audio_transcription"]["language"] = self.language
        
        self.session_json = json.dumps(session_config)

    async def stream_mic(self):
        print(f"\n--- {datetime.now()} ---")
        q = asyncio.Queue(maxsize=100)
        loop = asyncio.get_running_loop()
        self.recording = True
        self.speech_started = False

        def cb(indata, frames, timeinfo, status):
            if self.recording or self.speech_started:
                try:
                    if self.gain != 1.0:
                        audio_data = np.frombuffer(indata, dtype=np.int16)
                        amplified = np.clip(audio_data * self.gain, -32768, 32767)
                        audio_bytes = amplified.astype(np.int16).tobytes()
                    else:
                        audio_bytes = bytes(indata)
                    
                    loop.call_soon_threadsafe(q.put_nowait, audio_bytes)
                except:
                    pass

        with sd.RawInputStream(samplerate=SR, blocksize=BLOCK_SIZE,
                               dtype="int16", channels=1, callback=cb):
            async with websockets.connect(RT_URL, additional_headers=self.headers, max_size=None) as ws:
                await ws.send(self.session_json)

                self.current_transcription = []

                async def sender():
                    while self.recording or self.speech_started:
                        try:
                            try:
                                raw = await asyncio.wait_for(q.get(), timeout=0.1)
                            except asyncio.TimeoutError:
                                continue
                            try:
                                b64 = base64.b64encode(raw).decode("ascii")
                                await ws.send(json.dumps({
                                    "type": "input_audio_buffer.append",
                                    "audio": b64
                                }))
                            except Exception as e:
                                print(f"Error in sender: {e}", file=sys.stderr)
                                break

                        except asyncio.CancelledError:
                            break

                async def receiver():
                    while self.recording or self.speech_started:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        except asyncio.CancelledError:
                            break

                        try:
                            ev = json.loads(raw)
                            ev_type = ev.get("type", "")

                            if ev_type == EV_SPEECH_STARTED:
                                self.speech_started = True

                            elif ev_type == EV_DELTA:
                                d = ev.get("delta")
                                if d:
                                    self.current_transcription.append(d)
                                    self.speech_started = True
                                    print(d, end="", flush=True)

                            elif ev_type == EV_DONE:
                                self.speech_started = False

                                if self.current_transcription:
                                    full_text = "".join(self.current_transcription)
                                    if self.recording:
                                        full_text += " "
                                    self.copy(full_text)
                                    self.paste()
                                    self.current_transcription.clear()

                                if not self.recording:
                                    break

                                print(" ", end="", flush=True)

                        except Exception as e:
                            print(f"Error in receiver: {e}", file=sys.stderr)
                            break

                        except asyncio.CancelledError:
                            break

                await asyncio.gather(sender(), receiver(), return_exceptions=True)


    def copy(self, text):
        try:
            pyperclip.copy(text)
        except Exception as e:
            print(f"Error copying to clipboard: {e}", file=sys.stderr)

    def paste(self):
        try:
            active_keys = self.keyboard.active_keys()
            shift_pressed = ecodes.KEY_LEFTSHIFT in active_keys or ecodes.KEY_RIGHTSHIFT in active_keys
            if shift_pressed:
                subprocess.run(['ydotool', 'key', '29:1', '42:1', '47:1', '29:0', '42:0', '47:0'], check=True)
            else:
                subprocess.run(['ydotool', 'key', '29:1', '47:1', '29:0', '47:0'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ydotool error: {e}", file=sys.stderr)
            print("Text is in clipboard, use Ctrl+V to paste.", file=sys.stderr)

    async def start_recording(self):
        if not self.recording:
            self.stream_task = asyncio.create_task(self.stream_mic())

    async def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.stream_task:
                await self.stream_task

def parse_hotkey_evdev(hotkey_str):
    hotkey_str = hotkey_str.lower().strip()
    if hotkey_str in F_KEY_CODES:
        return F_KEY_CODES[hotkey_str]
    else:
        raise ValueError(f"Unsupported key: {hotkey_str}. Use F1-F12")

def find_keyboard():
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    all_keyboards = []
    
    for device in devices:
        capabilities = device.capabilities(verbose=False)
        
        if ecodes.EV_KEY not in capabilities:
            continue
        keys = capabilities[ecodes.EV_KEY]

        if any(virt in device.name.lower() for virt in ['virtual', 'dummy', 'uinput', 'ydotool']):
            continue

        if any(k in [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE] for k in keys):
            continue
        if ecodes.EV_REL in capabilities:
            continue

        if ecodes.KEY_A not in keys or ecodes.KEY_Z not in keys:
            continue

        if ecodes.KEY_SPACE not in keys and ecodes.KEY_ENTER not in keys:
            continue

        if ecodes.KEY_F1 not in keys or ecodes.KEY_F12 not in keys:
            continue

        all_keyboards.append(device)
    
    if len(all_keyboards) == 1:
        device = all_keyboards[0]
    elif len(all_keyboards) > 1:
        print("\nMultiple physical keyboards found:")
        for i, device in enumerate(all_keyboards):
            print(f"  {i}: {device.path} - {device.name}")
        idx = int(input("Select your keyboard: "))
        device = all_keyboards[idx]
    else:
        print("\nNo physical keyboard detected automatically.")
        print("Available devices:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device.path} - {device.name}")
        idx = int(input("Select your keyboard manually: "))
        device = devices[idx]

    return device


async def keyboard_listener(device, hotkey_code, transcriber):
    hotkey_pressed = False

    async for event in device.async_read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = evdev.categorize(event)
            
            if key_event.scancode == hotkey_code:
                if key_event.keystate == evdev.KeyEvent.key_down and not hotkey_pressed:
                    hotkey_pressed = True
                    await transcriber.start_recording()
                    
                elif key_event.keystate == evdev.KeyEvent.key_up and hotkey_pressed:
                    hotkey_pressed = False
                    await transcriber.stop_recording()


async def main():
    epilog = """Configuration files:
  The script loads configuration from two .env files (if they exist):
  1. .env file in the script's directory
  2. ~/.config/twistt/config.env (overrides values from #1)
  
  Environment variables (in order of priority):
  - Command-line arguments (highest priority)
  - User config file (~/.config/twistt/config.env)
  - Local .env file (script directory)
  - System environment variables (lowest priority)
  
  Each option can be set via environment variable using the TWISTT_ prefix.
  API key can also use OPENAI_API_KEY (without prefix).
  """
    
    parser = argparse.ArgumentParser(
        description="Push to talk transcription via OpenAI",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Get values from environment variables or use defaults
    default_hotkey = os.getenv(f"{ENV_PREFIX}HOTKEY", "F9")
    default_model = os.getenv(f"{ENV_PREFIX}MODEL", "gpt-4o-transcribe")
    default_language = os.getenv(f"{ENV_PREFIX}LANGUAGE", None)
    default_gain = float(os.getenv(f"{ENV_PREFIX}GAIN", "1.0"))
    # API key priority: TWISTT_OPENAI_API_KEY > OPENAI_API_KEY
    default_api_key = os.getenv(f"{ENV_PREFIX}OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    parser.add_argument("--hotkey", default=default_hotkey, 
                       help="Push-to-talk key, F1-F12 (env: TWISTT_HOTKEY)")
    parser.add_argument("--model", default=default_model,
                       choices=["gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
                       help="OpenAI model to use for transcription (env: TWISTT_MODEL)")
    parser.add_argument("--language", default=default_language, 
                       help="Transcription language, leave empty for auto-detect (env: TWISTT_LANGUAGE)")
    parser.add_argument("--gain", type=float, default=default_gain, 
                       help="Microphone amplification factor, 1.0=normal, 2.0=double (env: TWISTT_GAIN)")
    parser.add_argument("--api-key", default=default_api_key,
                       help="OpenAI API key (env: TWISTT_OPENAI_API_KEY or OPENAI_API_KEY)")
    args = parser.parse_args()

    # Check API key
    if not args.api_key:
        print("ERROR: OpenAI API key is not defined", file=sys.stderr)
        print("Please set OPENAI_API_KEY or TWISTT_OPENAI_API_KEY environment variable (can optionally be in a .env file)", file=sys.stderr)
        print("Or pass it via --api-key argument", file=sys.stderr)
        return

    try:
        hotkey_code = parse_hotkey_evdev(args.hotkey)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return

    # Find keyboard device
    try:
        keyboard = find_keyboard()
    except Exception as e:
        print(f"ERROR: Unable to find keyboard: {e}", file=sys.stderr)
        return

    print(f"Transcription model: {args.model}")
    if args.language:
        print(f"Language: {args.language}")
    else:
        print(f"Language: Auto-detect")
    if args.gain != 1.0:
        print(f"Audio gain: {args.gain}x")
    print(f"Using key '{args.hotkey.upper()}' for push-to-talk (Listening on {keyboard.name}).")
    print("Text will be pasted by simulating Ctrl+V (or Ctrl+Shift+V if Shift is held at the time).")
    print("Press Ctrl+C to stop the script.")

    # Create transcriber and start listening
    transcriber = AudioTranscriber(openai_api_key=args.api_key, language=args.language, model=args.model, gain=args.gain, keyboard=keyboard)
    
    try:
        await keyboard_listener(keyboard, hotkey_code, transcriber)
    except KeyboardInterrupt:
        print("\nStopping program.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
