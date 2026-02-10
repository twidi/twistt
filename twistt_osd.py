#!/usr/bin/env python3
"""
Twistt OSD - Live transcription overlay for Wayland.

A floating, non-focusable overlay window that shows:
- Real-time speech transcript text with glow effects
- Audio spectrum analyzer with mirrored gradient bars
- State indicators (recording, transcribing, post-processing)

Uses GTK4 + gtk4-layer-shell for Wayland layer surface rendering.
Communicates with twistt.py via Unix socket (length-prefixed JSON).
Controlled via signals: SIGUSR1=show, SIGUSR2=hide, SIGTERM=stop.

Can be run standalone for testing:
    python3 osd_twistt.py

Or as a daemon (started by twistt.py):
    python3 osd_twistt.py --daemon
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

try:
    gi.require_version("Gtk4LayerShell", "1.0")
    LAYER_SHELL_AVAILABLE = True
except ValueError:
    LAYER_SHELL_AVAILABLE = False

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")

from gi.repository import Gdk, GLib, Gtk, Pango, PangoCairo
import cairo

if LAYER_SHELL_AVAILABLE:
    from gi.repository import Gtk4LayerShell


# ── Paths ──────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "twistt"
SOCKET_PATH = DATA_DIR / "osd.sock"
PID_FILE = DATA_DIR / "osd.pid"


# ── IPC Protocol ──────────────────────────────────────────────────────


class OSDProtocol:
    """Length-prefixed JSON message protocol over Unix sockets."""

    @staticmethod
    def encode_message(msg: dict) -> bytes:
        payload = json.dumps(msg, default=str).encode("utf-8")
        return struct.pack("!I", len(payload)) + payload

    @staticmethod
    def decode_messages(buffer: bytes) -> tuple[list[dict], bytes]:
        messages = []
        offset = 0
        while offset + 4 <= len(buffer):
            (length,) = struct.unpack("!I", buffer[offset : offset + 4])
            if offset + 4 + length > len(buffer):
                break  # incomplete message
            payload = buffer[offset + 4 : offset + 4 + length]
            try:
                messages.append(json.loads(payload.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # skip malformed
            offset += 4 + length
        return messages, buffer[offset:]


# ── Audio Monitor ─────────────────────────────────────────────────────


class AudioMonitor:
    """Real-time microphone audio monitor for spectrum visualization.

    Uses sounddevice to capture audio and provides peak levels and raw
    samples, thread-safely.
    """

    def __init__(self, samplerate: int = 44100, blocksize: int = 1024):
        if sd is None:
            raise ImportError("sounddevice is required for audio monitoring")
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.stream = None
        self.running = False
        self.peak_level = 0.0
        self.samples = np.zeros(blocksize)
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0].copy()
        peak = float(np.max(np.abs(samples)))
        with self._lock:
            self.peak_level = peak
            self.samples = samples

    def start(self):
        if self.running:
            return
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=self._audio_callback,
            )
            self.stream.start()
            self.running = True
        except Exception as e:
            raise RuntimeError(f"Failed to start audio monitoring: {e}")

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
        with self._lock:
            self.peak_level = 0.0
            self.samples = np.zeros(self.blocksize)

    def get_level(self) -> float:
        with self._lock:
            return self.peak_level

    def get_samples(self) -> np.ndarray:
        with self._lock:
            return self.samples.copy()


# ── Cairo Renderer ────────────────────────────────────────────────────


class OSDRenderer:
    """Cairo + PangoCairo renderer for the OSD overlay.

    Draws a glass-morphism background, mirrored gradient spectrum bars,
    glowing transcript text, and pulsing state indicators.
    """

    NUM_BARS = 60
    BAR_GAP = 2
    DB_FLOOR = -60.0  # dB floor for absolute scaling (silence threshold)
    LEVEL_INDICATOR_WIDTH = 30  # px reserved for the dB level indicator

    def __init__(self):
        self._bar_heights = np.zeros(self.NUM_BARS)
        self._peak_heights = np.zeros(self.NUM_BARS)
        self._peak_velocities = np.zeros(self.NUM_BARS)
        self._peak_db: float = self.DB_FLOOR
        self._peak_db_smooth: float = self.DB_FLOOR

    # ── main entry ─────────────────────────────────────────────────

    def draw(
        self,
        cr: cairo.Context,
        width: float,
        height: float,
        level: float,
        samples: np.ndarray | None,
        text_state: dict,
    ):
        padding = 14

        # 1. Background
        self._draw_background(cr, width, height)

        # 2. Spectrum (top 30%)
        spectrum_h = height * 0.28
        self._draw_spectrum(
            cr, padding, 10, width - 2 * padding, spectrum_h, samples, level
        )

        # 3. Separator
        sep_y = spectrum_h + 16
        self._draw_separator(cr, padding, sep_y, width - 2 * padding)

        # 4. Text area
        text_y = sep_y + 6
        text_h = height - text_y - 28
        self._draw_text_area(cr, padding, text_y, width - 2 * padding, text_h, text_state)

        # 5. State indicator
        indicator_y = height - 16
        self._draw_state_indicator(cr, padding, indicator_y, text_state)

    # ── background ─────────────────────────────────────────────────

    def _draw_background(self, cr: cairo.Context, w: float, h: float):
        radius = 16
        self._rounded_rect(cr, 0, 0, w, h, radius)

        # Dark semi-transparent fill
        gradient = cairo.LinearGradient(0, 0, 0, h)
        gradient.add_color_stop_rgba(0, 0.04, 0.04, 0.08, 0.93)
        gradient.add_color_stop_rgba(1, 0.07, 0.07, 0.12, 0.89)
        cr.set_source(gradient)
        cr.fill_preserve()

        # Top glass highlight
        highlight = cairo.LinearGradient(0, 0, 0, h * 0.25)
        highlight.add_color_stop_rgba(0, 1.0, 1.0, 1.0, 0.05)
        highlight.add_color_stop_rgba(1, 1.0, 1.0, 1.0, 0.0)
        cr.set_source(highlight)
        cr.fill_preserve()

        # Gradient border
        border = cairo.LinearGradient(0, 0, w, h)
        border.add_color_stop_rgba(0.0, 0.15, 0.70, 1.0, 0.25)
        border.add_color_stop_rgba(0.5, 0.10, 0.90, 0.70, 0.20)
        border.add_color_stop_rgba(1.0, 0.00, 1.00, 0.55, 0.25)
        cr.set_source(border)
        cr.set_line_width(1.2)
        cr.stroke()

    # ── spectrum ───────────────────────────────────────────────────

    def _draw_spectrum(
        self,
        cr: cairo.Context,
        x: float,
        y: float,
        w: float,
        h: float,
        samples: np.ndarray | None,
        level: float,
    ):
        num_bars = self.NUM_BARS
        bar_gap = self.BAR_GAP

        # Reserve space for the level indicator on the right
        indicator_w = self.LEVEL_INDICATOR_WIDTH
        bars_w = w - indicator_w
        bar_width = (bars_w - (num_bars - 1) * bar_gap) / num_bars
        center_y = y + h / 2 - 1.5  # nudge up to balance visual padding

        # ── Compute overall peak dB from raw samples ──
        raw_peak = float(np.max(np.abs(samples))) if samples is not None and len(samples) > 0 else 0.0
        if raw_peak > 1e-10:
            self._peak_db = 20.0 * np.log10(raw_peak)
        else:
            self._peak_db = self.DB_FLOOR
        # Smooth: fast rise, slow fall
        if self._peak_db > self._peak_db_smooth:
            self._peak_db_smooth = 0.3 * self._peak_db + 0.7 * self._peak_db_smooth
        else:
            self._peak_db_smooth = 0.95 * self._peak_db_smooth + 0.05 * self._peak_db

        # ── FFT analysis with absolute dB scaling ──
        if samples is not None and len(samples) > 0:
            windowed = samples * np.hanning(len(samples))
            fft = np.abs(np.fft.rfft(windowed))
            n_fft = len(fft)
            if n_fft > 1:
                freq_bins = np.logspace(
                    np.log10(1), np.log10(max(2, n_fft - 1)), num_bars + 1
                ).astype(int)
                freq_bins = np.clip(freq_bins, 0, n_fft - 1)
                new_heights = np.zeros(num_bars)
                for i in range(num_bars):
                    start = freq_bins[i]
                    end = max(start + 1, freq_bins[i + 1])
                    end = min(end, n_fft)
                    if start < end:
                        new_heights[i] = np.mean(fft[start:end])

                # Convert to absolute dB scale (0.0 = silence, 1.0 = full scale)
                db_floor = self.DB_FLOOR
                for i in range(num_bars):
                    if new_heights[i] > 1e-10:
                        db_val = 20.0 * np.log10(new_heights[i])
                        new_heights[i] = max(0.0, (db_val - db_floor) / -db_floor)
                    else:
                        new_heights[i] = 0.0

                # Smooth: fast rise, slow fall
                for i in range(num_bars):
                    if new_heights[i] > self._bar_heights[i]:
                        self._bar_heights[i] = (
                            0.5 * new_heights[i] + 0.5 * self._bar_heights[i]
                        )
                    else:
                        self._bar_heights[i] *= 0.88

                    # Peak hold with gravity
                    if new_heights[i] > self._peak_heights[i]:
                        self._peak_heights[i] = new_heights[i]
                        self._peak_velocities[i] = 0.0
                    else:
                        self._peak_velocities[i] += 0.004
                        self._peak_heights[i] -= self._peak_velocities[i]
                        if self._peak_heights[i] < 0:
                            self._peak_heights[i] = 0.0

        max_bar_h = (h / 2) - 15  # padding accounts for glow halos

        # ── Draw spectrum bars ──
        for i in range(num_bars):
            t = i / max(1, num_bars - 1)
            bar_h = max(1.5, self._bar_heights[i] * max_bar_h)
            bx = x + i * (bar_width + bar_gap)

            # Color: cyan -> teal -> green
            r = 0.05 + t * 0.0
            g = 0.65 + t * 0.35
            b = 1.0 - t * 0.5

            # Glow halo
            glow_extra = 3 + self._bar_heights[i] * 4
            cr.set_source_rgba(r, g, b, 0.08 + self._bar_heights[i] * 0.12)
            cr.rectangle(
                bx - glow_extra,
                center_y - bar_h - glow_extra,
                bar_width + 2 * glow_extra,
                2 * bar_h + 2 * glow_extra,
            )
            cr.fill()

            # Top half bar (gradient: bright tip -> dim center)
            bar_grad_top = cairo.LinearGradient(0, center_y - bar_h, 0, center_y)
            bar_grad_top.add_color_stop_rgba(0, r, g, b, 0.92)
            bar_grad_top.add_color_stop_rgba(1, r, g, b, 0.30)
            cr.set_source(bar_grad_top)
            cr.rectangle(bx, center_y - bar_h, bar_width, bar_h)
            cr.fill()

            # Bottom half bar (mirrored)
            bar_grad_bot = cairo.LinearGradient(0, center_y, 0, center_y + bar_h)
            bar_grad_bot.add_color_stop_rgba(0, r, g, b, 0.30)
            bar_grad_bot.add_color_stop_rgba(1, r, g, b, 0.92)
            cr.set_source(bar_grad_bot)
            cr.rectangle(bx, center_y, bar_width, bar_h)
            cr.fill()

            # Tip caps
            cap_h = 1.5
            cr.set_source_rgba(r, g, b, 1.0)
            cr.rectangle(bx, center_y - bar_h, bar_width, cap_h)
            cr.fill()
            cr.rectangle(bx, center_y + bar_h - cap_h, bar_width, cap_h)
            cr.fill()

            # Peak hold indicator
            peak_h = self._peak_heights[i] * max_bar_h
            if peak_h > 2:
                cr.set_source_rgba(1.0, 1.0, 1.0, 0.6)
                cr.rectangle(bx, center_y - peak_h, bar_width, 1.5)
                cr.fill()
                cr.rectangle(bx, center_y + peak_h - 1.5, bar_width, 1.5)
                cr.fill()

        # ── Draw level indicator ──
        # Center the indicator in the reserved space (with slight left margin for separation)
        margin_left = 4
        self._draw_level_indicator(cr, x + bars_w + margin_left, y, indicator_w - margin_left, h)

    # ── level indicator ─────────────────────────────────────────

    def _draw_level_indicator(
        self, cr: cairo.Context, x: float, y: float, w: float, h: float
    ):
        """Draw a vertical dB level meter with color zones and numeric label."""
        db = self._peak_db_smooth
        db_floor = self.DB_FLOOR

        # Normalized level (0.0 = floor, 1.0 = 0 dB)
        level_norm = max(0.0, min(1.0, (db - db_floor) / -db_floor))

        bar_w = 4
        bar_x = x + w - bar_w - 2  # align toward right edge of reserved space
        bar_top = y + 4
        bar_bottom = y + h - 4
        bar_h = bar_bottom - bar_top

        # Background track
        cr.set_source_rgba(0.15, 0.15, 0.20, 0.35)
        self._rounded_rect(cr, bar_x - 1, bar_top - 1, bar_w + 2, bar_h + 2, 2)
        cr.fill()

        # Filled portion (from bottom up)
        fill_h = level_norm * bar_h
        if fill_h > 1:
            fill_top = bar_bottom - fill_h

            # Clip to the fill area
            cr.save()
            cr.rectangle(bar_x, fill_top, bar_w, fill_h)
            cr.clip()

            # Multi-stop gradient: green → cyan → yellow → red (bottom to top)
            fill_grad = cairo.LinearGradient(0, bar_bottom, 0, bar_top)
            fill_grad.add_color_stop_rgba(0.0, 0.15, 0.60, 0.30, 0.90)   # green (quiet)
            fill_grad.add_color_stop_rgba(0.33, 0.10, 0.75, 0.70, 0.90)  # cyan (normal)
            fill_grad.add_color_stop_rgba(0.67, 0.10, 0.85, 0.85, 0.90)  # teal (good)
            fill_grad.add_color_stop_rgba(0.90, 0.95, 0.85, 0.20, 0.90)  # yellow (loud)
            fill_grad.add_color_stop_rgba(1.0, 1.00, 0.25, 0.20, 0.95)   # red (clipping)
            cr.set_source(fill_grad)
            cr.rectangle(bar_x, bar_top, bar_w, bar_h)
            cr.fill()

            cr.restore()

        # Current level tick mark
        tick_y = bar_bottom - level_norm * bar_h
        # Get the color at this level for the tick
        if db > -3:
            tick_color = (1.0, 0.25, 0.20, 0.95)
        elif db > -6:
            tick_color = (0.95, 0.85, 0.20, 0.90)
        elif db > -20:
            tick_color = (0.10, 0.85, 0.85, 0.90)
        elif db > -40:
            tick_color = (0.10, 0.75, 0.70, 0.80)
        else:
            tick_color = (0.15, 0.60, 0.30, 0.60)

        cr.set_source_rgba(*tick_color)
        cr.rectangle(bar_x - 2, tick_y - 1, bar_w + 4, 2)
        cr.fill()

        # dB label
        db_display = max(db, db_floor)
        db_text = f"{db_display:.0f}"

        layout = PangoCairo.create_layout(cr)
        font = Pango.FontDescription.from_string("Sans 7")
        layout.set_font_description(font)
        layout.set_text(db_text, -1)
        _, logical = layout.get_pixel_extents()

        # Position label centered below the bar
        label_x = bar_x + bar_w / 2 - logical.width / 2
        label_y = bar_bottom + 3

        # Glow behind text
        cr.set_source_rgba(*tick_color[:3], 0.15)
        cr.move_to(label_x - 0.5, label_y - 0.5)
        PangoCairo.show_layout(cr, layout)
        cr.move_to(label_x + 0.5, label_y + 0.5)
        PangoCairo.show_layout(cr, layout)

        # Main label
        cr.set_source_rgba(*tick_color[:3], 0.75)
        cr.move_to(label_x, label_y)
        PangoCairo.show_layout(cr, layout)

    # ── separator ──────────────────────────────────────────────────

    def _draw_separator(self, cr: cairo.Context, x: float, y: float, w: float):
        gradient = cairo.LinearGradient(x, y, x + w, y)
        gradient.add_color_stop_rgba(0.0, 0.15, 0.70, 1.0, 0.0)
        gradient.add_color_stop_rgba(0.2, 0.15, 0.70, 1.0, 0.25)
        gradient.add_color_stop_rgba(0.5, 0.10, 0.85, 0.65, 0.30)
        gradient.add_color_stop_rgba(0.8, 0.00, 1.00, 0.55, 0.25)
        gradient.add_color_stop_rgba(1.0, 0.00, 1.00, 0.55, 0.0)
        cr.set_source(gradient)
        cr.set_line_width(0.8)
        cr.move_to(x, y)
        cr.line_to(x + w, y)
        cr.stroke()

    # ── text area ──────────────────────────────────────────────────

    def _draw_text_area(
        self,
        cr: cairo.Context,
        x: float,
        y: float,
        w: float,
        h: float,
        text_state: dict,
    ):
        speech_text = text_state.get("speech_text", "")
        speech_final = text_state.get("speech_final", False)
        is_recording = text_state.get("is_recording", False)
        is_speaking = text_state.get("is_speaking", False)
        post_text = text_state.get("post_text", "")
        post_final = text_state.get("post_final", False)
        post_enabled = text_state.get("post_enabled", True)
        is_post_active = text_state.get("is_post_active", False)
        session_active = text_state.get("session_active", False)

        cur_y = y
        has_post = post_enabled and (post_text or is_post_active)

        # Compute available heights
        if has_post:
            speech_area_h = h * 0.48
            post_area_h = h * 0.48
        else:
            speech_area_h = h
            post_area_h = 0

        # Speech section
        label_h = self._draw_section_label(
            cr, x, cur_y, "Speech", (0.20, 0.78, 1.0)
        )
        cur_y += label_h + 2

        speech_text_h = speech_area_h - label_h - 2
        if speech_text:
            active = not speech_final and (is_recording or is_speaking)
            self._draw_glowing_text(
                cr, x, cur_y, w, speech_text_h,
                speech_text,
                glow_color=(0.20, 0.78, 1.0),
                alpha=1.0 if not speech_final else 0.55,
                show_cursor=active and not speech_final,
            )
        elif session_active:
            self._draw_placeholder(cr, x, cur_y, "Listening...")

        cur_y = y + speech_area_h

        # Post-treatment section
        if has_post:
            cur_y += 4
            label_h = self._draw_section_label(
                cr, x, cur_y, "Post-treatment", (0.60, 0.30, 0.90)
            )
            cur_y += label_h + 2

            post_text_h = post_area_h - label_h - 6
            if post_text:
                self._draw_glowing_text(
                    cr, x, cur_y, w, post_text_h,
                    post_text,
                    glow_color=(0.60, 0.30, 0.90),
                    alpha=1.0 if not post_final else 0.55,
                    show_cursor=is_post_active and not post_final,
                )
            elif is_post_active:
                self._draw_placeholder(cr, x, cur_y, "Processing...")

    def _draw_section_label(
        self,
        cr: cairo.Context,
        x: float,
        y: float,
        label: str,
        color: tuple[float, float, float],
    ) -> float:
        layout = PangoCairo.create_layout(cr)
        font = Pango.FontDescription.from_string("Sans 9")
        font.set_weight(Pango.Weight.SEMIBOLD)
        layout.set_font_description(font)
        layout.set_text(label, -1)
        _, logical = layout.get_pixel_extents()

        cr.set_source_rgba(*color, 0.65)
        cr.move_to(x, y)
        PangoCairo.show_layout(cr, layout)

        return logical.height

    def _draw_glowing_text(
        self,
        cr: cairo.Context,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        glow_color: tuple[float, float, float],
        alpha: float = 1.0,
        show_cursor: bool = False,
    ):
        layout = PangoCairo.create_layout(cr)
        font = Pango.FontDescription.from_string("Sans 13")
        layout.set_font_description(font)
        layout.set_width(int(w * Pango.SCALE))
        layout.set_wrap(Pango.WrapMode.WORD_CHAR)
        layout.set_ellipsize(Pango.EllipsizeMode.NONE)

        display_text = text.strip()
        if show_cursor:
            display_text += " "  # space for cursor

        layout.set_text(display_text, -1)

        # Measure total text height, show only last visible portion
        _, logical = layout.get_pixel_extents()
        text_total_h = logical.height

        # Clip to text area
        cr.save()
        cr.rectangle(x, y, w, h)
        cr.clip()

        # If text is taller than area, shift up to show bottom
        offset_y = 0
        if text_total_h > h:
            offset_y = text_total_h - h

        render_y = y - offset_y

        # Glow layer (4 offsets for blur effect)
        cr.set_source_rgba(*glow_color, 0.10 * alpha)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cr.move_to(x + dx, render_y + dy)
            PangoCairo.show_layout(cr, layout)

        # Main text
        cr.set_source_rgba(0.90, 0.93, 1.0, alpha)
        cr.move_to(x, render_y)
        PangoCairo.show_layout(cr, layout)

        # Blinking cursor
        if show_cursor:
            blink = (time.time() * 2.5) % 1.0 > 0.35
            if blink:
                # Get cursor position at end of text
                cursor_pos = layout.get_cursor_pos(len(display_text.encode("utf-8")) - 1)
                strong_pos = cursor_pos[0]
                cursor_x = x + strong_pos.x / Pango.SCALE + strong_pos.width / Pango.SCALE
                cursor_y_pos = render_y + strong_pos.y / Pango.SCALE
                cursor_h = strong_pos.height / Pango.SCALE

                cr.set_source_rgba(*glow_color, 0.9)
                cr.rectangle(cursor_x + 1, cursor_y_pos + 1, 2, cursor_h - 2)
                cr.fill()

        # Top fade-out: inner shadow so text fades under the section label
        if text_total_h > h:
            fade_h = 18
            fade = cairo.LinearGradient(0, y, 0, y + fade_h)
            fade.add_color_stop_rgba(0, 0.04, 0.04, 0.06, 0.95)
            fade.add_color_stop_rgba(0.6, 0.04, 0.04, 0.06, 0.40)
            fade.add_color_stop_rgba(1, 0.04, 0.04, 0.06, 0.0)
            cr.set_source(fade)
            cr.rectangle(x, y, w, fade_h)
            cr.fill()

        cr.restore()

    def _draw_placeholder(self, cr: cairo.Context, x: float, y: float, text: str):
        layout = PangoCairo.create_layout(cr)
        font = Pango.FontDescription.from_string("Sans Italic 12")
        layout.set_font_description(font)
        layout.set_text(text, -1)
        cr.set_source_rgba(0.40, 0.50, 0.60, 0.40)
        cr.move_to(x, y)
        PangoCairo.show_layout(cr, layout)

    # ── state indicator ────────────────────────────────────────────

    def _draw_state_indicator(
        self, cr: cairo.Context, x: float, y: float, text_state: dict
    ):
        is_recording = text_state.get("is_recording", False)
        is_speaking = text_state.get("is_speaking", False)
        is_post_active = text_state.get("is_post_active", False)
        session_active = text_state.get("session_active", False)

        # Build list of active states (multiple can be active simultaneously)
        active_states: list[tuple[tuple[float, float, float], str]] = []
        if is_recording:
            active_states.append(((1.0, 0.25, 0.30), "Listening"))
        if is_speaking:
            active_states.append(((0.20, 0.90, 0.40), "Transcribing"))
        if is_post_active:
            active_states.append(((0.65, 0.30, 0.90), "Post-processing"))
        if not active_states and session_active:
            active_states.append(((0.40, 0.70, 0.90), "Processing"))

        if not active_states:
            return

        t = time.time()
        pulse = 0.65 + 0.35 * math.sin(t * 4.0)
        font = Pango.FontDescription.from_string("Sans 9")
        cx = x

        for color, label in active_states:
            # Outer glow
            cr.set_source_rgba(*color, 0.15 * pulse)
            cr.arc(cx + 7, y, 9, 0, 2 * math.pi)
            cr.fill()

            # Inner dot
            cr.set_source_rgba(*color, 0.85 * pulse)
            cr.arc(cx + 7, y, 4.5, 0, 2 * math.pi)
            cr.fill()

            # Label
            layout = PangoCairo.create_layout(cr)
            layout.set_font_description(font)
            layout.set_text(label, -1)
            cr.set_source_rgba(*color, 0.70 * pulse)
            cr.move_to(cx + 18, y - 6)
            PangoCairo.show_layout(cr, layout)

            # Advance x for next indicator (dot width + label width + gap)
            label_w, _ = layout.get_pixel_size()
            cx += 18 + label_w + 16

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _rounded_rect(
        cr: cairo.Context, x: float, y: float, w: float, h: float, r: float
    ):
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()


# ── GTK4 Layer-Shell Window ───────────────────────────────────────────


class OSDWindow(Gtk.Window):
    """GTK4 layer-shell overlay window for the transcription OSD."""

    def __init__(self, width: int = 550, height: int = 220):
        super().__init__()
        self._width = width
        self._height = height
        self._renderer = OSDRenderer()
        self._audio_level = 0.0
        self._audio_samples: np.ndarray | None = None
        self._text_state: dict = {}

        self._setup_layer_shell()
        self._setup_window()
        self._setup_drawing_area()

    def _setup_layer_shell(self):
        if not LAYER_SHELL_AVAILABLE:
            return
        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_namespace(self, "twistt-osd")
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)

        # Anchor: top center
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.TOP, True)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.BOTTOM, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.LEFT, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.RIGHT, False)

        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.TOP, 40)
        Gtk4LayerShell.set_exclusive_zone(self, -1)
        Gtk4LayerShell.set_keyboard_mode(
            self, Gtk4LayerShell.KeyboardMode.NONE
        )

    def _setup_window(self):
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_default_size(self._width, self._height)
        self.add_css_class("twistt-osd-window")

    def _setup_drawing_area(self):
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_content_width(self._width)
        self.drawing_area.set_content_height(self._height)
        self.drawing_area.set_draw_func(self._on_draw)
        self.set_child(self.drawing_area)

    def _on_draw(self, area, cr, width, height):
        self._renderer.draw(
            cr, width, height,
            self._audio_level, self._audio_samples,
            self._text_state,
        )

    def update(
        self,
        level: float,
        samples: np.ndarray | None,
        text_state: dict,
    ):
        self._audio_level = level
        self._audio_samples = samples
        self._text_state = text_state
        self.drawing_area.queue_draw()


def _load_css():
    css_provider = Gtk.CssProvider()
    css_provider.load_from_string(
        """
        .twistt-osd-window {
            background-color: transparent;
        }
        """
    )
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )


# ── Main Application ─────────────────────────────────────────────────


class TranscriptionOSD:
    """GTK4 application that shows a live transcription overlay.

    Receives text data via Unix socket, captures audio independently
    for spectrum visualization, and renders with Cairo at 60fps.
    """

    def __init__(self, width: int = 550, height: int = 220, daemon: bool = False):
        self.main_loop: GLib.MainLoop | None = None
        self.window: OSDWindow | None = None
        self.audio_monitor: AudioMonitor | None = None
        self.daemon = daemon
        self.visible = False
        self._width = width
        self._height = height
        self._should_stop = False

        # Timers
        self._update_timer_id: int | None = None

        # Socket server
        self._server_socket: socket.socket | None = None
        self._client_conn: socket.socket | None = None
        self._recv_buffer = b""
        self._socket_watch_ids: list[int] = []

        # Text state (updated via IPC)
        self._text_state: dict = {
            "speech_text": "",
            "speech_final": False,
            "post_text": "",
            "post_final": False,
            "is_recording": False,
            "is_speaking": False,
            "is_post_active": False,
            "post_enabled": True,
            "session_active": False,
        }

    def run(self):
        Gtk.init()
        _load_css()

        self.window = OSDWindow(self._width, self._height)

        self._start_socket_server()
        self._initial_visibility()

        if self._should_stop:
            self._cleanup()
            return

        self.main_loop = GLib.MainLoop()
        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _initial_visibility(self):
        if self.visible:
            self._show()
        elif self.daemon:
            self.window.set_visible(False)
        else:
            self._show()

    # ── socket server ──────────────────────────────────────────────

    def _start_socket_server(self):
        # Clean up stale socket
        try:
            SOCKET_PATH.unlink(missing_ok=True)
        except Exception:
            pass

        SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.setblocking(False)
        self._server_socket.bind(str(SOCKET_PATH))
        self._server_socket.listen(1)

        chan = GLib.IOChannel.unix_new(self._server_socket.fileno())
        chan.set_encoding(None)
        chan.set_buffered(False)
        watch_id = GLib.io_add_watch(
            chan,
            GLib.PRIORITY_DEFAULT,
            GLib.IOCondition.IN,
            self._on_socket_accept,
        )
        self._socket_watch_ids.append(watch_id)

    def _on_socket_accept(self, channel, condition):
        try:
            conn, _ = self._server_socket.accept()
            conn.setblocking(False)

            # Close previous client
            if self._client_conn is not None:
                try:
                    self._client_conn.close()
                except Exception:
                    pass

            self._client_conn = conn
            self._recv_buffer = b""

            chan = GLib.IOChannel.unix_new(conn.fileno())
            chan.set_encoding(None)
            chan.set_buffered(False)
            watch_id = GLib.io_add_watch(
                chan,
                GLib.PRIORITY_DEFAULT,
                GLib.IOCondition.IN | GLib.IOCondition.HUP | GLib.IOCondition.ERR,
                self._on_socket_data,
            )
            self._socket_watch_ids.append(watch_id)
        except Exception:
            pass
        return True

    def _on_socket_data(self, channel, condition):
        if condition & (GLib.IOCondition.HUP | GLib.IOCondition.ERR):
            self._disconnect_client()
            return False

        try:
            data = self._client_conn.recv(4096)
            if not data:
                self._disconnect_client()
                return False

            self._recv_buffer += data
            messages, self._recv_buffer = OSDProtocol.decode_messages(self._recv_buffer)

            for msg in messages:
                self._handle_message(msg)

        except (ConnectionResetError, BrokenPipeError, OSError):
            self._disconnect_client()
            return False

        return True

    def _disconnect_client(self):
        if self._client_conn is not None:
            try:
                self._client_conn.close()
            except Exception:
                pass
            self._client_conn = None
        self._recv_buffer = b""

    def _handle_message(self, msg: dict):
        msg_type = msg.get("type", "")

        if msg_type == "session_start":
            self._text_state.update({
                "speech_text": "",
                "speech_final": False,
                "post_text": "",
                "post_final": False,
                "is_recording": True,
                "is_speaking": False,
                "is_post_active": False,
                "session_active": True,
            })
            self._show()

        elif msg_type == "speech_state":
            self._text_state["is_recording"] = msg.get("recording", False)
            self._text_state["is_speaking"] = msg.get("speaking", False)
            # Check if session ended
            if (
                not msg.get("recording", False)
                and not msg.get("speaking", False)
                and self._text_state.get("session_active", False)
            ):
                self._text_state["speech_final"] = True
                self._maybe_end_session()

        elif msg_type == "speech_text":
            self._text_state["speech_text"] = msg.get("text", "")
            self._text_state["speech_final"] = msg.get("final", False)
            if msg.get("final", False):
                self._maybe_end_session()

        elif msg_type == "post_state":
            self._text_state["is_post_active"] = msg.get("active", False)
            if not msg.get("active", False):
                self._maybe_end_session()

        elif msg_type == "post_text":
            self._text_state["post_text"] = msg.get("text", "")
            self._text_state["post_final"] = msg.get("final", False)
            if msg.get("final", False):
                self._maybe_end_session()

        elif msg_type == "post_enabled":
            self._text_state["post_enabled"] = msg.get("active", True)

        elif msg_type == "shutdown":
            self._hide()
            return

        # Queue redraw
        if self.window and self.visible:
            self.window.drawing_area.queue_draw()

    def _maybe_end_session(self):
        """Check if the session is complete and schedule hide."""
        st = self._text_state
        if not st.get("session_active"):
            return
        speech_done = st.get("speech_final", False) and not st.get("is_recording") and not st.get("is_speaking")
        if not speech_done:
            return
        post_enabled = st.get("post_enabled", True)
        if post_enabled:
            post_done = st.get("post_final", False) or (not st.get("is_post_active") and not st.get("post_text"))
            if not post_done:
                return
        # Session complete - hide after delay
        GLib.timeout_add(2000, self._session_end_hide)

    def _session_end_hide(self):
        """Hide the OSD after session end delay."""
        self._text_state["session_active"] = False
        self._hide()
        return False  # Don't repeat

    # ── show / hide ────────────────────────────────────────────────

    def _show(self):
        if self.visible and self.audio_monitor and self._update_timer_id:
            return

        if not self.window:
            self.visible = True
            return

        self.visible = True
        self.window.set_visible(True)

        # Start audio monitoring
        if not self.audio_monitor:
            self.audio_monitor = AudioMonitor(samplerate=44100, blocksize=1024)

        try:
            self.audio_monitor.start()
        except RuntimeError as e:
            print(f"[TWISTT-OSD] Audio monitoring failed: {e}", flush=True)
            # Continue without audio - spectrum will be flat

        # 60fps update timer
        if not self._update_timer_id:
            self._update_timer_id = GLib.timeout_add(16, self._update)

    def _hide(self):
        if not self.visible:
            return
        if not self.window:
            self.visible = False
            return

        self.visible = False
        self.window.set_visible(False)

        if self._update_timer_id:
            GLib.source_remove(self._update_timer_id)
            self._update_timer_id = None

        if self.audio_monitor:
            self.audio_monitor.stop()
            self.audio_monitor = None

    def _update(self) -> bool:
        if self.audio_monitor and self.window and self.visible:
            level = self.audio_monitor.get_level()
            samples = self.audio_monitor.get_samples()
            self.window.update(level, samples, self._text_state)
        return True  # Continue timer

    # ── lifecycle ──────────────────────────────────────────────────

    def stop(self):
        if self.main_loop:
            self.main_loop.quit()
        else:
            self._should_stop = True
            self._cleanup()

    def _cleanup(self):
        if self._update_timer_id:
            GLib.source_remove(self._update_timer_id)
            self._update_timer_id = None

        if self.audio_monitor:
            self.audio_monitor.stop()
            self.audio_monitor = None

        self._disconnect_client()

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        # Clean up socket file
        try:
            SOCKET_PATH.unlink(missing_ok=True)
        except Exception:
            pass

        self.window = None


# ── Signal Handlers ───────────────────────────────────────────────────

_app: TranscriptionOSD | None = None


def _signal_handler(signum, frame):
    if _app:
        _app.stop()


def _sigusr1_handler(signum, frame):
    if _app:
        GLib.idle_add(_app._show)


def _sigusr2_handler(signum, frame):
    if _app:
        GLib.idle_add(_app._hide)


# ── Entry Point ───────────────────────────────────────────────────────


def main():
    global _app

    parser = argparse.ArgumentParser(
        prog="osd-twistt",
        description="Twistt live transcription OSD overlay",
    )
    parser.add_argument(
        "-d", "--daemon",
        action="store_true",
        help="Run as daemon (start hidden, show on SIGUSR1)",
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=550,
        help="Window width in pixels (default: 550)",
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=220,
        help="Window height in pixels (default: 220)",
    )
    args = parser.parse_args()

    # PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    # Signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)
    signal.signal(signal.SIGUSR2, _sigusr2_handler)

    _app = TranscriptionOSD(
        width=args.width,
        height=args.height,
        daemon=args.daemon,
    )

    try:
        _app.run()
    finally:
        # Clean up PID file
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
