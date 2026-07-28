#!/usr/bin/env python3
"""capkit_lite.py -- word-highlighted animated captions, one file, zero deps.

Takes a word-timestamped transcript (Whisper/faster-whisper/WhisperX/Deepgram/
AssemblyAI JSON, or plain SRT/VTT) and writes an .ass subtitle file with the
spoken word highlighted as it's said. Optionally burns it into a video via
ffmpeg, if ffmpeg is on PATH.

Usage:
    python capkit_lite.py transcript.json -o captions.ass
    python capkit_lite.py transcript.json --video clip.mp4 --burn out.mp4
    python capkit_lite.py subs.srt -o captions.ass

This is the free, single-preset version of CapKit. The full tool
(https://y1fygy-sg.myshopify.com) adds 8 style presets, box/karaoke/solo
modes, and per-run style overrides for $29 one-time, no subscription.

MIT License. See LICENSE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__version__ = "1.0.0"


@dataclass
class Word:
    text: str
    start: float
    end: float

    def __post_init__(self):
        self.text = self.text.strip()
        if self.end < self.start:
            self.end = self.start


@dataclass
class Cue:
    words: List[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0


STYLE = {
    "font": "Arial Black",
    "size": 110,
    "base_color": "#FFFFFF",
    "active_color": "#FFDD00",
    "outline_color": "#000000",
    "shadow_color": "#000000",
    "outline": 8.0,
    "shadow": 3.0,
    "uppercase": True,
    "pop": 1.12,
    "margin_h": 80,
    "max_words": 3,
    "max_chars": 20,
    "max_duration": 2.5,
}


class ParseError(Exception):
    pass


def _coerce_word(raw: Dict[str, Any]) -> Optional[Word]:
    text = raw.get("punctuated_word") or raw.get("word") or raw.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    start = raw.get("start", raw.get("startTime", raw.get("from")))
    end = raw.get("end", raw.get("endTime", raw.get("to")))
    if start is None or end is None:
        return None
    try:
        return Word(text=text, start=float(start), end=float(end))
    except (TypeError, ValueError):
        return None


def _to_seconds(words: List[Word]) -> List[Word]:
    for w in words:
        w.start /= 1000.0
        w.end /= 1000.0
    return words


def _looks_like_milliseconds(words: List[Word]) -> bool:
    if not words or words[-1].end <= 60:
        return False
    return all(float(w.start).is_integer() and float(w.end).is_integer() for w in words)


def parse_json(raw_text: str) -> List[Word]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"file is not valid JSON: {exc}") from exc

    words: List[Word] = []
    native_unit: Optional[str] = None

    if isinstance(data, dict) and isinstance(data.get("word_segments"), list):
        words = [w for r in data["word_segments"] if (w := _coerce_word(r))]
        if words:
            native_unit = "s"

    if not words and isinstance(data, dict) and isinstance(data.get("segments"), list):
        for seg in data["segments"]:
            if not isinstance(seg, dict):
                continue
            for raw in seg.get("words") or []:
                w = _coerce_word(raw)
                if w:
                    words.append(w)
        if words:
            native_unit = "s"

    if not words and isinstance(data, dict) and isinstance(data.get("words"), list):
        words = [w for r in data["words"] if (w := _coerce_word(r))]
        if words:
            native_unit = "ms"

    if not words and isinstance(data, dict):
        try:
            alts = data["results"]["channels"][0]["alternatives"][0]
            words = [w for r in (alts.get("words") or []) if (w := _coerce_word(r))]
            if words:
                native_unit = "s"
        except (KeyError, IndexError, TypeError):
            pass

    if not words and isinstance(data, list):
        words = [w for r in data if isinstance(r, dict) and (w := _coerce_word(r))]

    if not words:
        raise ParseError(
            "no word-level timestamps found. Re-run your transcriber with word "
            "timestamps enabled (Whisper: word_timestamps=True)."
        )

    words.sort(key=lambda w: w.start)
    if native_unit == "ms":
        return _to_seconds(words)
    if native_unit == "s":
        return words
    return _to_seconds(words) if _looks_like_milliseconds(words) else words


_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_TAG = re.compile(r"<[^>]+>")


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(raw_text: str) -> List[Word]:
    words: List[Word] = []
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _TS.search(lines[i])
        if not m:
            i += 1
            continue
        start = _ts_to_seconds(*m.groups()[:4])
        end = _ts_to_seconds(*m.groups()[4:])
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip() and not _TS.search(lines[i]):
            text_lines.append(lines[i])
            i += 1
        text = _TAG.sub("", " ".join(text_lines)).strip()
        tokens = [t for t in text.split() if t]
        if not tokens:
            continue
        span = max(end - start, 0.001)
        weights = [max(len(t), 1) for t in tokens]
        total = sum(weights)
        cursor = start
        for tok, weight in zip(tokens, weights):
            dur = span * (weight / total)
            words.append(Word(text=tok, start=cursor, end=cursor + dur))
            cursor += dur
    if not words:
        raise ParseError("no cues found in subtitle file")
    return words


def load(path: str) -> List[Word]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    head = raw.lstrip()[:1]
    if head in ("{", "["):
        return parse_json(raw)
    if _TS.search(raw):
        return parse_srt(raw)
    return parse_json(raw)


_SENTENCE_END = (".", "!", "?", "…")
_SOFT_END = (",", ";", ":", "-")


def chunk(words: List[Word], max_words: int, max_chars: int, max_duration: float,
          max_gap: float = 0.55, hold_last: float = 0.12) -> List[Cue]:
    cues: List[Cue] = []
    current: List[Word] = []

    def flush():
        if current:
            cues.append(Cue(words=list(current)))
            current.clear()

    for word in words:
        if current:
            gap = word.start - current[-1].end
            prospective_chars = len(" ".join(w.text for w in current)) + 1 + len(word.text)
            duration = word.end - current[0].start
            if gap > max_gap:
                flush()
            elif len(current) >= max_words:
                flush()
            elif prospective_chars > max_chars:
                flush()
            elif duration > max_duration:
                flush()
            elif current[-1].text.endswith(_SOFT_END) and len(current) >= max(2, max_words - 1):
                flush()
        current.append(word)
        if word.text.endswith(_SENTENCE_END):
            flush()
    flush()

    if hold_last > 0:
        for i, cue in enumerate(cues):
            if not cue.words:
                continue
            limit = cues[i + 1].start if i + 1 < len(cues) else cue.end + hold_last
            cue.words[-1].end = min(cue.end + hold_last, limit)

    return cues


def hex_to_ass(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"bad colour '{hex_color}', expected #RRGGBB")
    int(h, 16)
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def to_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    return (
        text.replace("\\", "")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def build_header(style: dict, width: int, height: int) -> str:
    fields = [
        "Default",
        style["font"],
        str(style["size"]),
        hex_to_ass(style["base_color"]),
        hex_to_ass(style["active_color"]),
        hex_to_ass(style["outline_color"]),
        hex_to_ass(style["shadow_color"]),
        "-1", "0", "0", "0", "100", "100", "0", "0",
        "1",
        f"{style['outline']:g}",
        f"{style['shadow']:g}",
        "5",
        str(style["margin_h"]), str(style["margin_h"]), "0",
        "1",
    ]
    return (
        "[Script Info]\n"
        "; Generated by capkit_lite\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: None\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {','.join(fields)}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )


def _render_text(cue: Cue, active: int, style: dict) -> str:
    base_tag = f"{{\\c{hex_to_ass(style['base_color'])}\\fscx100\\fscy100}}"
    scale = int(round(style["pop"] * 100))
    active_tag = f"{{\\c{hex_to_ass(style['active_color'])}\\fscx{scale}\\fscy{scale}}}"
    parts = []
    for idx, w in enumerate(cue.words):
        token = w.text.upper() if style["uppercase"] else w.text
        token = escape(token)
        parts.append(f"{active_tag}{token}" if idx == active else f"{base_tag}{token}")
    return " ".join(parts)


def build_events(cues, style: dict) -> List[str]:
    events: List[str] = []
    for cue in cues:
        if not cue.words:
            continue
        last = len(cue.words) - 1
        for i, word in enumerate(cue.words):
            start = word.start
            end = cue.words[i + 1].start if i < last else cue.end
            if end <= start:
                end = start + 0.04
            text = _render_text(cue, active=i, style=style)
            events.append(
                f"Dialogue: 0,{to_timestamp(start)},{to_timestamp(end)},"
                f"Default,,0,0,0,,{text}"
            )
    return events


def write_ass(path: str, cues, style: dict, width: int, height: int) -> None:
    content = build_header(style, width, height) + "\n".join(build_events(cues, style)) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
