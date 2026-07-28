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
