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
