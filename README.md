# capkit_lite

Word-highlighted animated captions for short-form video. One file, one
style preset, zero dependencies.

Takes a word-timestamped transcript and writes an .ass subtitle file
where the word being spoken is highlighted as it's said - the caption
style you see on most short-form video. Optionally burns it straight into
your video if ffmpeg is on your PATH.

## Why

Most caption tools are $20-40/month subscriptions. This is free, runs
offline, and is a single Python file you can read top to bottom in five
minutes.

## Install

Nothing to install. Requires Python 3.10+, stdlib only.

```
python capkit_lite.py --help
```

## Use

```
python capkit_lite.py transcript.json -o captions.ass
python capkit_lite.py transcript.json --video clip.mp4 --burn out.mp4
python capkit_lite.py subs.srt -o captions.ass
```

## What it doesn't do

- One style only. No presets, no per-run color/layout tuning beyond
  --active-color and --no-uppercase.
- No batch processing.
