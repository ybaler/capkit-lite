# capkit_lite

![capkit_lite punch preset, real output](images/hero.jpg)

Word-highlighted animated captions for short-form video. One file, zero
dependencies. Most caption tools are $20-40/month subscriptions; this is
free and runs offline.

## Install

Requires Python 3.10+, stdlib only.

```
python capkit_lite.py --help
```

## Use

```
python capkit_lite.py transcript.json -o captions.ass
python capkit_lite.py transcript.json --video clip.mp4 --burn out.mp4
python capkit_lite.py subs.srt -o captions.ass
```

Accepts Whisper / faster-whisper / WhisperX / Deepgram / AssemblyAI JSON,
or plain SRT/VTT.

One style only, no batch processing. --burn needs ffmpeg on PATH.

Full version, CapKit, is a $29 one-time purchase, no subscription:
https://y1fygy-sg.myshopify.com/?utm_source=github&utm_medium=readme&utm_campaign=capkit_launch

## Tests

    python -m unittest test_capkit_lite -v

## License

MIT. See LICENSE.
