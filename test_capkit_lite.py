import json
import os
import tempfile
import unittest

import capkit_lite as ck


class TestParseJson(unittest.TestCase):
    def test_whisper_segments(self):
        data = {"segments": [{"words": [
            {"word": "hi", "start": 0.0, "end": 0.3},
            {"word": "there", "start": 0.3, "end": 0.6},
        ]}]}
        words = ck.parse_json(json.dumps(data))
        self.assertEqual([w.text for w in words], ["hi", "there"])

    def test_assemblyai_milliseconds_converted(self):
        data = {"words": [
            {"text": "hi", "start": 0, "end": 300},
            {"text": "there", "start": 300, "end": 62000},
        ]}
        words = ck.parse_json(json.dumps(data))
        self.assertAlmostEqual(words[-1].end, 62.0, places=3)

    def test_bad_json_raises(self):
        with self.assertRaises(ck.ParseError):
            ck.parse_json("{not json")

    def test_missing_timestamps_raises(self):
        with self.assertRaises(ck.ParseError):
            ck.parse_json(json.dumps({"segments": [{"words": [{"word": "hi"}]}]}))


class TestParseSrt(unittest.TestCase):
    def test_basic_srt(self):
        srt = (
            "1\n00:00:00,000 --> 00:00:02,000\nHello world\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nBye\n"
        )
        words = ck.parse_srt(srt)
        self.assertEqual([w.text for w in words], ["Hello", "world", "Bye"])

    def test_strips_vtt_tags(self):
        srt = "1\n00:00:00,000 --> 00:00:01,000\n<b>hi</b> there\n"
        words = ck.parse_srt(srt)
        self.assertEqual([w.text for w in words], ["hi", "there"])

    def test_no_cues_raises(self):
        with self.assertRaises(ck.ParseError):
            ck.parse_srt("not a subtitle file")
