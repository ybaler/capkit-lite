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


class TestChunker(unittest.TestCase):
    def test_splits_on_sentence_end(self):
        words = [ck.Word("Hi.", 0.0, 0.3), ck.Word("There", 0.4, 0.7)]
        cues = ck.chunk(words, max_words=10, max_chars=100, max_duration=10.0)
        self.assertEqual(len(cues), 2)

    def test_respects_max_words(self):
        words = [ck.Word(str(i), i * 0.2, i * 0.2 + 0.15) for i in range(6)]
        cues = ck.chunk(words, max_words=3, max_chars=100, max_duration=10.0)
        self.assertTrue(all(len(c.words) <= 3 for c in cues))

    def test_hold_last_does_not_overlap_next_cue(self):
        words = [ck.Word("a.", 0.0, 0.3), ck.Word("b.", 0.35, 0.5)]
        cues = ck.chunk(words, max_words=1, max_chars=100, max_duration=10.0, hold_last=1.0)
        self.assertLessEqual(cues[0].words[-1].end, cues[1].start)


class TestAssWriter(unittest.TestCase):
    def test_hex_to_ass_roundtrip_order(self):
        self.assertEqual(ck.hex_to_ass("#FFDD00"), "&H0000DDFF")

    def test_bad_colour_raises(self):
        with self.assertRaises(ValueError):
            ck.hex_to_ass("#ZZZ")

    def test_timestamp_format(self):
        self.assertEqual(ck.to_timestamp(0), "0:00:00.00")
        self.assertEqual(ck.to_timestamp(3661.25), "1:01:01.25")

    def test_negative_timestamp_clamped(self):
        self.assertEqual(ck.to_timestamp(-5), "0:00:00.00")

    def test_write_ass_produces_one_event_per_word(self):
        words = [ck.Word("hi", 0.0, 0.3), ck.Word("there", 0.3, 0.6)]
        cues = ck.chunk(words, max_words=10, max_chars=100, max_duration=10.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.ass")
            ck.write_ass(path, cues, dict(ck.STYLE), 1080, 1920)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        self.assertEqual(content.count("Dialogue:"), 2)
        self.assertIn("[V4+ Styles]", content)


class TestEndToEnd(unittest.TestCase):
    def test_load_dispatches_json_vs_srt(self):
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "t.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump({"words": [{"text": "hi", "start": 0, "end": 500}]}, fh)
            words = ck.load(json_path)
            self.assertEqual(words[0].text, "hi")

            srt_path = os.path.join(d, "t.srt")
            with open(srt_path, "w", encoding="utf-8") as fh:
                fh.write("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
            words = ck.load(srt_path)
            self.assertEqual(words[0].text, "hello")


if __name__ == "__main__":
    unittest.main(verbosity=2)
