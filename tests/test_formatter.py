from whisper_cli.formatter import format_txt, format_srt, format_vtt


class TestFormatTxt:
    def test_joins_segments_as_plain_text(self, sample_segments):
        result = format_txt(sample_segments)
        assert result == "Hello world.\nThis is a test.\nFinal segment here."

    def test_empty_segments(self):
        assert format_txt([]) == ""


class TestFormatSrt:
    def test_numbered_entries_with_timestamps(self, sample_segments):
        result = format_srt(sample_segments)
        lines = result.strip().split("\n")
        # First entry
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,000 --> 00:00:02,500"
        assert lines[2] == "Hello world."
        # Blank line separator
        assert lines[3] == ""
        # Second entry
        assert lines[4] == "2"
        assert lines[5] == "00:00:02,500 --> 00:00:05,000"
        assert lines[6] == "This is a test."

    def test_empty_segments(self):
        assert format_srt([]) == ""


class TestFormatVtt:
    def test_webvtt_header_and_entries(self, sample_segments):
        result = format_vtt(sample_segments)
        lines = result.strip().split("\n")
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""
        assert lines[2] == "00:00:00.000 --> 00:00:02.500"
        assert lines[3] == "Hello world."

    def test_empty_segments(self):
        assert format_vtt([]) == "WEBVTT"
