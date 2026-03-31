import json

from whisper_cli.formatter import format_txt, format_srt, format_vtt, format_json, format_csv, format_md


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


class TestFormatJson:
    def test_produces_valid_json_array(self, sample_segments):
        result = format_json(sample_segments)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_first_segment_fields(self, sample_segments):
        result = format_json(sample_segments)
        data = json.loads(result)
        assert data[0]["start"] == 0.0
        assert data[0]["end"] == 2.5
        assert data[0]["text"] == "Hello world."

    def test_text_is_stripped(self, sample_segments):
        result = format_json(sample_segments)
        data = json.loads(result)
        for entry in data:
            assert entry["text"] == entry["text"].strip()

    def test_indented_with_two_spaces(self, sample_segments):
        result = format_json(sample_segments)
        # json.dumps with indent=2 produces lines starting with two spaces
        assert '  "start"' in result

    def test_empty_segments(self):
        assert format_json([]) == "[]"


class TestFormatCsv:
    def test_header_row(self, sample_segments):
        result = format_csv(sample_segments)
        first_line = result.split("\n")[0]
        assert first_line == "start,end,text"

    def test_data_rows_count(self, sample_segments):
        result = format_csv(sample_segments)
        lines = result.split("\n")
        # header + 3 data rows + trailing newline produces empty string at end
        data_lines = [l for l in lines[1:] if l]
        assert len(data_lines) == 3

    def test_float_formatting_three_decimals(self, sample_segments):
        result = format_csv(sample_segments)
        lines = result.split("\n")
        assert lines[1] == "0.000,2.500,Hello world."
        assert lines[2] == "2.500,5.000,This is a test."

    def test_empty_segments(self):
        result = format_csv([])
        assert result == "start,end,text\n"


class TestFormatMd:
    def test_starts_with_h1_transcript(self, sample_segments):
        result = format_md(sample_segments)
        assert result.startswith("# Transcript")

    def test_inline_timestamp_mm_ss_format(self, sample_segments):
        result = format_md(sample_segments)
        assert "**[00:00]**" in result
        assert "**[00:02]**" in result

    def test_text_content(self, sample_segments):
        result = format_md(sample_segments)
        assert "Hello world." in result
        assert "This is a test." in result
        assert "Final segment here." in result

    def test_each_segment_on_its_own_paragraph(self, sample_segments):
        result = format_md(sample_segments)
        # Segments separated by blank lines
        assert "\n\n**[" in result

    def test_empty_segments(self):
        assert format_md([]) == "# Transcript"

    def test_hh_mm_ss_for_long_audio(self):
        segments = [{"start": 3661.0, "end": 3665.0, "text": " Long audio."}]
        result = format_md(segments)
        assert "**[01:01:01]**" in result
