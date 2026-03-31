import os
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from whisper_cli.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_audio(tmp_path):
    """Create a fake file to pass existence checks."""
    f = tmp_path / "test.mp4"
    f.write_bytes(b"fake")
    return str(f)


class TestCliValidation:
    def test_no_files_shows_usage(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_nonexistent_file_reports_error(self, mock_ffmpeg, runner):
        result = runner.invoke(main, ["/nonexistent/file.mp4"])
        assert result.exit_code == 2

    def test_invalid_model_rejected(self, runner, temp_audio):
        result = runner.invoke(main, [temp_audio, "--model", "invalid"])
        assert result.exit_code != 0

    def test_stdout_and_output_mutually_exclusive(self, runner, temp_audio):
        result = runner.invoke(main, [temp_audio, "--stdout", "--output", "/tmp"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or result.exit_code == 2

    def test_version_flag(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_missing_ffmpeg_exits_with_2(self, runner, temp_audio):
        with patch("whisper_cli.cli.check_ffmpeg", return_value=False):
            result = runner.invoke(main, [temp_audio])
            assert result.exit_code == 2
            assert "ffmpeg" in result.output.lower()


class TestCliOutput:
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_writes_txt_file(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        result = runner.invoke(main, [str(audio)])
        assert result.exit_code == 0
        output_file = tmp_path / "test.txt"
        assert output_file.exists()
        assert output_file.read_text().strip() == "Hello."

    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_stdout_flag_prints_to_stdout(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        result = runner.invoke(main, [str(audio), "--stdout"])
        assert result.exit_code == 0
        assert "Hello." in result.output

    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_custom_output_dir(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        result = runner.invoke(main, [str(audio), "--output", str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "test.txt").exists()

    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_partial_failure_returns_exit_1(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        good = tmp_path / "good.mp4"
        good.write_bytes(b"fake")
        mock_transcribe.side_effect = [
            [{"start": 0.0, "end": 1.0, "text": " Hello."}],
            Exception("decode error"),
        ]
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"fake")
        result = runner.invoke(main, [str(good), str(bad)])
        assert result.exit_code == 1

    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_stdout_with_multiple_files_shows_headers(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        a = tmp_path / "a.mp4"
        a.write_bytes(b"fake")
        b = tmp_path / "b.mp4"
        b.write_bytes(b"fake")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        result = runner.invoke(main, [str(a), str(b), "--stdout"])
        assert result.exit_code == 0
        assert "=== a ===" in result.output
        assert "=== b ===" in result.output

    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    @patch("whisper_cli.cli.logging")
    def test_quiet_flag_suppresses_logging(self, mock_logging, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        result = runner.invoke(main, [str(audio), "--quiet"])
        assert result.exit_code == 0
        mock_logging.getLogger.assert_called_with("whisper")
        mock_logger.setLevel.assert_called_once()

    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_format_srt_writes_srt_file(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        result = runner.invoke(main, [str(audio), "--format", "srt"])
        assert result.exit_code == 0
        assert (tmp_path / "test.srt").exists()

    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_format_json_writes_json_file(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        result = runner.invoke(main, [str(audio), "--format", "json"])
        assert result.exit_code == 0
        assert (tmp_path / "test.json").exists()


class TestCliYoutubeInput:
    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.download_audio")
    @patch("whisper_cli.cli.is_youtube_url", return_value=True)
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_youtube_url_transcribes(self, mock_ffmpeg, mock_yt, mock_download, mock_transcribe, mock_load, mock_cleanup, runner, tmp_path):
        fake_audio = tmp_path / "abc123.mp3"
        fake_audio.write_bytes(b"fake")
        mock_download.return_value = (str(tmp_path), str(fake_audio), "Test Video")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello from YouTube."},
        ]
        result = runner.invoke(main, ["https://youtu.be/abc123", "--stdout"])
        assert result.exit_code == 0
        assert "Hello from YouTube." in result.output

    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.download_audio")
    @patch("whisper_cli.cli.is_youtube_url", side_effect=lambda x: x.startswith("https://"))
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_mixed_url_and_file(self, mock_ffmpeg, mock_yt, mock_download, mock_transcribe, mock_load, mock_cleanup, runner, tmp_path):
        # Local file
        local = tmp_path / "local.mp4"
        local.write_bytes(b"fake")
        # YouTube mock
        fake_audio = tmp_path / "yt.mp3"
        fake_audio.write_bytes(b"fake")
        mock_download.return_value = (str(tmp_path), str(fake_audio), "YT Video")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        result = runner.invoke(main, ["https://youtu.be/abc", str(local), "--stdout"])
        assert result.exit_code == 0
        assert "Hello." in result.output

    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.download_audio", side_effect=RuntimeError("Download failed"))
    @patch("whisper_cli.cli.is_youtube_url", return_value=True)
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_youtube_download_failure_exits_2(self, mock_ffmpeg, mock_yt, mock_download, mock_cleanup, runner):
        result = runner.invoke(main, ["https://youtu.be/bad"])
        assert result.exit_code == 2

    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.download_audio")
    @patch("whisper_cli.cli.is_youtube_url", return_value=True)
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_temp_dirs_cleaned_up(self, mock_ffmpeg, mock_yt, mock_download, mock_transcribe, mock_load, mock_cleanup, runner, tmp_path):
        fake_audio = tmp_path / "abc.mp3"
        fake_audio.write_bytes(b"fake")
        mock_download.return_value = (str(tmp_path / "tempdir"), str(fake_audio), "Video")
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 1.0, "text": " Hello."},
        ]
        runner.invoke(main, ["https://youtu.be/abc", "--stdout"])
        mock_cleanup.assert_called_once_with(str(tmp_path / "tempdir"))


class TestCliMixedInputsAndFormats:
    """Test multiple inputs (local files + YouTube URLs) with various output formats."""

    SEGMENTS = [
        {"start": 0.0, "end": 2.5, "text": " Hello world."},
        {"start": 2.5, "end": 5.0, "text": " This is a test."},
    ]

    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.download_audio")
    @patch("whisper_cli.cli.is_youtube_url", side_effect=lambda x: x.startswith("https://"))
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_mixed_inputs_write_files_for_each(self, mock_ffmpeg, mock_yt, mock_download, mock_transcribe, mock_load, mock_cleanup, runner, tmp_path):
        # Local files
        vid1 = tmp_path / "interview.mp4"
        vid1.write_bytes(b"fake")
        vid2 = tmp_path / "meeting.wav"
        vid2.write_bytes(b"fake")
        # YouTube mocks
        yt_audio1 = tmp_path / "yt1.mp3"
        yt_audio1.write_bytes(b"fake")
        yt_audio2 = tmp_path / "yt2.mp3"
        yt_audio2.write_bytes(b"fake")
        mock_download.side_effect = [
            (str(tmp_path / "tmp1"), str(yt_audio1), "My YouTube Video"),
            (str(tmp_path / "tmp2"), str(yt_audio2), "Another: Cool Video!"),
        ]
        mock_transcribe.return_value = self.SEGMENTS

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = runner.invoke(main, [
            str(vid1),
            "https://youtu.be/abc",
            str(vid2),
            "https://youtu.be/def",
            "--output", str(out_dir),
        ])
        assert result.exit_code == 0
        # Local files use basename without extension
        assert (out_dir / "interview.txt").exists()
        assert (out_dir / "meeting.txt").exists()
        # YouTube files use sanitized title
        assert (out_dir / "My_YouTube_Video.txt").exists()
        assert (out_dir / "Another_Cool_Video.txt").exists()

    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.download_audio")
    @patch("whisper_cli.cli.is_youtube_url", side_effect=lambda x: x.startswith("https://"))
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_mixed_inputs_stdout_shows_all_headers(self, mock_ffmpeg, mock_yt, mock_download, mock_transcribe, mock_load, mock_cleanup, runner, tmp_path):
        vid = tmp_path / "local.mp4"
        vid.write_bytes(b"fake")
        yt_audio = tmp_path / "yt.mp3"
        yt_audio.write_bytes(b"fake")
        mock_download.return_value = (str(tmp_path / "tmp"), str(yt_audio), "YT Title")
        mock_transcribe.return_value = self.SEGMENTS

        result = runner.invoke(main, [
            str(vid), "https://youtu.be/abc", "--stdout",
        ])
        assert result.exit_code == 0
        assert "=== local ===" in result.output
        assert "=== YT_Title ===" in result.output
        assert "Hello world." in result.output

    @pytest.mark.parametrize("fmt,ext", [
        ("txt", "txt"),
        ("srt", "srt"),
        ("vtt", "vtt"),
        ("json", "json"),
        ("csv", "csv"),
        ("md", "md"),
    ])
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_each_format_creates_correct_extension(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path, fmt, ext):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        mock_transcribe.return_value = self.SEGMENTS
        result = runner.invoke(main, [str(audio), "--format", fmt])
        assert result.exit_code == 0
        assert (tmp_path / f"test.{ext}").exists()

    @pytest.mark.parametrize("fmt", ["json", "csv", "md"])
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_ai_formats_stdout_contain_content(self, mock_ffmpeg, mock_transcribe, mock_load, runner, tmp_path, fmt):
        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"fake")
        mock_transcribe.return_value = self.SEGMENTS
        result = runner.invoke(main, [str(audio), "--format", fmt, "--stdout"])
        assert result.exit_code == 0
        assert "Hello world." in result.output
        assert "This is a test." in result.output

    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.download_audio")
    @patch("whisper_cli.cli.is_youtube_url", side_effect=lambda x: x.startswith("https://"))
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_youtube_with_json_format_writes_valid_json(self, mock_ffmpeg, mock_yt, mock_download, mock_transcribe, mock_load, mock_cleanup, runner, tmp_path):
        import json
        yt_audio = tmp_path / "yt.mp3"
        yt_audio.write_bytes(b"fake")
        mock_download.return_value = (str(tmp_path / "tmp"), str(yt_audio), "My Video")
        mock_transcribe.return_value = self.SEGMENTS

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = runner.invoke(main, [
            "https://youtu.be/abc", "--format", "json", "--output", str(out_dir),
        ])
        assert result.exit_code == 0
        json_file = out_dir / "My_Video.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert len(data) == 2
        assert data[0]["text"] == "Hello world."
        assert data[1]["start"] == 2.5

    @patch("whisper_cli.cli.cleanup_temp_dir")
    @patch("whisper_cli.cli.load_model")
    @patch("whisper_cli.cli.transcribe_file")
    @patch("whisper_cli.cli.download_audio")
    @patch("whisper_cli.cli.is_youtube_url", side_effect=lambda x: x.startswith("https://"))
    @patch("whisper_cli.cli.check_ffmpeg", return_value=True)
    def test_partial_failure_with_mixed_inputs(self, mock_ffmpeg, mock_yt, mock_download, mock_transcribe, mock_load, mock_cleanup, runner, tmp_path):
        # One good local file, one failing YouTube download
        vid = tmp_path / "good.mp4"
        vid.write_bytes(b"fake")
        mock_download.side_effect = RuntimeError("Network error")
        mock_transcribe.return_value = self.SEGMENTS

        result = runner.invoke(main, [str(vid), "https://youtu.be/bad"])
        assert result.exit_code == 1  # partial success
