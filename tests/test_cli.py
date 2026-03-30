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
