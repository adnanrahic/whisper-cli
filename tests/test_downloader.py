import os
import pytest
from unittest.mock import patch, MagicMock
from whisper_cli.downloader import is_youtube_url, download_audio, sanitize_filename, cleanup_temp_dir


class TestIsYoutubeUrl:
    def test_standard_watch_url(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_short_url(self):
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True

    def test_shorts_url(self):
        assert is_youtube_url("https://www.youtube.com/shorts/abc123") is True

    def test_local_file(self):
        assert is_youtube_url("video.mp4") is False

    def test_other_url(self):
        assert is_youtube_url("https://example.com/video") is False

    def test_empty_string(self):
        assert is_youtube_url("") is False


class TestSanitizeFilename:
    def test_simple_name(self):
        assert sanitize_filename("My Video Title") == "My_Video_Title"

    def test_special_chars(self):
        result = sanitize_filename("Video: A/B Test! (2024)")
        assert "/" not in result
        assert ":" not in result
        assert "!" not in result

    def test_truncation(self):
        long_name = "a" * 250
        assert len(sanitize_filename(long_name)) <= 200

    def test_collapses_underscores(self):
        result = sanitize_filename("hello___world")
        assert "__" not in result


class TestDownloadAudio:
    @patch("whisper_cli.downloader.yt_dlp.YoutubeDL")
    def test_returns_temp_dir_audio_path_and_title(self, mock_ydl_class, tmp_path):
        # Set up mock
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "abc123", "title": "Test Video"}

        # Create a fake audio file in a predictable temp dir
        with patch("whisper_cli.downloader.tempfile.mkdtemp", return_value=str(tmp_path)):
            fake_audio = tmp_path / "abc123.mp3"
            fake_audio.write_bytes(b"fake audio")

            temp_dir, audio_path, title = download_audio("https://youtu.be/abc123")

            assert temp_dir == str(tmp_path)
            assert os.path.exists(audio_path)
            assert title == "Test Video"

    @patch("whisper_cli.downloader.yt_dlp.YoutubeDL")
    def test_raises_on_no_audio_file(self, mock_ydl_class, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "abc123", "title": "Test Video"}

        with patch("whisper_cli.downloader.tempfile.mkdtemp", return_value=str(tmp_path)):
            with pytest.raises(RuntimeError, match="No audio file"):
                download_audio("https://youtu.be/abc123")


class TestCleanupTempDir:
    def test_removes_directory(self, tmp_path):
        test_dir = tmp_path / "temp"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")
        cleanup_temp_dir(str(test_dir))
        assert not test_dir.exists()

    def test_ignores_nonexistent_dir(self):
        cleanup_temp_dir("/nonexistent/path")  # Should not raise
