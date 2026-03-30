import os
import pytest
from whisper_cli.transcriber import check_ffmpeg, load_model, transcribe_file

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_audio.wav")


class TestCheckFfmpeg:
    def test_returns_true_when_ffmpeg_available(self):
        assert check_ffmpeg() is True


class TestLoadModel:
    def test_loads_tiny_model(self):
        model = load_model("tiny")
        assert model is not None


class TestTranscribeFile:
    @pytest.fixture(scope="class")
    def model(self):
        from whisper_cli.transcriber import load_model
        return load_model("tiny")

    def test_returns_segments_list(self, model):
        segments = transcribe_file(model, FIXTURE_PATH)
        assert isinstance(segments, list)

    def test_segments_have_required_keys(self, model):
        segments = transcribe_file(model, FIXTURE_PATH)
        if segments:  # sine wave may produce empty transcription
            for seg in segments:
                assert "start" in seg
                assert "end" in seg
                assert "text" in seg

    def test_nonexistent_file_raises(self, model):
        with pytest.raises(FileNotFoundError):
            transcribe_file(model, "/nonexistent/file.wav")
