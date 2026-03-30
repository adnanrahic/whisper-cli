import pytest


@pytest.fixture
def sample_segments():
    """Mock Whisper transcription segments."""
    return [
        {"start": 0.0, "end": 2.5, "text": " Hello world."},
        {"start": 2.5, "end": 5.0, "text": " This is a test."},
        {"start": 5.0, "end": 8.3, "text": " Final segment here."},
    ]
