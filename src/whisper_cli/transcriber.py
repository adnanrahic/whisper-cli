import os
import shutil
import whisper


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def load_model(name: str) -> whisper.Whisper:
    return whisper.load_model(name)


def transcribe_file(
    model: whisper.Whisper,
    file_path: str,
    language: str | None = None,
) -> list[dict]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    result = model.transcribe(file_path, language=language)
    return result["segments"]
