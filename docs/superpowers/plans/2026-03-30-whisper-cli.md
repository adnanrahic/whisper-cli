# Whisper CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that transcribes audio/video files to text using OpenAI's Whisper library.

**Architecture:** A `click`-based CLI (`cli.py`) delegates to a transcriber module (`transcriber.py`) for Whisper model loading and transcription, and a formatter module (`formatter.py`) for output as txt/srt/vtt. The model is loaded once and reused across all input files.

**Tech Stack:** Python, openai-whisper, click, pytest, ffmpeg (system dependency)

**Spec:** `docs/superpowers/specs/2026-03-30-whisper-cli-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `.gitignore` | Exclude __pycache__, egg-info, dist, build |
| `pyproject.toml` | Project metadata, dependencies, `whisper-cli` entry point |
| `src/whisper_cli/__init__.py` | Package init, `__version__` |
| `src/whisper_cli/formatter.py` | Format transcription segments as txt, srt, or vtt |
| `src/whisper_cli/transcriber.py` | Load Whisper model, transcribe files, check ffmpeg |
| `src/whisper_cli/cli.py` | Click CLI: arg parsing, validation, orchestration |
| `tests/conftest.py` | Shared fixtures (sample segments, temp dirs) |
| `tests/test_formatter.py` | Unit tests for all three formatters |
| `tests/test_transcriber.py` | Integration tests with real Whisper (tiny model) |
| `tests/test_cli.py` | CLI tests via click's CliRunner |
| `tests/fixtures/test_audio.wav` | Generated 5s sine-wave audio fixture |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/whisper_cli/__init__.py`

- [ ] **Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.egg-info/
.eggs/
dist/
build/
*.egg
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "whisper-cli"
version = "0.1.0"
description = "Transcribe audio/video files to text using OpenAI Whisper"
requires-python = ">=3.9"
dependencies = [
    "openai-whisper",
    "click>=8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[project.scripts]
whisper-cli = "whisper_cli.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Create src/whisper_cli/__init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Install in dev mode**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully, `whisper-cli --help` will fail (cli.py doesn't exist yet, that's fine)

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml src/whisper_cli/__init__.py
git commit -m "feat: scaffold project with pyproject.toml and package init"
```

---

### Task 2: Formatter Module (TDD)

**Files:**
- Create: `src/whisper_cli/formatter.py`
- Create: `tests/conftest.py`
- Create: `tests/test_formatter.py`

- [ ] **Step 1: Create test fixtures in conftest.py**

```python
import pytest


@pytest.fixture
def sample_segments():
    """Mock Whisper transcription segments."""
    return [
        {"start": 0.0, "end": 2.5, "text": " Hello world."},
        {"start": 2.5, "end": 5.0, "text": " This is a test."},
        {"start": 5.0, "end": 8.3, "text": " Final segment here."},
    ]
```

- [ ] **Step 2: Write failing tests for all three formatters**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_formatter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'whisper_cli.formatter'`

- [ ] **Step 4: Implement formatter.py**

```python
def _timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_txt(segments: list[dict]) -> str:
    return "\n".join(seg["text"].strip() for seg in segments)


def format_srt(segments: list[dict]) -> str:
    if not segments:
        return ""
    entries = []
    for i, seg in enumerate(segments, 1):
        start = _timestamp_srt(seg["start"])
        end = _timestamp_srt(seg["end"])
        text = seg["text"].strip()
        entries.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(entries) + "\n"


def format_vtt(segments: list[dict]) -> str:
    if not segments:
        return "WEBVTT"
    entries = ["WEBVTT"]
    for seg in segments:
        start = _timestamp_vtt(seg["start"])
        end = _timestamp_vtt(seg["end"])
        text = seg["text"].strip()
        entries.append(f"\n{start} --> {end}\n{text}")
    return "\n".join(entries) + "\n"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_formatter.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/whisper_cli/formatter.py tests/conftest.py tests/test_formatter.py
git commit -m "feat: add formatter module with txt, srt, vtt output"
```

---

### Task 3: Transcriber Module (TDD)

**Files:**
- Create: `src/whisper_cli/transcriber.py`
- Create: `tests/test_transcriber.py`
- Create: `tests/fixtures/test_audio.wav`

- [ ] **Step 1: Generate test audio fixture**

```bash
python -c "
import numpy as np
import wave
import struct

sample_rate = 16000
duration = 3
frequency = 440
samples = np.sin(2 * np.pi * frequency * np.arange(sample_rate * duration) / sample_rate)
samples = (samples * 32767).astype(np.int16)

import os
os.makedirs('tests/fixtures', exist_ok=True)
with wave.open('tests/fixtures/test_audio.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(samples.tobytes())
print('Created test_audio.wav')
"
```

- [ ] **Step 2: Write failing tests for transcriber**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'whisper_cli.transcriber'`

- [ ] **Step 4: Implement transcriber.py**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_transcriber.py -v`
Expected: All tests PASS (may take ~30s first run to download tiny model)

- [ ] **Step 6: Commit**

```bash
git add src/whisper_cli/transcriber.py tests/test_transcriber.py tests/fixtures/test_audio.wav
git commit -m "feat: add transcriber module with model loading and file transcription"
```

---

### Task 4: CLI Module (TDD)

**Files:**
- Create: `src/whisper_cli/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI argument parsing and validation**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'whisper_cli.cli'`

- [ ] **Step 3: Implement cli.py**

```python
import os
import sys
import logging
import click
from whisper_cli import __version__
from whisper_cli.transcriber import check_ffmpeg, load_model, transcribe_file
from whisper_cli.formatter import format_txt, format_srt, format_vtt

FORMATTERS = {
    "txt": format_txt,
    "srt": format_srt,
    "vtt": format_vtt,
}

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]


@click.command()
@click.argument("files", nargs=-1, required=True, type=click.Path())
@click.option("-m", "--model", default="base", type=click.Choice(MODEL_CHOICES), help="Whisper model size.")
@click.option("-o", "--output", default=None, type=click.Path(), help="Output directory.")
@click.option("-f", "--format", "fmt", default="txt", type=click.Choice(["txt", "srt", "vtt"]), help="Output format.")
@click.option("--stdout", is_flag=True, help="Print to stdout instead of writing files.")
@click.option("-l", "--language", default=None, help="Force source language.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress Whisper progress logging.")
@click.version_option(version=__version__)
def main(files, model, output, fmt, stdout, language, quiet):
    """Transcribe audio/video files to text using OpenAI Whisper."""
    if stdout and output:
        click.echo("Error: --stdout and --output are mutually exclusive.", err=True)
        sys.exit(2)

    if not check_ffmpeg():
        click.echo(
            "Error: ffmpeg not found. Install it:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html",
            err=True,
        )
        sys.exit(2)

    # Validate all files exist upfront
    valid_files = []
    for f in files:
        if not os.path.isfile(f):
            click.echo(f"Error: File not found: {f}", err=True)
        else:
            valid_files.append(f)

    if not valid_files:
        click.echo("Error: No valid input files.", err=True)
        sys.exit(2)

    # Validate output directory if specified
    if output and not os.access(output, os.W_OK):
        click.echo(f"Error: Output directory not writable: {output}", err=True)
        sys.exit(2)

    if quiet:
        logging.getLogger("whisper").setLevel(logging.ERROR)

    whisper_model = load_model(model)
    formatter = FORMATTERS[fmt]

    succeeded = 0
    failed = 0
    current_output_path = None

    try:
        for f in valid_files:
            try:
                click.echo(f"Transcribing: {f}", err=True)
                segments = transcribe_file(whisper_model, f, language=language)
                formatted = formatter(segments)

                if stdout:
                    if len(valid_files) > 1:
                        click.echo(f"=== {os.path.basename(f)} ===")
                    click.echo(formatted)
                else:
                    base = os.path.splitext(os.path.basename(f))[0]
                    out_dir = output or os.path.dirname(os.path.abspath(f))
                    current_output_path = os.path.join(out_dir, f"{base}.{fmt}")
                    with open(current_output_path, "w") as out_file:
                        out_file.write(formatted)
                    current_output_path = None
                    click.echo(f"  -> {os.path.join(out_dir, base + '.' + fmt)}", err=True)

                succeeded += 1
            except Exception as e:
                click.echo(f"Error processing {f}: {e}", err=True)
                failed += 1
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        if current_output_path and os.path.exists(current_output_path):
            os.remove(current_output_path)
        sys.exit(2)

    if failed > 0 and succeeded > 0:
        sys.exit(1)
    elif failed > 0 and succeeded == 0:
        sys.exit(2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whisper_cli/cli.py tests/test_cli.py
git commit -m "feat: add CLI with click, validation, and output handling"
```

---

### Task 5: Run Full Test Suite and Verify Entry Point

**Files:**
- None new — verification only

- [ ] **Step 1: Run all unit tests (skip integration)**

Run: `pytest tests/test_formatter.py tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_transcriber.py -v`
Expected: All tests PASS (requires ~30s for tiny model download on first run)

- [ ] **Step 3: Verify CLI entry point works**

Run: `whisper-cli --version`
Expected: `whisper-cli, version 0.1.0`

Run: `whisper-cli --help`
Expected: Shows usage with all flags documented

- [ ] **Step 4: Manual smoke test with the test fixture**

Run: `whisper-cli tests/fixtures/test_audio.wav --model tiny --stdout`
Expected: Prints transcription (likely empty or noise for sine wave — that's fine, confirms the pipeline works)

Run: `whisper-cli tests/fixtures/test_audio.wav --model tiny --format srt`
Expected: Creates `tests/fixtures/test_audio.srt`

- [ ] **Step 5: Clean up smoke test artifacts and commit**

```bash
rm -f tests/fixtures/test_audio.srt tests/fixtures/test_audio.txt
git status
git commit -m "chore: clean up smoke test artifacts" --allow-empty
```

Note: If `git status` shows no changes after cleanup, skip the commit.
