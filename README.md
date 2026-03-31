# whisper-cli

A local command-line tool that transcribes audio/video files and YouTube videos to text using [OpenAI's Whisper](https://github.com/openai/whisper). Give it local files or YouTube URLs, and it outputs transcriptions in multiple formats including plain text, subtitles, JSON, CSV, and Markdown.

## Features

- Transcribe any audio/video format supported by ffmpeg
- Transcribe YouTube videos directly from URLs
- Multiple output formats: txt, srt, vtt, json, csv, markdown
- AI-friendly formats (json, csv, md) for feeding transcripts into LLMs
- Batch processing: pass multiple files and/or URLs in one command
- Choose from 5 Whisper model sizes (tiny, base, small, medium, large)
- Print to stdout or write to files
- Auto-detects language (or force a specific one)

## Prerequisites

**ffmpeg** must be installed on your system. Whisper uses it to extract audio from media files.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Installation

### Standalone binary (no Python required)

Download the latest binary for your platform from the [Releases](https://github.com/adnanrahic/whisper-cli/releases) page.

Available platforms:
- macOS (Apple Silicon)
- Linux (x86_64)
- Linux (arm64)
- Windows (x86_64)

### From source (requires Python 3.9+)

```bash
git clone https://github.com/adnanrahic/whisper-cli.git
cd whisper-cli
pip install -e .
```

## Usage

### Basic transcription

```bash
whisper-cli video.mp4
```

This creates `video.txt` in the same directory as the input file using the `base` model.

### Choose a model

Larger models are more accurate but slower. The `large` model maps to the latest variant (large-v3).

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~75 MB | Fastest | Lower |
| `base` | ~150 MB | Fast | Decent |
| `small` | ~500 MB | Moderate | Good |
| `medium` | ~1.5 GB | Slow | High |
| `large` | ~3 GB | Slowest | Best |

```bash
whisper-cli interview.mp4 --model small
```

### YouTube videos

Pass YouTube URLs directly — audio is downloaded automatically:

```bash
whisper-cli https://www.youtube.com/watch?v=VIDEO_ID
whisper-cli https://youtu.be/VIDEO_ID --model small --format json
```

Mix local files and YouTube URLs in one command:

```bash
whisper-cli interview.mp4 https://youtu.be/VIDEO_ID --output ./transcripts
```

Supported URL formats: `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`

### Output formats

```bash
# Plain text (default)
whisper-cli video.mp4 --format txt

# SRT subtitles (with timestamps)
whisper-cli video.mp4 --format srt

# WebVTT subtitles (with timestamps)
whisper-cli video.mp4 --format vtt

# JSON (structured, AI-friendly)
whisper-cli video.mp4 --format json

# CSV (tabular, easy to import)
whisper-cli video.mp4 --format csv

# Markdown (with inline timestamps, great for LLMs)
whisper-cli video.mp4 --format md
```

### Multiple files

```bash
whisper-cli episode1.mp4 episode2.mp4 episode3.mp4
```

### Custom output directory

```bash
whisper-cli recording.mp4 --output ./transcriptions
```

### Print to stdout

```bash
whisper-cli meeting.mp4 --stdout
```

When processing multiple files with `--stdout`, each file's output is separated by a header:

```
=== episode1.mp4 ===
[transcription]

=== episode2.mp4 ===
[transcription]
```

### Force a language

Whisper auto-detects the language by default. To force a specific language:

```bash
whisper-cli video.mp4 --language en
```

### Suppress progress output

```bash
whisper-cli video.mp4 --quiet
```

## All options

```
Usage: whisper-cli [OPTIONS] INPUTS...

  Transcribe audio/video files or YouTube URLs to text using OpenAI Whisper.

Options:
  -m, --model [tiny|base|small|medium|large]       Whisper model size (default: base)
  -o, --output PATH                                Output directory
  -f, --format [txt|srt|vtt|json|csv|md]           Output format (default: txt)
  --stdout                                         Print to stdout instead of writing files
  -l, --language TEXT                               Force source language
  -q, --quiet                                      Suppress Whisper progress logging
  --version                                        Show version and exit
  --help                                           Show help and exit
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All files transcribed successfully |
| 1 | Some files failed (partial success) |
| 2 | Total failure (no files processed, missing dependency, or invalid input) |

## Development

A `Makefile` is included for common tasks. Run `make help` to see all available commands.

### Setup

```bash
git clone https://github.com/adnanrahic/whisper-cli.git
cd whisper-cli
make install-dev
```

### Run tests

```bash
make test            # Run all tests
make test-unit       # Unit tests only (fast, no model download)
make test-integration # Integration tests (downloads tiny model on first run)
```

### Build standalone binary

```bash
make build           # Build binary to ./dist/whisper-cli
make smoke-test      # Verify the binary works
make clean           # Remove build artifacts
```

## Project structure

```
whisper-cli/
├── src/whisper_cli/
│   ├── __init__.py        # Package version
│   ├── cli.py             # CLI entry point (click)
│   ├── downloader.py      # YouTube audio download (yt-dlp)
│   ├── formatter.py       # Output formatting (txt, srt, vtt, json, csv, md)
│   └── transcriber.py     # Whisper model loading and transcription
├── tests/
│   ├── test_cli.py        # CLI tests (mocked transcription)
│   ├── test_downloader.py # YouTube downloader tests (mocked)
│   ├── test_formatter.py  # Formatter unit tests
│   └── test_transcriber.py # Integration tests (real Whisper)
├── whisper-cli.spec       # PyInstaller build configuration
├── pyproject.toml         # Project metadata and dependencies
└── .github/workflows/
    ├── release.yml        # Build binaries on Release publish
    └── dev-build.yml      # Run tests + build on push to main
```

## License

This project uses [OpenAI's Whisper](https://github.com/openai/whisper) which is released under the MIT License.
