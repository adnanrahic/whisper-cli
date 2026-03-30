# Whisper CLI — Design Spec

Local CLI tool that transcribes audio from video/audio files into text using OpenAI's Whisper.

## CLI Interface

```
whisper-cli <file1> [file2 ...] [options]
```

**Positional arguments:** One or more input files (.mp4, .mp3, .wav, .webm, etc.)

**Flags:**

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model` | `-m` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` | `base` |
| `--output` | `-o` | Output directory for transcription files | Same directory as input file |
| `--format` | `-f` | Output format: `txt`, `srt`, `vtt` | `txt` |
| `--stdout` | | Print transcription to stdout instead of writing files | `false` |
| `--language` | `-l` | Force a source language (auto-detect by default) | `None` |

**Behavior:**

- Each input file produces one output file named `<filename>.<format>`.
- With `--stdout`, transcriptions print to stdout separated by `=== filename.mp4 ===` headers.
- Invalid files produce a clear error; processing continues to the next file.
- Progress is shown per file (Whisper logs to stderr by default).

## Architecture

```
whisper/
├── pyproject.toml          # Dependencies, CLI entry point
├── src/
│   └── whisper_cli/
│       ├── __init__.py
│       ├── cli.py          # CLI argument parsing (click)
│       ├── transcriber.py  # Whisper model loading & transcription
│       └── formatter.py    # Output formatting (txt, srt, vtt)
└── tests/
    ├── test_cli.py
    ├── test_transcriber.py
    └── test_formatter.py
```

## Data Flow

1. `cli.py` parses arguments, validates that input files exist and are readable.
2. Loads Whisper model once via `transcriber.py`, reuses across all files.
3. For each file: `transcriber.py` calls `whisper.transcribe()` and returns segments (text + timestamps).
4. `formatter.py` formats segments as txt (plain text), srt (numbered + timestamps), or vtt (WebVTT).
5. Output is written to file or stdout based on flags.

**Key decisions:**

- Model loaded once, not per-file, to avoid redundant multi-hundred-MB loads.
- `click` for CLI parsing — cleaner than `argparse`, native multi-value argument support.
- Formatter is its own module so adding formats later is trivial.
- No audio extraction step — Whisper handles video files directly via ffmpeg.

## Dependencies

- `openai-whisper` — core transcription library
- `click` — CLI framework
- `ffmpeg` — system dependency (must be installed separately via brew/apt/etc.)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Missing ffmpeg | Detect at startup, print install instructions, exit with code 2 |
| File not found | Skip with error message, continue to next file |
| Unsupported format | Catch ffmpeg/Whisper error, report, continue to next file |
| Invalid model name | Click validates against allowed choices, exits with usage help |
| Keyboard interrupt | Clean exit, no partial output files |

**Exit codes:**

- `0` — all files transcribed successfully
- `1` — some files failed (partial success)
- `2` — total failure (no files processed, or missing dependency)

## Testing

**Unit tests:**

- `test_formatter.py` — txt/srt/vtt formatting from mock segment data. Pure functions, no Whisper needed.
- `test_cli.py` — argument parsing, validation, error cases via click's `CliRunner`.

**Integration tests:**

- `test_transcriber.py` — actual transcription with a short (~5s) test audio file using `tiny` model.

**Out of scope:** Whisper accuracy, exhaustive audio format coverage.

## Technology

- **Language:** Python
- **Whisper integration:** Direct library import (`openai-whisper`), not subprocess wrapper
- **CLI framework:** click
- **Package management:** pyproject.toml with pip
