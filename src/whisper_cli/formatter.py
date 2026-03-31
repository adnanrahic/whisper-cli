import csv
import io
import json


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


def format_json(segments: list[dict]) -> str:
    data = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in segments
    ]
    return json.dumps(data, indent=2)


def format_csv(segments: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["start", "end", "text"])
    for seg in segments:
        writer.writerow([
            f"{seg['start']:.3f}",
            f"{seg['end']:.3f}",
            seg["text"].strip(),
        ])
    return output.getvalue()


def _timestamp_md(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_md(segments: list[dict]) -> str:
    if not segments:
        return "# Transcript"
    lines = ["# Transcript"]
    for seg in segments:
        ts = _timestamp_md(seg["start"])
        text = seg["text"].strip()
        lines.append(f"\n**[{ts}]** {text}")
    return "\n".join(lines)
