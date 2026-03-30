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
