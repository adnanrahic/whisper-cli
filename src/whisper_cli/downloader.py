import glob
import os
import re
import shutil
import tempfile

import yt_dlp

_YT_PATTERN = re.compile(
    r'^https?://(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)'
)


def is_youtube_url(s: str) -> bool:
    return bool(_YT_PATTERN.match(s))


def download_audio(url: str, quiet: bool = True) -> tuple[str, str, str]:
    """Download audio from a YouTube URL.

    Returns:
        (temp_dir, audio_file_path, title)

    Raises:
        RuntimeError: if download fails or no audio file is found.
    """
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "quiet": quiet,
        "no_warnings": quiet,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        title = info.get("title", "unknown")
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Download failed: {exc}") from exc

    # Glob for any audio file the post-processor may have produced
    audio_extensions = ["*.mp3", "*.m4a", "*.wav", "*.ogg", "*.opus", "*.flac", "*.aac", "*.webm"]
    audio_files = []
    for pattern in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(temp_dir, pattern)))

    if not audio_files:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"No audio file found in temp dir after download: {temp_dir}")

    return temp_dir, audio_files[0], title


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe version of name."""
    # Replace disallowed characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9._\- ]', '_', name)
    # Collapse consecutive underscores and spaces into a single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Strip leading/trailing underscores and whitespace
    sanitized = sanitized.strip('_ ')
    # Truncate to 200 chars
    return sanitized[:200]


def cleanup_temp_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
