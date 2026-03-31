import multiprocessing
import os
import sys
import logging
import click
from whisper_cli import __version__
from whisper_cli.transcriber import check_ffmpeg, load_model, transcribe_file
from whisper_cli.formatter import (
    format_txt, format_srt, format_vtt,
    format_json, format_csv, format_md,
)
from whisper_cli.downloader import (
    is_youtube_url, download_audio, cleanup_temp_dir, sanitize_filename,
)

FORMATTERS = {
    "txt": format_txt,
    "srt": format_srt,
    "vtt": format_vtt,
    "json": format_json,
    "csv": format_csv,
    "md": format_md,
}

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]


@click.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option("-m", "--model", default="base", type=click.Choice(MODEL_CHOICES), help="Whisper model size.")
@click.option("-o", "--output", default=None, type=click.Path(), help="Output directory.")
@click.option("-f", "--format", "fmt", default="txt", type=click.Choice(list(FORMATTERS.keys())), help="Output format.")
@click.option("--stdout", is_flag=True, help="Print to stdout instead of writing files.")
@click.option("-l", "--language", default="en", help="Source language (ISO 639-1 code, or \"auto\" to detect).")
@click.option("-q", "--quiet", is_flag=True, help="Suppress Whisper progress logging.")
@click.version_option(version=__version__)
def main(inputs, model, output, fmt, stdout, language, quiet):
    """Transcribe audio/video files or YouTube URLs to text using OpenAI Whisper."""
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

    # Create output directory if it doesn't exist
    if output:
        os.makedirs(output, exist_ok=True)
        if not os.access(output, os.W_OK):
            click.echo(f"Error: Output directory not writable: {output}", err=True)
            sys.exit(2)

    # Resolve inputs: classify as YouTube URLs or local files
    resolved = []   # list of (display_name, file_path)
    temp_dirs = []  # track temp dirs for cleanup
    failed = 0

    for item in inputs:
        if is_youtube_url(item):
            try:
                click.echo(f"Downloading: {item}", err=True)
                temp_dir, audio_path, title = download_audio(item, quiet=True)
                temp_dirs.append(temp_dir)
                resolved.append((sanitize_filename(title), audio_path))
            except Exception as e:
                click.echo(f"Error downloading {item}: {e}", err=True)
                failed += 1
        elif os.path.isfile(item):
            resolved.append((os.path.splitext(os.path.basename(item))[0], item))
        else:
            click.echo(f"Error: File not found: {item}", err=True)
            failed += 1

    if not resolved:
        click.echo("Error: No valid inputs.", err=True)
        for td in temp_dirs:
            cleanup_temp_dir(td)
        sys.exit(2)

    if quiet:
        logging.getLogger("whisper").setLevel(logging.ERROR)

    whisper_model = load_model(model)
    formatter = FORMATTERS[fmt]

    succeeded = 0
    current_output_path = None

    try:
        for display_name, file_path in resolved:
            try:
                click.echo(f"Transcribing: {display_name}", err=True)
                lang = None if language == "auto" else language
                segments = transcribe_file(whisper_model, file_path, language=lang)
                formatted = formatter(segments)

                if stdout:
                    if len(resolved) > 1:
                        click.echo(f"=== {display_name} ===")
                    click.echo(formatted)
                else:
                    out_dir = output or os.path.dirname(os.path.abspath(file_path))
                    current_output_path = os.path.join(out_dir, f"{display_name}.{fmt}")
                    with open(current_output_path, "w") as out_file:
                        out_file.write(formatted)
                    current_output_path = None
                    click.echo(f"  -> {os.path.join(out_dir, display_name + '.' + fmt)}", err=True)

                succeeded += 1
            except Exception as e:
                click.echo(f"Error processing {display_name}: {e}", err=True)
                failed += 1
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        if current_output_path and os.path.exists(current_output_path):
            os.remove(current_output_path)
        sys.exit(2)
    finally:
        for td in temp_dirs:
            cleanup_temp_dir(td)

    if failed > 0 and succeeded > 0:
        sys.exit(1)
    elif failed > 0 and succeeded == 0:
        sys.exit(2)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
