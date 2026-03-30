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
