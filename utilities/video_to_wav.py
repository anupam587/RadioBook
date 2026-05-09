#!/usr/bin/env python3
"""
Video to WAV Audio Converter
Converts a video from an HTTP URL to a WAV audio file.

Requirements:
    pip install yt-dlp moviepy requests

Usage:
    python video_to_wav.py <video_url> [output_filename]

Examples:
    python video_to_wav.py https://example.com/video.mp4
    python video_to_wav.py https://example.com/video.mp4 my_audio.wav
    python video_to_wav.py https://www.youtube.com/watch?v=xxxxx output.wav
"""

import sys
import os
import tempfile
import argparse
import urllib.request
from pathlib import Path


def download_and_convert_with_moviepy(url: str, output_path: str) -> bool:
    """
    Download video from URL and extract audio using moviepy.
    Works best for direct video file URLs (e.g., .mp4, .mkv, .avi).
    """
    try:
        from moviepy.editor import VideoFileClip
        import requests

        print(f"[1/3] Downloading video from: {url}")

        # Download the video to a temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

        print(f"\n[2/3] Extracting audio...")

        # Extract audio using moviepy
        with VideoFileClip(tmp_path) as clip:
            if clip.audio is None:
                raise ValueError("The video file has no audio track.")
            clip.audio.write_audiofile(
                output_path,
                fps=44100,       # 44.1 kHz sample rate
                nbytes=2,        # 16-bit depth
                codec="pcm_s16le",
                logger=None
            )

        os.unlink(tmp_path)
        print(f"[3/3] Done! Audio saved to: {output_path}")
        return True

    except ImportError:
        print("  moviepy not available, trying next method...")
        return False
    except Exception as e:
        print(f"  moviepy method failed: {e}")
        return False


def download_and_convert_with_ytdlp(url: str, output_path: str) -> bool:
    """
    Download and extract audio using yt-dlp.
    Works for YouTube, Vimeo, and 1000+ other platforms.
    """
    try:
        import yt_dlp

        print(f"[1/2] Downloading and extracting audio from: {url}")

        base_path = str(Path(output_path).with_suffix(""))

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": base_path + ".%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",  # best quality
                }
            ],
            "quiet": False,
            "no_warnings": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp may name the file differently; find it
        wav_file = base_path + ".wav"
        if os.path.exists(wav_file) and wav_file != output_path:
            os.rename(wav_file, output_path)

        print(f"[2/2] Done! Audio saved to: {output_path}")
        return True

    except ImportError:
        print("  yt-dlp not available, trying next method...")
        return False
    except Exception as e:
        print(f"  yt-dlp method failed: {e}")
        return False


def download_and_convert_with_ffmpeg(url: str, output_path: str) -> bool:
    """
    Use ffmpeg subprocess directly to stream and convert.
    Requires ffmpeg installed on the system.
    """
    import subprocess
    import shutil

    if not shutil.which("ffmpeg"):
        print("  ffmpeg not found in PATH.")
        return False

    try:
        print(f"[1/2] Streaming and converting with ffmpeg: {url}")

        cmd = [
            "ffmpeg",
            "-y",                   # overwrite output
            "-i", url,              # input URL (ffmpeg handles HTTP natively)
            "-vn",                  # no video
            "-acodec", "pcm_s16le", # WAV codec (16-bit PCM)
            "-ar", "44100",         # 44.1 kHz sample rate
            "-ac", "2",             # stereo
            output_path
        ]

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            print(f"[2/2] Done! Audio saved to: {output_path}")
            return True
        else:
            print(f"  ffmpeg exited with code {result.returncode}")
            return False

    except Exception as e:
        print(f"  ffmpeg method failed: {e}")
        return False


def convert_video_url_to_wav(url: str, output_path: str = "output.wav") -> str:
    """
    Convert a video at the given HTTP URL to a WAV audio file.

    Tries multiple backends in order:
      1. yt-dlp  — best for YouTube/Vimeo/social media URLs
      2. moviepy — best for direct .mp4/.mkv/.avi links
      3. ffmpeg  — fallback using system ffmpeg binary

    Args:
        url: HTTP/HTTPS URL pointing to a video.
        output_path: Desired output .wav file path.

    Returns:
        Absolute path to the created WAV file.

    Raises:
        RuntimeError: If all conversion methods fail.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL must start with http:// or https://. Got: {url}")

    output_path = str(Path(output_path).with_suffix(".wav"))
    print(f"\n{'='*55}")
    print(f"  Video → WAV Converter")
    print(f"{'='*55}")
    print(f"  Input  : {url}")
    print(f"  Output : {output_path}")
    print(f"{'='*55}\n")

    # Try each method in order of preference
    methods = [
        ("yt-dlp",   lambda: download_and_convert_with_ytdlp(url, output_path)),
        ("moviepy",  lambda: download_and_convert_with_moviepy(url, output_path)),
        ("ffmpeg",   lambda: download_and_convert_with_ffmpeg(url, output_path)),
    ]

    for name, fn in methods:
        print(f"→ Trying method: {name}")
        if fn():
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n✅ Success!")
            print(f"   File : {os.path.abspath(output_path)}")
            print(f"   Size : {size_mb:.2f} MB")
            return os.path.abspath(output_path)
        print()

    raise RuntimeError(
        "All conversion methods failed.\n"
        "Please ensure at least one of the following is installed:\n"
        "  pip install yt-dlp\n"
        "  pip install moviepy requests\n"
        "  (or install ffmpeg system-wide)"
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a video from an HTTP URL to a WAV audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="HTTP/HTTPS URL of the video")
    parser.add_argument(
        "output",
        nargs="?",
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    args = parser.parse_args()

    try:
        result = convert_video_url_to_wav(args.url, args.output)
        print(f"\nOutput file: {result}")
    except (ValueError, RuntimeError) as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
