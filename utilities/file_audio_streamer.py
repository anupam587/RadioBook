"""
File Audio to Streaming Audio
================================
Reads an audio file and streams it in real-time over HTTP.

Supports: WAV, MP3, FLAC, OGG, AAC, M4A, and most common formats (via pydub).

Requirements:
    pip install flask pydub
    # Also install ffmpeg for non-WAV formats:
    # macOS:   brew install ffmpeg
    # Linux:   sudo apt install ffmpeg
    # Windows: https://ffmpeg.org/download.html

Usage:
    python file_audio_streamer.py --file path/to/audio.mp3
    python file_audio_streamer.py --file path/to/audio.wav --port 8080
    python file_audio_streamer.py --file path/to/audio.mp3 --loop
    python file_audio_streamer.py --file path/to/audio.mp3 --chunk-size 4096

Once running, open http://localhost:5000 in your browser to play the stream,
or point any media player to http://localhost:5000/stream
"""

import argparse
import os
import struct
import sys
import time
import wave

# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream an audio file over HTTP in real time."
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the input audio file (WAV, MP3, FLAC, OGG, etc.)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="HTTP port to serve the stream on (default: 5000)"
    )
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        help="Loop the audio file continuously"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=2048,
        help="Audio chunk size in bytes per stream packet (default: 2048)"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="Override output sample rate in Hz (e.g. 44100). Defaults to file's native rate."
    )
    return parser.parse_args()


# ── Audio Loading ─────────────────────────────────────────────────────────────

def load_audio_as_wav_bytes(filepath: str, target_rate: int = None) -> tuple[bytes, int, int, int]:
    """
    Load any audio file and return raw WAV bytes plus metadata.

    Returns:
        (raw_pcm_bytes, sample_rate, channels, sample_width_bytes)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".wav" and target_rate is None:
        # Fast path: read WAV directly without pydub
        with wave.open(filepath, "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            pcm = wf.readframes(wf.getnframes())
        print(f"  Format   : WAV (native)")
        print(f"  Rate     : {rate} Hz")
        print(f"  Channels : {channels}")
        print(f"  Bit depth: {sampwidth * 8}-bit")
        return pcm, rate, channels, sampwidth

    # Use pydub for everything else (MP3, FLAC, OGG, M4A, AAC …)
    try:
        from pydub import AudioSegment
    except ImportError:
        sys.exit(
            "pydub is required for non-WAV formats.\n"
            "Install it with:  pip install pydub\n"
            "Also make sure ffmpeg is installed on your system."
        )

    print(f"  Decoding with pydub / ffmpeg…")
    audio = AudioSegment.from_file(filepath)

    if target_rate:
        audio = audio.set_frame_rate(target_rate)

    rate     = audio.frame_rate
    channels = audio.channels
    sampwidth = audio.sample_width   # bytes per sample

    print(f"  Format   : {ext.lstrip('.')}")
    print(f"  Rate     : {rate} Hz")
    print(f"  Channels : {channels}")
    print(f"  Bit depth: {sampwidth * 8}-bit")
    print(f"  Duration : {len(audio) / 1000:.2f}s")

    pcm = audio.raw_data
    return pcm, rate, channels, sampwidth


# ── WAV Header Builder ────────────────────────────────────────────────────────

def make_wav_header(rate: int, channels: int, sampwidth: int) -> bytes:
    """
    Build a WAV header manually using struct.pack.
    Data chunk size is set to 0xFFFFFFFF (max uint32) to signal an
    open-ended / streaming file — safe on all Python versions.
    """
    import struct

    data_size    = 0xFFFFFFFF          # unknown / streaming length
    byte_rate    = rate * channels * sampwidth
    block_align  = channels * sampwidth
    bits         = sampwidth * 8
    riff_size    = 0xFFFFFFFF          # also mark RIFF chunk as open-ended

    header = struct.pack(
        "<4sI4s"     # RIFF chunk descriptor
        "4sIHHIIHH"  # fmt  sub-chunk
        "4sI",       # data sub-chunk header
        # RIFF
        b"RIFF", riff_size, b"WAVE",
        # fmt
        b"fmt ", 16, 1,          # PCM = audio format 1
        channels, rate,
        byte_rate, block_align,
        bits,
        # data
        b"data", data_size,
    )
    return header


# ── Streaming Generator ───────────────────────────────────────────────────────

def audio_generator(pcm: bytes, rate: int, channels: int, sampwidth: int,
                    chunk_size: int, loop: bool):
    """
    Yield WAV header once, then stream PCM data in timed chunks.
    Timing is calculated so chunks flow at real-time audio speed.
    """
    yield make_wav_header(rate, channels, sampwidth)

    bytes_per_second = rate * channels * sampwidth
    sleep_time = chunk_size / bytes_per_second   # real-time pacing

    while True:
        offset = 0
        while offset < len(pcm):
            chunk = pcm[offset: offset + chunk_size]
            yield chunk
            offset += chunk_size
            time.sleep(sleep_time)   # pace the stream in real time

        if not loop:
            break
        print("[Streamer] File ended — looping…")


# ── Flask HTTP Server ─────────────────────────────────────────────────────────

def run_server(filepath: str, port: int, loop: bool,
               chunk_size: int, target_rate: int):

    try:
        from flask import Flask, Response, request
    except ImportError:
        sys.exit("Flask is required. Install it with:  pip install flask")

    app = Flask(__name__)

    # ── Load the file once at startup ──
    print(f"\n[Loading] {filepath}")
    pcm, rate, channels, sampwidth = load_audio_as_wav_bytes(filepath, target_rate)
    duration_s = len(pcm) / (rate * channels * sampwidth)
    print(f"  PCM size : {len(pcm):,} bytes")
    print(f"  Duration : {duration_s:.2f}s")
    print()

    # ── Routes ──

    @app.route("/")
    def index():
        filename = os.path.basename(filepath)
        loop_label = "on" if loop else "off"
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>🎵 Audio Stream</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 100vh; margin: 0;
      background: #0f0f13; color: #e0e0e0;
    }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.4rem; }}
    p  {{ color: #888; font-size: 0.9rem; margin: 0.2rem 0; }}
    audio {{ margin-top: 1.6rem; width: 420px; }}
    a {{ color: #7c9cff; }}
  </style>
</head>
<body>
  <h1>🎵 {filename}</h1>
  <p>Duration: {duration_s:.1f}s &nbsp;|&nbsp; {rate} Hz, {channels}ch &nbsp;|&nbsp; Loop: {loop_label}</p>
  <audio controls autoplay src="/stream">
    Your browser does not support the audio element.
  </audio>
  <p style="margin-top:1rem">
    Direct stream URL: <a href="/stream">/stream</a>
  </p>
</body>
</html>"""

    @app.route("/stream")
    def stream():
        def generate():
            yield from audio_generator(pcm, rate, channels, sampwidth, chunk_size, loop)

        return Response(
            generate(),
            mimetype="audio/wav",
            headers={
                # Prevent buffering by proxies / browsers
                "Cache-Control": "no-cache, no-store",
                "X-Content-Type-Options": "nosniff",
                "Transfer-Encoding": "chunked",
            },
        )

    @app.route("/info")
    def info():
        return {
            "file": os.path.basename(filepath),
            "sample_rate": rate,
            "channels": channels,
            "bit_depth": sampwidth * 8,
            "duration_seconds": round(duration_s, 3),
            "loop": loop,
            "chunk_size": chunk_size,
        }

    # ── Start ──
    print(f"[Server] Streaming '{os.path.basename(filepath)}'")
    print(f"         Browser player : http://localhost:{port}/")
    print(f"         Raw stream URL : http://localhost:{port}/stream")
    print(f"         JSON info      : http://localhost:{port}/info")
    print(f"\n         Press Ctrl+C to stop.\n")

    app.run(host="0.0.0.0", port=port, threaded=True)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Error: File not found: {args.file}")

    run_server(
        filepath=args.file,
        port=args.port,
        loop=args.loop,
        chunk_size=args.chunk_size,
        target_rate=args.rate,
    )
