#!/usr/bin/env python3
"""
Audio URL → WAV + Text Transcriber
Downloads a WAV (or any audio) file from an HTTP URL and transcribes it to text.

Requirements:
    pip install requests openai-whisper webrtcvad pydub numpy

    # For faster/better transcription alternatives:
    pip install faster-whisper          # faster local transcription
    pip install SpeechRecognition       # Google/other cloud STT

    # ffmpeg must be installed system-wide (required by whisper and pydub):
    # macOS:   brew install ffmpeg
    # Ubuntu:  sudo apt install ffmpeg
    # Windows: https://ffmpeg.org/download.html

Usage:
    python audio_to_text.py <audio_url> [options]

Examples:
    # Basic — downloads WAV and transcribes
    python audio_to_text.py "https://example.com/audio.wav"

    # Save transcript to a specific file
    python audio_to_text.py "https://example.com/audio.wav" --output transcript.txt

    # Choose whisper model size (tiny/base/small/medium/large)
    python audio_to_text.py "https://example.com/audio.wav" --model medium

    # Use faster-whisper backend
    python audio_to_text.py "https://example.com/audio.wav" --backend faster-whisper

    # Use Google Speech Recognition (free, needs internet)
    python audio_to_text.py "https://example.com/audio.wav" --backend google

    # Skip transcription if no speech detected (default: True)
    python audio_to_text.py "https://example.com/audio.wav" --no-vad

    # Set VAD aggressiveness (0=least, 3=most aggressive filtering)
    python audio_to_text.py "https://example.com/audio.wav" --vad-aggressiveness 2

    # Get a per-second speech timeline of the audio
    python audio_to_text.py "https://example.com/audio.wav" --vad-timeline
"""

import sys
import os
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime


# ── Helpers ────────────────────────────────────────────────────────────────────

def default_output_path(ext: str = "txt") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"transcript_{timestamp}.{ext}"


def download_audio(url: str, dest_path: str) -> str:
    """Download audio from URL to dest_path. Returns dest_path."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed. Run: pip install requests")

    print(f"[1/4] Downloading audio from:\n      {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Progress: {pct:.1f}% ({downloaded/1024/1024:.2f} MB)", end="", flush=True)

    print(f"\n  ✔ Saved to: {dest_path}  ({os.path.getsize(dest_path)/1024/1024:.2f} MB)")
    return dest_path


# ── Voice Activity Detection (VAD) ─────────────────────────────────────────────

class VADResult:
    """Container for VAD analysis results."""

    def __init__(
        self,
        has_speech: bool,
        speech_ratio: float,
        speech_duration_sec: float,
        total_duration_sec: float,
        speech_segments: list,      # list of (start_sec, end_sec) tuples
        timeline: list,             # list of (second, is_speech) per-second
        method: str,
    ):
        self.has_speech = has_speech
        self.speech_ratio = speech_ratio
        self.speech_duration_sec = speech_duration_sec
        self.total_duration_sec = total_duration_sec
        self.speech_segments = speech_segments
        self.timeline = timeline
        self.method = method

    def summary(self) -> str:
        bar_len = 40
        filled = int(self.speech_ratio * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        lines = [
            f"  Method          : {self.method}",
            f"  Has speech      : {'YES ✔' if self.has_speech else 'NO ✘'}",
            f"  Speech ratio    : {self.speech_ratio:.1%}  [{bar}]",
            f"  Speech duration : {self.speech_duration_sec:.1f}s / {self.total_duration_sec:.1f}s total",
            f"  Speech segments : {len(self.speech_segments)}",
        ]
        for i, (s, e) in enumerate(self.speech_segments, 1):
            lines.append(f"    Segment {i:02d}    : {s:.2f}s → {e:.2f}s  ({e - s:.2f}s)")
        return "\n".join(lines)

    def timeline_str(self) -> str:
        """Compact per-second timeline: S=speech, .=silence."""
        if not self.timeline:
            return "  (no timeline available)"
        row = []
        for second, is_speech in self.timeline:
            row.append("S" if is_speech else ".")
        # wrap every 60 chars with timestamps
        chunk_size = 60
        lines = ["  Per-second timeline (S=speech, .=silence):"]
        for i in range(0, len(row), chunk_size):
            chunk = row[i:i + chunk_size]
            ts = f"{i:4d}s"
            lines.append(f"  {ts} |{''.join(chunk)}|")
        return "\n".join(lines)


def _convert_to_pcm_wav(audio_path: str) -> str:
    """
    Convert any audio file to a 16-bit 16kHz mono PCM WAV using pydub+ffmpeg.
    Returns path to the converted temp file (caller must delete it).
    Required because webrtcvad only accepts raw PCM WAV.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub not installed. Run: pip install pydub")

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    tmp = tempfile.NamedTemporaryFile(suffix="_pcm16k.wav", delete=False)
    tmp.close()
    audio.export(tmp.name, format="wav")
    return tmp.name


def _vad_webrtc(audio_path: str, aggressiveness: int = 1) -> VADResult:
    """
    VAD using Google's WebRTC VAD (via webrtcvad package).
    Processes 30ms frames at 16kHz; merges nearby segments.

    aggressiveness: 0 (permissive) … 3 (aggressive / filters more non-speech)
    """
    import struct

    try:
        import webrtcvad
    except ImportError:
        raise RuntimeError("webrtcvad not installed. Run: pip install webrtcvad")

    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub not installed. Run: pip install pydub")

    pcm_path = _convert_to_pcm_wav(audio_path)
    try:
        vad = webrtcvad.Vad(aggressiveness)

        audio = AudioSegment.from_wav(pcm_path)
        sample_rate = audio.frame_rate          # should be 16000
        frame_duration_ms = 30                  # webrtcvad supports 10, 20, 30 ms
        frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
        raw_data = audio.raw_data

        total_samples = len(raw_data) // 2      # 16-bit = 2 bytes per sample
        total_duration = total_samples / sample_rate

        frame_results = []  # list of (timestamp_sec, is_speech)
        offset = 0
        frame_bytes = frame_size * 2            # 2 bytes per 16-bit sample

        while offset + frame_bytes <= len(raw_data):
            frame = raw_data[offset: offset + frame_bytes]
            timestamp = (offset / 2) / sample_rate  # seconds
            try:
                is_speech = vad.is_speech(frame, sample_rate)
            except Exception:
                is_speech = False
            frame_results.append((timestamp, is_speech))
            offset += frame_bytes

        # ── Merge nearby speech frames into segments ────────────────
        # Pad: a silence gap shorter than this is bridged (avoids choppy segments)
        padding_ms = 300
        padding_frames = int(padding_ms / frame_duration_ms)

        segments = []
        triggered = False
        ring_buffer = []
        voiced_frames = []

        for ts, is_speech in frame_results:
            if not triggered:
                ring_buffer.append((ts, is_speech))
                num_voiced = sum(1 for _, s in ring_buffer if s)
                if num_voiced > 0.9 * len(ring_buffer):
                    triggered = True
                    seg_start = ring_buffer[0][0]
                    voiced_frames.extend(ring_buffer)
                    ring_buffer = []
            else:
                voiced_frames.append((ts, is_speech))
                ring_buffer.append((ts, is_speech))
                if len(ring_buffer) > padding_frames:
                    ring_buffer.pop(0)
                num_unvoiced = sum(1 for _, s in ring_buffer if not s)
                if num_unvoiced > 0.9 * len(ring_buffer):
                    triggered = False
                    seg_end = voiced_frames[-1][0] + frame_duration_ms / 1000
                    segments.append((seg_start, seg_end))
                    ring_buffer = []
                    voiced_frames = []

        if triggered and voiced_frames:
            seg_end = voiced_frames[-1][0] + frame_duration_ms / 1000
            segments.append((seg_start, seg_end))

        # ── Per-second timeline ────────────────────────────────────
        speech_by_second: dict = {}
        for ts, is_speech in frame_results:
            sec = int(ts)
            if is_speech:
                speech_by_second[sec] = True
            elif sec not in speech_by_second:
                speech_by_second[sec] = False

        total_secs = int(total_duration) + 1
        timeline = [(s, speech_by_second.get(s, False)) for s in range(total_secs)]

        # ── Summary stats ──────────────────────────────────────────
        speech_duration = sum(e - s for s, e in segments)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0
        has_speech = speech_ratio >= 0.01  # at least 1% speech

        return VADResult(
            has_speech=has_speech,
            speech_ratio=speech_ratio,
            speech_duration_sec=speech_duration,
            total_duration_sec=total_duration,
            speech_segments=segments,
            timeline=timeline,
            method=f"WebRTC VAD (aggressiveness={aggressiveness})",
        )
    finally:
        os.unlink(pcm_path)


def _vad_energy(audio_path: str, threshold_db: float = -40.0) -> VADResult:
    """
    Fallback energy-based VAD using pydub.
    Splits audio into 100ms chunks and checks RMS energy.
    No extra dependencies beyond pydub.

    threshold_db: dBFS below which a chunk is considered silence (default: -40 dBFS)
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub not installed. Run: pip install pydub")

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)

    chunk_ms = 100  # ms per analysis window
    total_duration = len(audio) / 1000.0

    chunks = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i: i + chunk_ms]
        ts = i / 1000.0
        # pydub returns -inf dBFS for completely silent chunks
        rms = chunk.dBFS
        is_speech = rms > threshold_db
        chunks.append((ts, is_speech))

    # ── Merge adjacent speech chunks with small silence gap bridging ──
    min_silence_ms = 300
    min_gap_chunks = int(min_silence_ms / chunk_ms)

    segments = []
    in_segment = False
    seg_start = 0.0
    silence_count = 0

    for ts, is_speech in chunks:
        if is_speech:
            if not in_segment:
                seg_start = ts
                in_segment = True
            silence_count = 0
        else:
            if in_segment:
                silence_count += 1
                if silence_count >= min_gap_chunks:
                    seg_end = ts
                    segments.append((seg_start, seg_end))
                    in_segment = False
                    silence_count = 0

    if in_segment:
        segments.append((seg_start, total_duration))

    # ── Per-second timeline ────────────────────────────────────────
    speech_by_second: dict = {}
    for ts, is_speech in chunks:
        sec = int(ts)
        if is_speech:
            speech_by_second[sec] = True
        elif sec not in speech_by_second:
            speech_by_second[sec] = False

    total_secs = int(total_duration) + 1
    timeline = [(s, speech_by_second.get(s, False)) for s in range(total_secs)]

    speech_duration = sum(e - s for s, e in segments)
    speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0
    has_speech = speech_ratio >= 0.01

    return VADResult(
        has_speech=has_speech,
        speech_ratio=speech_ratio,
        speech_duration_sec=speech_duration,
        total_duration_sec=total_duration,
        speech_segments=segments,
        timeline=timeline,
        method=f"Energy VAD (threshold={threshold_db} dBFS)",
    )


def detect_speech(
    audio_path: str,
    aggressiveness: int = 1,
    show_timeline: bool = False,
) -> VADResult:
    """
    Run Voice Activity Detection on an audio file.
    Tries WebRTC VAD first; falls back to energy-based VAD if webrtcvad is
    not installed.

    Args:
        audio_path    : Path to any audio file (WAV, MP3, etc.)
        aggressiveness: WebRTC VAD aggressiveness 0–3 (ignored for energy VAD)
        show_timeline : Print per-second S/. timeline to stdout

    Returns:
        VADResult with has_speech, speech_ratio, segments, timeline, etc.
    """
    print(f"[2/4] Running Voice Activity Detection...")
    try:
        result = _vad_webrtc(audio_path, aggressiveness=aggressiveness)
    except RuntimeError as e:
        if "webrtcvad" in str(e):
            print(f"  ⚠  webrtcvad not available — falling back to energy-based VAD")
            print(f"     (install with: pip install webrtcvad  for better accuracy)")
            result = _vad_energy(audio_path)
        else:
            raise

    print(result.summary())
    if show_timeline:
        print(result.timeline_str())

    return result


# ── Transcription backends ─────────────────────────────────────────────────────

def transcribe_with_whisper(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe using OpenAI Whisper (runs fully offline/locally).
    Models: tiny, base, small, medium, large  (larger = more accurate, slower)
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError("whisper not installed. Run: pip install openai-whisper")

    print(f"[3/4] Transcribing with Whisper (model: {model_size}) — this may take a moment...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False)
    return result["text"].strip()


def transcribe_with_faster_whisper(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe using faster-whisper (CTranslate2 backend — 2-4x faster than whisper).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

    print(f"[3/4] Transcribing with faster-whisper (model: {model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    print(f"  Detected language: {info.language} (confidence: {info.language_probability:.0%})")

    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip()


def transcribe_with_google(audio_path: str) -> str:
    """
    Transcribe using Google Speech Recognition (free, requires internet).
    Limited to ~60 seconds of audio per request.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        raise RuntimeError("SpeechRecognition not installed. Run: pip install SpeechRecognition")

    print("[3/4] Transcribing with Google Speech Recognition...")
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        print("  Reading audio file...")
        audio_data = recognizer.record(source)

    print("  Sending to Google API...")
    text = recognizer.recognize_google(audio_data)
    return text.strip()


# ── Main orchestrator ──────────────────────────────────────────────────────────

def audio_url_to_text(
    url: str,
    output_dir: str = ".",
    backend: str = "whisper",
    model_size: str = "base",
    use_vad: bool = True,
    vad_aggressiveness: int = 1,
    vad_min_speech_ratio: float = 0.01,
    show_vad_timeline: bool = False,
) -> dict:
    """
    Download audio from URL, optionally run VAD, then transcribe to text.
    Both the WAV file and the transcript .txt are saved to the same directory.

    Args:
        url                  : HTTP/HTTPS URL of the audio file
        output_dir           : Directory where both WAV and TXT will be saved
        backend              : Transcription engine: 'whisper', 'faster-whisper', 'google'
        model_size           : Whisper model size: tiny | base | small | medium | large
        use_vad              : If True, run VAD before transcription; skip if no speech found
        vad_aggressiveness   : WebRTC VAD aggressiveness 0 (permissive) … 3 (aggressive)
        vad_min_speech_ratio : Minimum fraction of audio that must be speech (default 1%)
        show_vad_timeline    : Print per-second S/. timeline after VAD

    Returns:
        dict with keys: transcript, wav_file, text_file, vad (VADResult or None)
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL must start with http:// or https://  Got: {url}")

    base_name = output_dir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(output_dir, base_name + ".wav")
    txt_path = os.path.join(output_dir, base_name + ".txt")

    print(f"\n{'='*60}")
    print(f"  Audio → Text Transcriber")
    print(f"{'='*60}")
    print(f"  Input   : {url}")
    print(f"  Backend : {backend}  (model: {model_size})")
    print(f"  VAD     : {'enabled (aggressiveness=' + str(vad_aggressiveness) + ')' if use_vad else 'disabled'}")
    print(f"  Dir     : {output_dir}")
    print(f"  WAV     : {base_name}.wav")
    print(f"  TXT     : {base_name}.txt")
    print(f"{'='*60}\n")

    # ── Step 1: Download ───────────────────────────────────────────
    download_audio(url, wav_path)

    # ── Step 2: Voice Activity Detection ──────────────────────────
    vad_result = None
    transcript = ""

    if use_vad:
        vad_result = detect_speech(
            wav_path,
            aggressiveness=vad_aggressiveness,
            show_timeline=show_vad_timeline,
        )

        if not vad_result.has_speech or vad_result.speech_ratio < vad_min_speech_ratio:
            print(f"\n⚠  No significant speech detected in the audio.")
            print(f"   Speech ratio ({vad_result.speech_ratio:.1%}) is below threshold "
                  f"({vad_min_speech_ratio:.1%}).")
            print(f"   Proceeding to transcription anyway...\n")
        else:
            print(f"\n  ✔ Speech detected — proceeding to transcription.\n")
    else:
        print(f"[2/4] VAD skipped (--no-vad).\n")

    # ── Step 3: Transcribe ─────────────────────────────────────────
    if backend == "whisper":
        transcript = transcribe_with_whisper(wav_path, model_size)
    elif backend == "faster-whisper":
        transcript = transcribe_with_faster_whisper(wav_path, model_size)
    elif backend == "google":
        transcript = transcribe_with_google(wav_path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose: whisper, faster-whisper, google")

    # ── Step 4: Save transcript ────────────────────────────────────
    print(f"\n[4/4] Saving transcript...")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Source : {url}\n")
        f.write(f"Backend: {backend} / model: {model_size}\n")
        if vad_result:
            f.write(f"VAD    : {vad_result.method} | "
                    f"speech {vad_result.speech_ratio:.1%} of audio\n")
        f.write(f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(transcript)
        f.write("\n")

    print(f"\n{'='*60}")
    print(f"  ✅ Done!")
    print(f"{'='*60}")
    print(f"\n📄 TRANSCRIPT:\n")
    print(transcript)
    print(f"\n{'='*60}")
    print(f"  Output dir : {output_dir}")
    print(f"  WAV file   : {wav_path}")
    print(f"  TXT file   : {txt_path}")
    print(f"{'='*60}\n")

    return {
        "transcript": transcript,
        "wav_file":   wav_path,
        "text_file":  txt_path,
        "vad":        vad_result,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download audio from a URL, detect speech, and transcribe it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="HTTP/HTTPS URL of the audio file (.wav, .mp3, .m4a, etc.)")
    parser.add_argument(
        "--dir", "-d",
        default=".",
        metavar="DIRECTORY",
        dest="output_dir",
        help="Directory to save both WAV and TXT files (default: current directory)",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["whisper", "faster-whisper", "google"],
        default="whisper",
        help="Transcription backend (default: whisper)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        dest="model_size",
        help="Whisper model size (default: base). Larger = more accurate but slower.",
    )

    # ── VAD options ────────────────────────────────────────────────
    vad_group = parser.add_argument_group("Voice Activity Detection (VAD)")
    vad_group.add_argument(
        "--no-vad",
        action="store_true",
        default=False,
        help="Disable VAD check and always transcribe (default: VAD is ON)",
    )
    vad_group.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        metavar="0-3",
        dest="vad_aggressiveness",
        help=(
            "WebRTC VAD aggressiveness level (default: 1). "
            "0 = least aggressive (keeps more audio), "
            "3 = most aggressive (strips more non-speech)."
        ),
    )
    vad_group.add_argument(
        "--vad-min-speech",
        type=float,
        default=0.01,
        metavar="RATIO",
        dest="vad_min_speech_ratio",
        help=(
            "Minimum speech ratio (0.0–1.0) required to proceed with transcription "
            "(default: 0.01 = 1%%). Audio with less speech is skipped."
        ),
    )
    vad_group.add_argument(
        "--vad-timeline",
        action="store_true",
        default=False,
        dest="show_vad_timeline",
        help="Print a per-second speech/silence timeline after VAD analysis.",
    )

    args = parser.parse_args()

    try:
        audio_url_to_text(
            url=args.url,
            output_dir=args.output_dir,
            backend=args.backend,
            model_size=args.model_size,
            use_vad=not args.no_vad,
            vad_aggressiveness=args.vad_aggressiveness,
            vad_min_speech_ratio=args.vad_min_speech_ratio,
            show_vad_timeline=args.show_vad_timeline,
        )
    except (ValueError, RuntimeError) as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
