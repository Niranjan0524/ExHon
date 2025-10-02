import os, sys, shutil, warnings, time
import numpy as np
import sounddevice as sd
import wavio
import librosa
from scipy.io import wavfile

# Optional: Whisper transcription (works without FFmpeg if we pass a NumPy array)
try:
    import whisper
    HAVE_WHISPER = True
except Exception as _:
    HAVE_WHISPER = False

# Speaker diarization (requires HF token)
from pyannote.audio import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------- CONFIG -------------------
DURATION_SEC = 10
TARGET_SR = 16000
INPUT_DEVICE_HINT = None  # e.g., "Microphone (Realtek)" or device index like 3; set to None to use default
APPLY_AUTOGAIN = True
TARGET_RMS = 0.1          # target loudness for autogain
MAX_GAIN = 20.0           # cap gain to avoid crazy amplification
CHEATING_KEYWORDS = ["answer", "option", "solve", "google", "tell me", "what is"]
# ----------------------------------------------

def list_input_devices():
    devices = sd.query_devices()
    print("\n🎤 Available audio input devices:")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  (in: {d['max_input_channels']}, sr≈{int(d.get('default_samplerate') or 0)})")
    print()

def pick_input_device(hint=None):
    if hint is None:
        return None  # use default
    devices = sd.query_devices()
    if isinstance(hint, int):
        return hint
    hint_lower = hint.lower()
    candidates = [i for i, d in enumerate(devices)
                  if d["max_input_channels"] > 0 and hint_lower in d["name"].lower()]
    return candidates[0] if candidates else None

def record_audio(duration=DURATION_SEC, sr=TARGET_SR, device_hint=INPUT_DEVICE_HINT):
    list_input_devices()
    idx = pick_input_device(device_hint)
    if idx is not None:
        sd.default.device = (idx, None)  # (input, output)
        print(f"✅ Using input device index {idx}")
    else:
        print("ℹ️ Using system default input device")

    print(f"🎙 Recording {duration}s @ {sr}Hz (mono)...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    print("✅ Recording complete")

    # Basic level report
    mx = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    print(f"🔊 Level check — max: {mx:.4f}, rms: {rms:.4f}")

    if APPLY_AUTOGAIN and rms > 0 and rms < TARGET_RMS/5:
        gain = min(MAX_GAIN, TARGET_RMS / rms)
        audio = np.clip(audio * gain, -1.0, 1.0)
        mx2 = float(np.max(np.abs(audio)))
        rms2 = float(np.sqrt(np.mean(audio**2)))
        print(f"✨ Auto-gain applied (x{gain:.1f}) → max: {mx2:.4f}, rms: {rms2:.4f}")
    elif rms == 0:
        print("⚠️ Silence detected. Check mic selection/volume.")

    return audio, sr

def save_wav(audio, sr, path="temp.wav"):
    path = os.path.abspath(path)
    try:
        wavio.write(path, audio, sr, sampwidth=2)
    except Exception as e1:
        print(f"⚠️ wavio write failed: {e1} → trying scipy")
        ai16 = np.int16(np.clip(audio, -1, 1) * 32767)
        wavfile.write(path, sr, ai16)
    size = os.path.getsize(path)
    print(f"💾 Saved WAV: {path} ({size} bytes)")
    return path

def load_diarization_pipeline():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("⚠️ HUGGINGFACE_TOKEN not set → speaker diarization unavailable.")
        return None
    print("🤖 Loading pyannote speaker diarization pipeline...")
    try:
        pl = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
        print("✅ Diarization pipeline ready")
        return pl
    except Exception as e:
        print(f"❌ Failed to load diarization pipeline: {e}")
        return None

def count_speakers_and_timeline(pipeline, wav_path):
    if pipeline is None:
        return 1, []

    diarization = pipeline(wav_path)
    segments = []
    speakers = set()
    for segment, _, label in diarization.itertracks(yield_label=True):
        speakers.add(label)
        segments.append((float(segment.start), float(segment.end), label))
    segments.sort(key=lambda x: x[0])
    return len(speakers), segments

def whisper_transcribe_numpy(audio, sr):
    if not HAVE_WHISPER:
        print("ℹ️ Whisper not installed; skipping transcription.")
        return ""
    try:
        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        model = whisper.load_model("base")
        print("🤖 Whisper model loaded")
        # If audio is too quiet, Whisper may return empty; we already autogained above.
        result = model.transcribe(audio.astype(np.float32), fp16=False)
        text = (result.get("text") or "").strip()
        if not text:
            print("⚠️ Whisper returned empty text (possibly silence or too noisy).")
        return text
    except Exception as e:
        print(f"⚠️ Whisper (NumPy) failed: {e}")
        return ""

def whisper_transcribe_file_if_ffmpeg(path):
    if not HAVE_WHISPER:
        return ""
    if shutil.which("ffmpeg") is None:
        print("ℹ️ FFmpeg not found; skipping file-based transcription.")
        return ""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(path)
        return (result.get("text") or "").strip()
    except Exception as e:
        print(f"⚠️ Whisper (file) failed: {e}")
        return ""

def has_cheating_keywords(text):
    low = text.lower()
    return any(k in low for k in CHEATING_KEYWORDS)

def pretty_time(s):
    return f"{s:6.2f}s"

def main():
    wav_path = None
    try:
        # 1) Record
        audio, sr = record_audio()

        # 2) Save (for diarization)
        wav_path = save_wav(audio, sr, "temp.wav")

        # 3) Speaker diarization
        pipeline = load_diarization_pipeline()
        n_speakers, segments = count_speakers_and_timeline(pipeline, wav_path)
        print(f"\n🔊 Speakers detected: {n_speakers}")
        if segments:
            print("🗂  Timeline:")
            for start, end, spk in segments:
                print(f"   {pretty_time(start)} → {pretty_time(end)}  | {spk}")

        # 4) Transcription (robust path first: NumPy)
        print("\n🎧 Transcribing (NumPy path)...")
        text = whisper_transcribe_numpy(audio, sr)

        # 5) If empty, try file-based only if FFmpeg is present
        if not text:
            print("🔁 Trying file-based transcription (requires FFmpeg)...")
            text = whisper_transcribe_file_if_ffmpeg(wav_path)

        if text:
            print(f"\n📝 Transcribed text:\n{text}")
            if has_cheating_keywords(text):
                print("🚨 Cheating keywords detected.")
            else:
                print("✅ No suspicious keywords found.")
        else:
            print("\n⚠️ No text transcribed (silence, very low level, or recognition failure).")

        print("\n✅ Analysis complete.")
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                print(f"🗑️ Cleaned up {wav_path}")
            except Exception as e:
                print(f"⚠️ Could not remove temp file: {e}")

if __name__ == "__main__":
    main()
