# src/asr_whisper.py
import whisper
import json
from pathlib import Path
import os
import time

def transcribe_whisper(audio_path, model_name="small", language=None, output_json="data/transcript.json", output_txt="data/transcript.txt"):
    """
    Transcribe audio with whisper and save segments + full text.
    - audio_path: path to input wav
    - model_name: "tiny", "base", "small", "medium", "large" (small is a good accuracy/speed balance)
    - language: optional language code (e.g., "en") — whisper will auto-detect if None
    - output_json: path to save detailed output including segments
    - output_txt: plain text transcript
    """
    start = time.time()
    print(f"Loading Whisper model '{model_name}' (this can take a while on CPU)...")
    model = whisper.load_model(model_name)  # CPU by default unless CUDA available
    opts = {"language": language} if language else {}
    print("Starting transcription (this can be slow on CPU).")
    result = model.transcribe(str(audio_path), **opts)
    elapsed = time.time() - start
    print(f"Transcription finished in {elapsed:.1f} sec (wall time).")
    # result contains 'text' and 'segments' (list of {start,end,text} if model provides)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(result.get("text", ""))
    print(f"Saved JSON -> {output_json}")
    print(f"Saved plain text -> {output_txt}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/asr_whisper.py data/example.wav [model_name]")
        sys.exit(1)
    audio = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "small"
    transcribe_whisper(audio, model_name=model)
