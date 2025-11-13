# import os
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# import evaluate

# # ---------------------------
# # Load transcript
# # ---------------------------
# transcript_path = "data/transcript.txt"
# if not os.path.exists(transcript_path):
#     raise FileNotFoundError("Transcript not found. Run ASR first (Day 1).")

# with open(transcript_path, "r", encoding="utf-8") as f:
#     text = f.read()

# # ---------------------------
# # Chunk long text for summarization
# # ---------------------------
# max_chunk = 500
# sentences = text.split(". ")
# chunks = []
# current_chunk = []

# for sentence in sentences:
#     if sum(len(s.split(" ")) for s in current_chunk) + len(sentence.split(" ")) <= max_chunk:
#         current_chunk.append(sentence)
#     else:
#         chunks.append(". ".join(current_chunk))
#         current_chunk = [sentence]

# if current_chunk:
#     chunks.append(". ".join(current_chunk))

# print(f"📘 Transcript split into {len(chunks)} chunks")

# # ---------------------------
# # Load T5 model and tokenizer
# # ---------------------------
# model_name = "t5-small"  # Fast and CPU-friendly
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# # ---------------------------
# # Summarize each chunk
# # ---------------------------
# summaries = []
# for i, chunk in enumerate(chunks):
#     input_text = "summarize: " + chunk
#     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
#     summary_ids = model.generate(inputs, max_length=120, min_length=30, length_penalty=2.0, num_beams=4)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     summaries.append(summary)
#     print(f"✅ Summarized chunk {i+1}/{len(chunks)}")

# # ---------------------------
# # Combine summaries
# # ---------------------------
# final_summary = " ".join(summaries)

# os.makedirs("data", exist_ok=True)
# with open("data/summary.txt", "w", encoding="utf-8") as f:
#     f.write(final_summary)

# print("\n💾 Summary saved to data/summary.txt")
# print("\n🧠 Final Summary Preview:\n", final_summary[:600], "...")


# # ---------------------------
# # Evaluate summary using ROUGE
# # ---------------------------
# rouge = evaluate.load("rouge")
# results = rouge.compute(predictions=[final_summary], references=[text])

# print("\n📊 Evaluation Metrics:")
# for k, v in results.items():
#     print(f"{k}: {v:.4f}")

# with open("data/evaluation.json", "w") as f:
#     import json
#     json.dump(results, f, indent=2)
# print("💾 Saved evaluation metrics to data/evaluation.json")


import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import evaluate

# -------------------------
# 🧹 Step 1: Auto-clean corrupted downloads
# -------------------------
def clean_incomplete_checkpoints(model_name):
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        print("⚠️ Hugging Face cache directory not found.")
        return

    for folder in cache_dir.glob(f"models--{model_name.replace('/', '--')}*"):
        print(f"🔍 Checking cache folder: {folder}")
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".incomplete") or f.endswith(".lock"):
                    print(f"🗑️ Removing incomplete file: {os.path.join(root, f)}")
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception as e:
                        print(f"⚠️ Could not remove {f}: {e}")
        # Remove corrupted snapshots with missing .bin files
        for subfolder in folder.glob("snapshots/*"):
            if not any(p.suffix == ".bin" for p in subfolder.glob("*")):
                print(f"🧹 Removing corrupted snapshot: {subfolder}")
                shutil.rmtree(subfolder, ignore_errors=True)

# -------------------------
# 🚀 Step 2: Safe model + tokenizer loader
# -------------------------
def load_model_safely(model_name="t5-base"):
    print(f"🚀 Checking cache integrity for {model_name}...")
    clean_incomplete_checkpoints(model_name)

    print("⬇️ Downloading verified model files (if missing)...")
    snapshot_download(model_name, force_download=False)

    print("✅ Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print("🎯 Model loaded successfully!\n")
    return tokenizer, model

# -------------------------
# 🧠 Step 3: Summarization Logic
# -------------------------
def summarize_text(text, model, tokenizer, max_input_length=1024, max_output_length=256):
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    )
    summary_ids = model.generate(
        inputs,
        max_length=max_output_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -------------------------
# 🧪 Step 4: Main
# -------------------------
if __name__ == "__main__":
    model_name = "t5-base"
    tokenizer, model = load_model_safely(model_name)

    # Example input text
    text = """
    Artificial Intelligence (AI) is transforming industries by automating processes, 
    improving decision-making, and creating new opportunities for innovation. 
    However, it also raises concerns about privacy, bias, and job displacement.
    """

    print("📝 Original Text:\n", text)
    summary = summarize_text(text, model, tokenizer)
    print("\n✨ Generated Summary:\n", summary)

    # Optional: Evaluate with ROUGE
    try:
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=[summary], references=[text])
        print("\n📊 ROUGE Evaluation Metrics:\n", results)
    except Exception as e:
        print(f"⚠️ ROUGE evaluation skipped: {e}")
