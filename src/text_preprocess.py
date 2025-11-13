import re
import nltk
import json
import os
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)             # Remove extra spaces
    text = text.lower()                          # Lowercase
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Load transcript
input_path = "data/transcript.txt"
output_cleaned = "data/cleaned_transcript.txt"
output_chunks = "data/chunks.json"

with open(input_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Clean the text
cleaned_text = clean_text(raw_text)

# Save cleaned transcript
with open(output_cleaned, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

# Split into chunks
def chunk_text(text, max_words=250):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield ' '.join(words[i:i+max_words])

chunks = list(chunk_text(cleaned_text))

# Save chunks as JSON
with open(output_chunks, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Cleaned transcript saved at {output_cleaned}")
print(f"✅ Split into {len(chunks)} chunks and saved at {output_chunks}")
