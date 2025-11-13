# import json
# import nltk
# from textblob import TextBlob
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np

# # download both sentence tokenizer + new punkt_tab resource
# nltk.download('punkt')
# nltk.download('punkt_tab')


# # --- Load transcript ---
# with open("data/transcript.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# # --- Sentiment analysis ---
# blob = TextBlob(text)
# sentiment = blob.sentiment.polarity

# if sentiment > 0.05:
#     sentiment_label = "positive"
# elif sentiment < -0.05:
#     sentiment_label = "negative"
# else:
#     sentiment_label = "neutral"

# # --- Topic extraction using TF-IDF + KMeans ---
# sentences = nltk.sent_tokenize(text)
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(sentences)

# num_topics = min(3, len(sentences))  # Keep 3 main themes
# kmeans = KMeans(n_clusters=num_topics, random_state=42)
# kmeans.fit(X)

# topics = []
# for i in range(num_topics):
#     cluster_indices = np.where(kmeans.labels_ == i)[0]
#     cluster_sentences = [sentences[j] for j in cluster_indices]
#     topic_keywords = [vectorizer.get_feature_names_out()[idx]
#                       for idx in kmeans.cluster_centers_[i].argsort()[-5:]]
#     topics.append({
#         "topic": f"Topic {i+1}",
#         "keywords": topic_keywords,
#         "example_sentence": cluster_sentences[0] if cluster_sentences else ""
#     })

# # --- Save output ---
# insights = {
#     "sentiment": sentiment_label,
#     "sentiment_score": sentiment,
#     "topics": topics
# }

# with open("data/insights.json", "w", encoding="utf-8") as f:
#     json.dump(insights, f, indent=4)

# print("✅ Sentiment & topics extracted successfully!")
# print(f"Sentiment: {sentiment_label} ({sentiment:.2f})")


##NEw code to check

import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter

# -------------------------
# 🧠 Safe NLTK downloads (runs only once)
# -------------------------
def safe_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)

safe_download('tokenizers/punkt')
safe_download('tokenizers/punkt_tab')
safe_download('sentiment/vader_lexicon')
safe_download('corpora/stopwords')

# -------------------------
# 📘 Load transcript
# -------------------------
transcript_path = "data/transcript.txt"
if not os.path.exists(transcript_path):
    raise FileNotFoundError("Transcript not found. Run ASR first (Day 1).")

with open(transcript_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# ✂️ Sentence tokenization
# -------------------------
sentences = sent_tokenize(text)

# -------------------------
# 💬 Sentiment analysis
# -------------------------
sia = SentimentIntensityAnalyzer()
sentiment_scores = [sia.polarity_scores(s)["compound"] for s in sentences]
avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
sentiment_label = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"

# -------------------------
# 🔑 Keyword extraction (simple frequency)
# -------------------------
stop_words = set(stopwords.words("english"))
words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
word_freq = Counter(words)
top_keywords = [w for w, _ in word_freq.most_common(10)]

# -------------------------
# 🧾 Output summary insights
# -------------------------
print("\n✅ Text Insights Generated:")
print(f"🔹 Sentiment: {sentiment_label} ({avg_sentiment:.2f})")
print("🔹 Top Keywords:", ", ".join(top_keywords))

# Save as JSON for later visualization
import json
insights = {"sentiment": sentiment_label, "score": avg_sentiment, "keywords": top_keywords}
os.makedirs("data", exist_ok=True)
with open("data/insights.json", "w", encoding="utf-8") as f:
    json.dump(insights, f, indent=2)

print("\n💾 Saved insights to data/insights.json")
