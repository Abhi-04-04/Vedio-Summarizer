import streamlit as st
import os, json
from PIL import Image

st.set_page_config(page_title="AI Video Summarizer", layout="wide")

st.title("🎥 AI-Powered Video & Document Summarization")
st.write("This app uses **Deep Learning & NLP** to summarize long-form videos and documents.")

# --------------------------
# Upload Section
# --------------------------
st.sidebar.header("📤 Upload Video")
uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "mkv", "avi"])

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

if uploaded_file is not None:
    video_path = os.path.join(data_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.video(video_path)
    st.success("✅ Video uploaded successfully!")

st.divider()

# --------------------------
# Transcript Display
# --------------------------
transcript_path = os.path.join(data_dir, "transcript.txt")
if os.path.exists(transcript_path):
    st.subheader("🗒️ Transcript")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    st.text_area("Transcript", transcript, height=250)
else:
    st.warning("Transcript not found. Please run ASR (Day 1).")

# --------------------------
# Keyframes Display
# --------------------------
keyframe_dir = os.path.join(data_dir, "keyframes")
if os.path.exists(keyframe_dir) and len(os.listdir(keyframe_dir)) > 0:
    st.subheader("🖼️ Keyframes")
    cols = st.columns(3)
    images = [os.path.join(keyframe_dir, f) for f in sorted(os.listdir(keyframe_dir)) if f.endswith(".jpg")]
    for i, img_path in enumerate(images):
        with cols[i % 3]:
            st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
else:
    st.info("No keyframes available. Please run keyframe extraction (Day 3).")

# --------------------------
# Insights Display
# --------------------------
insights_path = os.path.join(data_dir, "insights.json")
if os.path.exists(insights_path):
    st.subheader("📊 NLP Insights")
    with open(insights_path, "r", encoding="utf-8") as f:
        insights = json.load(f)
    st.write(f"**Sentiment:** {insights['sentiment']} ({insights['score']:.2f})")

    st.write("**Top Keywords:**")
    st.write(", ".join(insights["keywords"]))
else:
    st.warning("Insights not found. Please run text_insights.py (Day 4).")

# --------------------------
# Summarization Display
# --------------------------
summary_path = os.path.join(data_dir, "summary.txt")
if os.path.exists(summary_path):
    st.subheader("🧠 Auto-Generated Summary")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = f.read()
    st.success(summary)
    st.download_button("📥 Download Summary", summary, file_name="summary.txt")
else:
    if st.button("🚀 Generate Summary"):
        with st.spinner("Generating summary using T5 model... (may take 1–2 min)"):
            os.system("python src/summarizer_t5.py")
        st.rerun()

