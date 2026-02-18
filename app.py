import streamlit as st
from transformers import pipeline
import pandas as pd

# ---------------- Page Config ---------------- #
st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="🎭",
    layout="centered"
)

# ---------------- Title ---------------- #
st.title("🎭 Emotion Detection System")
st.write("Detect human emotions from text using a pretrained Transformer model.")

# ---------------- Load Model (Cached) ---------------- #
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

classifier = load_model()

# ---------------- Emoji Dictionary ---------------- #
emoji_dict = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# ---------------- User Input ---------------- #
user_input = st.text_area("Enter your text here:", height=150)

# ---------------- Detect Button ---------------- #
if st.button("🔍 Detect Emotion"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        try:
            with st.spinner("Analyzing emotion..."):
                results = classifier(user_input)

            # Handle nested list output safely
            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]

            # Sort by confidence
            results = sorted(results, key=lambda x: x["score"], reverse=True)

            # Get top emotion
            top_emotion = results[0]["label"].lower()
            confidence = results[0]["score"] * 100

            # ---------------- Display Result ---------------- #
            st.subheader("🎯 Predicted Emotion")
            st.success(f"{top_emotion.upper()} {emoji_dict.get(top_emotion, '')}")
            st.write(f"Confidence Score: {round(confidence, 2)} %")

            # ---------------- Chart ---------------- #
            df = pd.DataFrame(results)
            df["score"] = df["score"] * 100

            st.subheader("📊 Emotion Probability Distribution")
            st.bar_chart(df.set_index("label"))

        except Exception as e:
            st.error("Something went wrong while processing the text.")
            st.error(str(e))
