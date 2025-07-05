import streamlit as st
from transformers import pipeline
import pandas as pd
import pdfplumber
import docx

# Load models once
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

@st.cache_resource
def load_summarizer_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)

analyzer = load_sentiment_model()
summarizer = load_summarizer_model()

# --- UI ---
st.set_page_config(page_title="Review Analyzer", page_icon="🧠", layout="centered")
st.title("🧠 Review Analyzer: Summary + Sentiment for All File Types")

# Function to extract text from various file types
def extract_text_from_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8").split("\n")
    elif file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return [page.extract_text() for page in pdf.pages if page.extract_text()]
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return [para.text for para in doc.paragraphs if para.text.strip()]
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'.")
            return []
        return df["review"].dropna().astype(str).tolist()
    else:
        st.error("Unsupported file type.")
        return []

# === SINGLE TEXT INPUT ===
st.header("📝 Analyze Single Paragraph or Review")
input_text = st.text_area("Enter any review or paragraph:")

if st.button("Analyze Text"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            sentiment = analyzer(input_text)[0]

        st.success("Done!")
        #st.markdown(f"**Original Text:** {input_text}")
        st.markdown(f"**Summary:** {summary}")
        st.markdown(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2%})")

# === FILE UPLOAD ===
st.header("📄 Upload Document for Bulk Analysis")
uploaded_file = st.file_uploader("Upload a .txt, .pdf, .docx, or .csv file", type=["txt", "pdf", "docx", "csv"])

if uploaded_file:
    with st.spinner("Extracting text from file..."):
        texts = extract_text_from_file(uploaded_file)

    if texts:
        st.success(f"{len(texts)} text block(s) extracted.")

        if st.button("Analyze File Content"):
            with st.spinner("Analyzing content..."):
                summaries = summarizer(texts, max_length=150, min_length=30, do_sample=False)
                sentiments = analyzer(texts)

                result_df = pd.DataFrame({
                    "Text": texts,
                    "Summary": [s["summary_text"] for s in summaries],
                    "Sentiment": [s["label"] for s in sentiments],
                    "Confidence": [round(s["score"], 4) for s in sentiments],
                })

            st.success("Analysis complete!")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", csv, "analysis_output.csv", "text/csv")
