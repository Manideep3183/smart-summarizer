import streamlit as st
from transformers import pipeline
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models
@st.cache_resource
def load_abstractive_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

abstractive_model = load_abstractive_model()
nlp = spacy.load("en_core_web_sm")

# --- Extractive Summary ---
def extractive_summary(text, num_sentences=3):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer().fit_transform(sentences)
    similarity_matrix = cosine_similarity(vectorizer)
    scores = similarity_matrix.sum(axis=1)

    ranked_indices = np.argsort(scores)[-num_sentences:]
    ranked_sentences = [sentences[i] for i in sorted(ranked_indices)]

    return " ".join(ranked_sentences)

# --- Abstractive Summary (Dynamic) ---
def dynamic_abstractive_summary(text, model):
    word_count = len(text.split())
    max_len = min(300, int(word_count * 0.4))
    min_len = max(20, int(word_count * 0.1))

    chunk = text[:1024]
    summary = model(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
    return summary

# --- PDF/Text File Reader ---
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    return ""

# --- Streamlit App Config ---
st.set_page_config(page_title="📝 Smart AI Summarizer", layout="centered")

# --- Header UI ---
st.markdown("""
    <style>
    .main {background-color: #F8F9FA;}
    .block-container {padding-top: 2rem;}
    </style>
    <div style='text-align: center; padding: 1rem; border-radius: 10px; background-color: #eaf4ff;'>
        <h1 style='color: #3366cc;'>🧠 Smart AI Text Summarizer</h1>
        <p style='color: #444;'>Summarize long documents or content instantly using extractive or abstractive techniques.</p>
    </div>
""", unsafe_allow_html=True)

# --- Input Section ---
st.markdown("### 📥 Input Source")
input_mode = st.radio("Choose input mode:", ["Upload File", "Enter Text Manually"], horizontal=True)
input_text = ""

if input_mode == "Upload File":
    uploaded_file = st.file_uploader("Upload your PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        input_text = extract_text_from_file(uploaded_file)
elif input_mode == "Enter Text Manually":
    input_text = st.text_area("Paste or type your text below:", height=200, placeholder="Paste your article or notes here...")

st.markdown("---")

# --- Summary Options ---
st.markdown("### 🔍 Choose Summarization Method")
summary_type = st.selectbox("Summarization type:", ["Abstractive", "Extractive"])

# --- Display Input and Generate Summary ---
if input_text.strip():
    st.markdown("### 📝 Input Preview")
    st.text_area("Your content:", input_text, height=150, disabled=True)

    if st.button("✨ Generate Summary"):
        with st.spinner("Working on your summary..."):
            if summary_type == "Abstractive":
                summary = dynamic_abstractive_summary(input_text, abstractive_model)
            else:
                summary = extractive_summary(input_text)

        st.markdown("### ✅ Summary Output")
        st.success(summary)
        st.download_button("⬇️ Download Summary", summary, file_name="summary.txt", mime="text/plain")
else:
    st.info("📌 Please upload a file or paste text to get started.")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: 13px; color: gray;'>
        Built with ❤️ using Streamlit, Hugging Face Transformers, spaCy, and Scikit-learn.
    </div>
""", unsafe_allow_html=True)
