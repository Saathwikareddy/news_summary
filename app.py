import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PyPDF2 import PdfReader
from docx import Document

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------
# Load Model & Tokenizers
# -----------------------

model = load_model("meeting_notes_summarizer.h5")

with open("article_tokenizer.pkl", "rb") as f:
    article_tokenizer = pickle.load(f)

with open("summary_tokenizer.pkl", "rb") as f:
    summary_tokenizer = pickle.load(f)

max_article_len = 100
max_summary_len = 15

reverse_target_index = {
    i: word for word, i in summary_tokenizer.word_index.items()
}


# -----------------------
# Text Cleaning
# -----------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text


# -----------------------
# Generate Summary
# -----------------------

def generate_summary(text):

    cleaned = clean_text(text)

    seq = article_tokenizer.texts_to_sequences([cleaned])

    seq = pad_sequences(
        seq,
        maxlen=max_article_len,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(
        [seq, np.zeros((1, max_summary_len))],
        verbose=0
    )

    words = []

    for t in range(prediction.shape[1]):

        idx = np.argmax(prediction[0, t])

        word = reverse_target_index.get(idx, "")

        if word not in [
            "<start>",
            "<end>",
            "start",
            "end",
            ""
        ]:
            words.append(word)

    return " ".join(words)


# -----------------------
# PDF Reader
# -----------------------

def read_pdf(uploaded_file):

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        text += page.extract_text() + "\n"

    return text


# -----------------------
# DOCX Reader
# -----------------------

def read_docx(uploaded_file):

    doc = Document(uploaded_file)

    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


# -----------------------
# PDF Download Creator
# -----------------------

def create_pdf(summary_text):

    filename = "summary.pdf"

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    content = [
        Paragraph("AI Generated Summary", styles["Title"]),
        Paragraph(summary_text, styles["BodyText"])
    ]

    doc.build(content)

    return filename


# -----------------------
# UI
# -----------------------

st.set_page_config(
    page_title="AI Meeting Minutes Summarizer",
    layout="wide"
)

st.title("📝 AI Meeting Minutes Summarizer")

# ======================================
# Section 1
# ======================================

st.header("Section 1: Enter Meeting Notes")

meeting_notes = st.text_area(
    "Paste meeting notes here...",
    height=250
)

# ======================================
# Bonus 1 & 2
# ======================================

st.subheader("Upload PDF or DOCX")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "docx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".pdf"):
        meeting_notes = read_pdf(uploaded_file)

    elif uploaded_file.name.endswith(".docx"):
        meeting_notes = read_docx(uploaded_file)

    st.success("File loaded successfully!")

# ======================================
# Section 2
# ======================================

st.header("Section 2: Generate Summary")

generate = st.button("Generate Summary")

if generate:

    if meeting_notes.strip() == "":
        st.warning("Please enter meeting notes.")
        st.stop()

    summary = generate_summary(meeting_notes)

    # ==================================
    # Section 3
    # ==================================

    st.header("Section 3: AI Generated Summary")

    st.success(summary)

    # ==================================
    # Section 4
    # ==================================

    original_words = len(meeting_notes.split())
    summary_words = len(summary.split())

    compression_ratio = (
        (original_words - summary_words)
        / original_words
    ) * 100

    st.header("Section 4: Statistics")

    st.metric(
        "Original Words",
        original_words
    )

    st.metric(
        "Summary Words",
        summary_words
    )

    st.metric(
        "Compression Ratio",
        f"{compression_ratio:.2f}%"
    )

    # ==================================
    # Bonus 3
    # ==================================

    pdf_file = create_pdf(summary)

    with open(pdf_file, "rb") as f:

        st.download_button(
            label="Download Summary as PDF",
            data=f,
            file_name="Meeting_Summary.pdf",
            mime="application/pdf"
        )
