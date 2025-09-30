import streamlit as st
import pdfplumber
from transformers import pipeline
import boto3
import os
import re

# -----------------------------
# Summarizer & Quiz
# -----------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_generator = pipeline("text2text-generation", model="google/flan-t5-small")

# -----------------------------
# Amazon Polly
# -----------------------------
polly_client = boto3.client('polly', region_name='us-east-1')

def text_to_speech(text, filename, voice="Zeina", chunk_size=1500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    full_audio = b""
    for chunk in chunks:
        response = polly_client.synthesize_speech(
            Text=chunk,
            OutputFormat="mp3",
            VoiceId=voice
        )
        full_audio += response['AudioStream'].read()
    with open(filename, 'wb') as f:
        f.write(full_audio)

def split_text_by_language(text):
    arabic = ""
    english = ""
    for line in text.split("\n"):
        if re.search(r"[اأإء-ي]", line):
            arabic += line + "\n"
        else:
            english += line + "\n"
    return arabic.strip(), english.strip()

# -----------------------------
# PDF Extraction
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# -----------------------------
# تلخيص النصوص الطويلة
# -----------------------------
def summarize_long_text(text, chunk_size=1000):
    if not text.strip():
        return "❌ لم يتم استخراج أي نص من الـ PDF"
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        try:
            s = summarizer(chunk, max_length=300, min_length=50, do_sample=False)
            if s and len(s) > 0:
                summaries.append(s[0]['summary_text'])
        except Exception as e:
            summaries.append(f"❌ خطأ أثناء التلخيص: {str(e)}")
    return "\n".join(summaries)

# -----------------------------
# توليد Quiz مبسط
# -----------------------------
def generate_quiz_long_text(text, chunk_size=1000):
    if not text.strip():
        return "❌ لم يتم استخراج أي نص من الـ PDF"
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    all_quiz = []
    for chunk in chunks:
        try:
            prompt = f"Generate 5 multiple choice questions from this text:\n{chunk}"
            q = question_generator(prompt)[0]['generated_text']
            all_quiz.append(q)
        except Exception as e:
            all_quiz.append(f"❌ خطأ أثناء توليد Quiz: {str(e)}")
    return "\n".join(all_quiz)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI Lecture Summarizer + Podcast + Quiz")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text(uploaded_file)

    summary = summarize_long_text(text, chunk_size=1000)
    st.subheader("Summary")
    st.text(summary)

    arabic_text, english_text = split_text_by_language(summary)

    if arabic_text:
        if not os.path.exists("assets"):
            os.makedirs("assets")
        text_to_speech(arabic_text, "assets/podcast_ar.mp3", voice="Zeina")
        st.subheader("Podcast - Arabic")
        st.audio("assets/podcast_ar.mp3", format="audio/mp3")

    if english_text:
        if not os.path.exists("assets"):
            os.makedirs("assets")
        text_to_speech(english_text, "assets/podcast_en.mp3", voice="Joanna")
        st.subheader("Podcast - English")
        st.audio("assets/podcast_en.mp3", format="audio/mp3")

    quiz = generate_quiz_long_text(text, chunk_size=1000)
    st.subheader("Quiz")
    st.text(quiz)
