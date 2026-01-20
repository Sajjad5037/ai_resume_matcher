import os
import streamlit as st
from pypdf import PdfReader
from docx import Document
import io
import pandas as pd
import json
from openai import OpenAI


# ----------------------------
# OpenAI setup
# ----------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is NOT loaded. Check Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="AI Resume Matcher(10:31)",
    layout="centered"
)

st.success("OPENAI_API_KEY loaded successfully.")

st.title("AI Resume Matcher")
st.write("Upload a candidate CV to see which jobs fit best.")


# ----------------------------
# Helpers
# ----------------------------
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)

    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs)

    return ""


def get_available_jobs():
    df = pd.read_excel("jobs.xlsx")

    jobs = []
    for _, row in df.iterrows():
        jobs.append({
            "job_id": row["job_id"],
            "title": row["title"],
            "keywords": [
                kw.strip().lower()
                for kw in str(row["keywords"]).split(",")
                if kw.strip()
            ]
        })

    return jobs


def ai_match_job(cv_text, job):
    prompt = f"""
You are an experienced technical recruiter.

Evaluate how well the following candidate fits the job role.

Return ONLY valid JSON in this format:
{{
  "score": number between 0 and 100,
  "reason": "short explanation"
}}

Candidate CV:
\"\"\"
{cv_text[:6000]}
\"\"\"

Job Role:
Title: {job["title"]}
Required Skills: {", ".join(job["keywords"])}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional recruiter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        result = json.loads(raw)
        return int(result["score"]), result["reason"]

    except Exception:
        return 0, "AI service temporarily unavailable."


# ----------------------------
# UI
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload candidate CV (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Extracting CV text..."):
        cv_text = extract_text(uploaded_file)

    if not cv_text.strip():
        st.error("Could not extract text from this file.")
        st.stop()

    st.subheader("Extracted CV Text")
    st.text_area("", cv_text, height=250)

    st.divider()

    if st.button("Evaluate Candidate"):
        jobs = get_available_jobs()
        results = []

        for job in jobs:
            with st.spinner(f"Evaluating {job['title']}..."):
                score, reason = ai_match_job(cv_text, job)

            results.append({
                "Job Title": job["title"],
                "Match Score (%)": score,
                "Reason": reason
            })

        results = sorted(
            results,
            key=lambda x: x["Match Score (%)"],
            reverse=True
        )

        st.subheader("Job Match Results")
        st.table(results)

        best = results[0]
        st.success(
            f"Best match: **{best['Job Title']}** ({best['Match Score (%)']}%)"
        )
