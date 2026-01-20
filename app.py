import streamlit as st
from pypdf import PdfReader
from docx import Document
import io
import re
import pandas as pd
import os
import google.generativeai as genai
import json
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="centered"
)
if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is NOT loaded. Check Streamlit Secrets.")
    st.stop()
else:
    st.success("GEMINI_API_KEY loaded successfully.")


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

    response = model.generate_content(prompt)

    try:
        raw = response.text.strip()

        # Handle ```json ... ``` wrappers
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        result = json.loads(raw)
        return int(result["score"]), result["reason"]

    except Exception as e:
        return 0, "AI could not reliably evaluate this role."



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
    st.text_area(
        label="",
        value=cv_text,
        height=250
    )

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
    
        if not results:
            st.warning("No job evaluations available.")
            st.stop()
    
        results = sorted(results, key=lambda x: x["Match Score (%)"], reverse=True)
    
        st.subheader("Job Match Results")
        st.table(results)
    
        best = results[0]
        st.success(
            f"Best match: **{best['Job Title']}** ({best['Match Score (%)']}%)"
        )

