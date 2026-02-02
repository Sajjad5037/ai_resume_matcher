import os
import json
import mimetypes
import pandas as pd
import streamlit as st
import google.generativeai as genai

# ----------------------------
# Gemini Setup
# ----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("AI Resume Matcher")

st.write(
    "📌 For best accuracy, upload CVs in DOCX or text-based PDF format. "
    "Scanned PDFs may reduce matching quality."
)

# ----------------------------
# Helpers
# ----------------------------
def to_gemini_part(uploaded_file):
    uploaded_file.seek(0)

    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if uploaded_file.name.lower().endswith(".pdf"):
        mime_type = "application/pdf"

    if not mime_type:
        mime_type = uploaded_file.type or "application/octet-stream"

    return {
        "mime_type": mime_type,
        "data": uploaded_file.read(),
    }


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON found")
    return json.loads(text[start:end + 1])


def detect_seniority(job_context: str) -> str:
    keywords_entry = ["未経験OK", "経験不問", "第二新卒"]
    keywords_senior = ["3年以上", "5年以上", "リード", "マネージャー"]

    for k in keywords_entry:
        if k in job_context:
            return "ENTRY"

    for k in keywords_senior:
        if k in job_context:
            return "SENIOR"

    return "MID"


def get_available_jobs(df: pd.DataFrame):
    df.columns = df.columns.astype(str).str.strip()
    jobs = []

    def safe(v):
        return "" if pd.isna(v) else str(v).strip()

    for _, row in df.iterrows():
        job_context_parts = [
            f"Job Title: {safe(row.get('title'))}",
            f"Position: {safe(row.get('position'))}",
            f"Industry: {safe(row.get('job_industry'))}",
            f"Job Type: {safe(row.get('job_type'))}",
            f"Location: {safe(row.get('location'))}",
            f"Job Description: {safe(row.get('job_content'))}",
            f"Required Experience: {safe(row.get('required_experience'))}",
            f"Desired Experience: {safe(row.get('desired_experience'))}",
        ]

        job_context = "\n".join(
            p for p in job_context_parts if p.split(": ", 1)[1]
        )

        jobs.append({
            "title": safe(row.get("title")) or "Unknown Role",
            "job_context": job_context,
            "seniority": detect_seniority(job_context),
            "company_name": safe(row.get("company_name")),
        })

    return jobs


# ----------------------------
# AI Core
# ----------------------------
def generate_full_assessment(candidate_files, job, model_name, candidate_seniority):
    prompt = f"""
Return ONLY valid JSON.
No markdown.
No text outside JSON.

あなたは、キャリアアドバイザー兼採用担当者です。
以下の履歴書（CV）を丁寧に読み、
この求人に対して「なぜそう評価したのか」が
第三者にも分かるように説明してください。

【重要な前提】
- 評価は書類選考段階のものです
- CVに明示的に記載されている内容のみを根拠にしてください
- 推測や断定は禁止です
- ENTRY求人では経験不足を否定的に扱ってはいけません

【求人レベル】
{job["seniority"]}

【候補者レベル】
{candidate_seniority}

【職務内容】
{job["job_context"][:1500]}

【出力JSON形式（厳守）】
{{
  "SUMMARY": "",
  "MUST_HAVE": "",
  "PREFERRED": "",
  "ALIGNMENT": "",
  "score": 0
}}
"""

    model = genai.GenerativeModel(model_name)

    contents = [prompt, *candidate_files]

    response = model.generate_content(
        contents,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 900,
        }
    )

    return extract_json(response.text)


# ----------------------------
# UI
# ----------------------------
uploaded_cvs = st.file_uploader(
    "Upload CV files (PDF / DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

jobs_file = st.file_uploader(
    "Upload jobs Excel file",
    type=["xlsx"]
)

MODEL_NAME = "models/gemini-1.5-pro-001"

if uploaded_cvs and jobs_file and st.button("Evaluate CVs"):
    jobs_df = pd.read_excel(jobs_file)
    jobs = get_available_jobs(jobs_df)

    candidate_files = []
    for f in uploaded_cvs:
        candidate_files.append(to_gemini_part(f))

    candidate_seniority = "ENTRY"  # intentionally fixed (your original logic)

    st.subheader("📊 Results")

    for job in jobs:
        result = generate_full_assessment(
            candidate_files,
            job,
            MODEL_NAME,
            candidate_seniority
        )

        st.markdown(f"### {job['title']}")
        st.write(f"**Score:** {result['score']}%")
        st.write("**Summary**")
        st.write(result["SUMMARY"])
        st.write("**Must Have**")
        st.write(result["MUST_HAVE"])
        st.write("**Preferred**")
        st.write(result["PREFERRED"])
        st.write("**Alignment**")
        st.write(result["ALIGNMENT"])
        st.divider()
