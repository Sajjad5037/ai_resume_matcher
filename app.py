import os
import json
import streamlit as st
import pandas as pd

from google import genai
from google.genai import types

from pypdf import PdfReader
from docx import Document

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_NAME = "models/gemini-1.5-flash"
CHUNK_SIZE = 4000  # safe chunk size for Gemini
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI Resume Matcher – MVP", layout="centered")
st.title("AI Resume Matcher – MVP")

# -------------------------------------------------
# FILE TEXT EXTRACTION
# -------------------------------------------------
def extract_text(file) -> str:
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE):
    return [text[i:i + size] for i in range(0, len(text), size)]

def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in model output")
    return json.loads(text[start:end + 1])

# -------------------------------------------------
# STEP 1: SUMMARIZE CV CHUNKS
# -------------------------------------------------
def summarize_cv_chunk(chunk: str) -> dict:
    prompt = f"""
Return ONLY valid JSON.
No markdown. No explanations.

Summarize the CV text below using ONLY what is written.

Text:
{chunk}

Output format:
{{
  "summary": "",
  "skills": []
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt]
    )
    return extract_json(response.text or "")

# -------------------------------------------------
# STEP 2: BUILD FINAL CANDIDATE PROFILE
# -------------------------------------------------
def build_candidate_profile(full_cv_text: str) -> dict:
    chunks = chunk_text(full_cv_text)

    partial_summaries = []
    for c in chunks:
        partial_summaries.append(summarize_cv_chunk(c))

    merge_prompt = f"""
Return ONLY valid JSON.
No markdown. No explanations.

Merge the following partial CV summaries into ONE candidate profile.

Partial summaries:
{json.dumps(partial_summaries, ensure_ascii=False)}

Output format:
{{
  "summary": "",
  "key_skills": [],
  "experience_level": "ENTRY | MID | SENIOR"
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[merge_prompt]
    )
    return extract_json(response.text or "")

# -------------------------------------------------
# STEP 3: SCORE ONE JOB
# -------------------------------------------------
def score_job(candidate_profile: dict, job_text: str) -> dict:
    prompt = f"""
Return ONLY valid JSON.

Evaluate job fit at document screening stage.

Candidate profile:
{json.dumps(candidate_profile, ensure_ascii=False)}

Job description:
{job_text}

Rules:
- Be strict but fair
- ENTRY roles must not penalize lack of experience

Output format:
{{
  "score": 0,
  "reason": ""
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt]
    )
    return extract_json(response.text or "")

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_cvs = st.file_uploader(
    "Upload candidate CVs (PDF / DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

jobs_file = st.file_uploader(
    "Upload jobs Excel file",
    type=["xlsx"]
)

if uploaded_cvs and jobs_file and st.button("Run Evaluation"):

    # ----------------------------
    # READ CVs
    # ----------------------------
    with st.spinner("Reading CVs..."):
        all_cv_text = ""
        for f in uploaded_cvs:
            all_cv_text += extract_text(f) + "\n\n"

    if not all_cv_text.strip():
        st.error("No readable text found in CVs.")
        st.stop()

    # ----------------------------
    # BUILD CANDIDATE PROFILE
    # ----------------------------
    with st.spinner("Building candidate profile..."):
        candidate_profile = build_candidate_profile(all_cv_text)

    st.subheader("🧠 Candidate Profile")
    st.json(candidate_profile)

    # ----------------------------
    # READ JOBS
    # ----------------------------
    jobs_df = pd.read_excel(jobs_file)

    results = []

    with st.spinner("Scoring jobs..."):
        for _, row in jobs_df.iterrows():
            job_text = "\n".join(
                str(v) for v in row.values if pd.notna(v)
            )

            job_result = score_job(candidate_profile, job_text)

            results.append({
                "Job": row.get("title", "Unknown"),
                "Score": job_result["score"],
                "Reason": job_result["reason"]
            })

    results.sort(key=lambda x: x["Score"], reverse=True)

    # ----------------------------
    # DISPLAY RESULTS
    # ----------------------------
    st.subheader("📊 Job Match Results")

    for r in results:
        st.markdown(f"### {r['Job']}")
        st.write(f"**Score:** {r['Score']}%")
        st.write(r["Reason"])
        st.divider()
