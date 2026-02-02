import os
import json
import streamlit as st
import pandas as pd

from google import genai
from pypdf import PdfReader
from docx import Document

# ------------------------
# CONFIG
# ------------------------
MODEL_NAME = "models/gemini-1.5-flash"
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI Resume Matcher – MVP", layout="centered")
st.title("AI Resume Matcher – MVP")

# ------------------------
# HELPERS
# ------------------------
def extract_text(file) -> str:
    """Convert CV file into plain text (no AI, no logic)."""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON returned by model")
    return json.loads(text[start:end + 1])

# ------------------------
# STEP 1: BUILD CANDIDATE PROFILE (AI)
# ------------------------
def build_candidate_profile(cv_text: str) -> dict:
    prompt = f"""
Return ONLY valid JSON.
No markdown.
No text outside JSON.

The following text contains MULTIPLE CVs belonging to the SAME candidate.

CV TEXT:
{cv_text}

Create a unified candidate profile using ONLY what is written.

Output format:
{{
  "summary": "",
  "key_skills": [],
  "experience_level": "ENTRY | MID | SENIOR"
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt]
    )

    return extract_json(response.text or "")

# ------------------------
# STEP 2: SCORE ONE JOB (AI)
# ------------------------
def score_job(candidate_profile: dict, job_text: str) -> dict:
    prompt = f"""
Return ONLY valid JSON.

You are evaluating job fit at document screening stage.

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

# ------------------------
# UI
# ------------------------
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

    # ---- CV INGESTION ----
    with st.spinner("Reading CV documents…"):
        cv_text = ""
        for f in uploaded_cvs:
            cv_text += extract_text(f) + "\n\n"

    if not cv_text.strip():
        st.error("No readable text found in CVs.")
        st.stop()

    # ---- CANDIDATE PROFILE ----
    with st.spinner("Building candidate profile…"):
        candidate_profile = build_candidate_profile(cv_text)

    st.subheader("🧠 Candidate Profile")
    st.json(candidate_profile)

    # ---- JOB SCORING ----
    jobs_df = pd.read_excel(jobs_file)
    results = []

    with st.spinner("Scoring jobs…"):
        for _, row in jobs_df.iterrows():
            job_text = "\n".join(
                str(v) for v in row.values if pd.notna(v)
            )

            result = score_job(candidate_profile, job_text)

            results.append({
                "Job": row.get("title", "Unknown"),
                "Score": result["score"],
                "Reason": result["reason"]
            })

    results.sort(key=lambda x: x["Score"], reverse=True)

    # ---- OUTPUT ----
    st.subheader("📊 Results")
    for r in results:
        st.markdown(f"### {r['Job']}")
        st.write(f"**Score:** {r['Score']}%")
        st.write(r["Reason"])
        st.divider()
