import os
import json
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types

# --------------------
# CONFIG
# --------------------
MODEL_NAME = "models/gemini-1.5-flash"

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI Resume Matcher MVP", layout="centered")
st.title("AI Resume Matcher – MVP")

# --------------------
# HELPERS
# --------------------
def to_part(uploaded_file):
    uploaded_file.seek(0)
    return types.Part.from_bytes(
        data=uploaded_file.read(),
        mime_type=uploaded_file.type or "application/octet-stream"
    )

def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found")
    return json.loads(text[start:end + 1])

# --------------------
# STEP 1: BUILD CANDIDATE PROFILE (CVs passed directly)
# --------------------
def build_candidate_profile(cv_parts):
    prompt = """
Return ONLY valid JSON.
No markdown.
No text outside JSON.

You are analyzing multiple CV documents belonging to the SAME candidate.

Create a unified candidate profile using ONLY what is written.

Output format:
{
  "summary": "",
  "skills": [],
  "experience_level": "ENTRY | MID | SENIOR"
}
"""

    contents = [prompt, *cv_parts]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents
    )

    return extract_json(response.text or "")

# --------------------
# STEP 2: SCORE CANDIDATE AGAINST ONE JOB
# --------------------
def score_job(candidate_profile, job_text):
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

# --------------------
# UI
# --------------------
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
    with st.spinner("Building candidate profile…"):
        cv_parts = [to_part(f) for f in uploaded_cvs]
        candidate_profile = build_candidate_profile(cv_parts)

    st.subheader("🧠 Candidate Profile")
    st.json(candidate_profile)

    jobs_df = pd.read_excel(jobs_file)

    results = []

    with st.spinner("Scoring jobs…"):
        for _, row in jobs_df.iterrows():
            job_text = "\n".join(str(v) for v in row.values if pd.notna(v))
            result = score_job(candidate_profile, job_text)

            results.append({
                "Job": row.get("title", "Unknown"),
                "Score": result["score"],
                "Reason": result["reason"]
            })

    results = sorted(results, key=lambda x: x["Score"], reverse=True)

    st.subheader("📊 Results")
    for r in results:
        st.markdown(f"### {r['Job']}")
        st.write(f"**Score:** {r['Score']}%")
        st.write(r["Reason"])
        st.divider()
