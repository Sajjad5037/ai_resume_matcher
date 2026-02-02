import os
import streamlit as st
import io
import pandas as pd
import json
import re
import google.generativeai as genai
import mimetypes

# ----------------------------
# Utility functions
# ----------------------------

def get_display_score(score: int, seniority: str) -> int:
    if seniority == "ENTRY":
        return max(score, 25)
    return score


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


def to_gemini_part(uploaded_file):
    uploaded_file.seek(0)

    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if uploaded_file.name.lower().endswith(".pdf"):
        mime_type = "application/pdf"

    if not mime_type:
        mime_type = uploaded_file.type

    return {
        "mime_type": mime_type,
        "data": uploaded_file.read(),
    }


# ----------------------------
# Gemini setup
# ----------------------------

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is NOT loaded. Check Streamlit Secrets.")
    st.stop()


# ----------------------------
# App Config
# ----------------------------

st.set_page_config(
    page_title="AI Resume Matcher (new)",
    layout="centered"
)

if "results" not in st.session_state:
    st.session_state.results = []

if "explanations" not in st.session_state:
    st.session_state.explanations = {}

if "explain_open" not in st.session_state:
    st.session_state.explain_open = {}

if "cvs" not in st.session_state:
    st.session_state.cvs = None

if "active_candidate" not in st.session_state:
    st.session_state.active_candidate = None

st.success("Gemini API key loaded successfully.")
st.title("AI Resume Matcher (hello)")


# ----------------------------
# Model selection
# ----------------------------

MODEL_OPTIONS = {
    "Gemini 3 Flash (Preview)": "models/gemini-3-flash-preview"
}

selected_model_label = st.selectbox(
    "Select AI model",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

SELECTED_MODEL = MODEL_OPTIONS[selected_model_label]
st.caption(f"Using model: {SELECTED_MODEL}")


# ----------------------------
# JSON helpers
# ----------------------------

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Invalid JSON returned by model")

    return json.loads(text[start:end + 1])


# ----------------------------
# Candidate seniority detection
# ----------------------------

def detect_candidate_seniority_from_cv(candidate_files):
    text_hint = ""

    for f in candidate_files:
        text_hint += f["data"][:8000].decode(errors="ignore")

    senior_signals = ["店長", "マネージャー", "責任者", "統括"]
    mid_signals = [
        "役職", "リーダー", "主任", "売上", "実績", "成果",
        "達成", "年収", "万円", "契約", "案件", "顧客"
    ]

    if any(k in text_hint for k in senior_signals):
        return "SENIOR"

    if any(k in text_hint for k in mid_signals):
        return "MID"

    return "ENTRY"


# ----------------------------
# Job parsing
# ----------------------------

def get_title(row):
    for key in ["title", "position"]:
        if key in row.index:
            value = row[key]
            if not pd.isna(value) and str(value).strip():
                return str(value).strip()
    return "Unknown Role"


def get_available_jobs(df: pd.DataFrame):
    df.columns = df.columns.astype(str).str.strip()
    jobs = []

    def safe(v):
        return "" if pd.isna(v) else str(v).strip()

    for _, row in df.iterrows():
        title = get_title(row)

        job_context_parts = [
            f"Job Title: {safe(row.get('title'))}",
            f"Position: {safe(row.get('position'))}",
            f"Industry: {safe(row.get('job_industry'))}",
            f"Job Type: {safe(row.get('job_type'))}",
            f"Location: {safe(row.get('location'))}",
            f"Job Description: {safe(row.get('job_content'))}",
            f"Required Experience: {safe(row.get('required_experience'))}",
            f"Desired Experience: {safe(row.get('desired_experience'))}",
            f"Target Candidate: {safe(row.get('target_candidate'))}",
            f"Education: {safe(row.get('education'))}",
            f"Eligibility Details: {safe(row.get('eligibility_details'))}",
        ]

        job_context = "\n".join(
            part for part in job_context_parts if part.split(": ", 1)[1]
        )

        seniority = detect_seniority(job_context)

        jobs.append({
            "job_id": safe(row.get("job_url")),
            "title": title,
            "job_context": job_context,
            "seniority": seniority,
            "company_name": safe(row.get("company_name")),
            "passrate_for_doc_screening": safe(row.get("passrate_for_doc_screening")),
            "documents_to_job_offer_ratio": safe(row.get("documents_to_job_offer_ratio")),
            "fee": safe(row.get("fee")),
        })

    return jobs


# ----------------------------
# Scoring
# ----------------------------

def calculate_score(criteria: dict, seniority: str) -> int:
    weights = {"○": 1.0, "△": 0.6, "×": 0.0}

    raw = (
        weights.get(criteria.get("must_have_requirements"), 0) * 0.4 +
        weights.get(criteria.get("preferred_requirements"), 0) * 0.3 +
        weights.get(criteria.get("role_alignment"), 0) * 0.3
    )

    score = int(raw * 100)

    if seniority == "ENTRY":
        score = max(score, 35)
    elif seniority == "MID":
        score = max(score, 20)
    elif seniority == "SENIOR":
        score = max(score, 10)

    return min(score, 100)


# ----------------------------
# Gemini evaluation
# ----------------------------

def ai_match_job(candidate_files, job, model_name, candidate_seniority):
    prompt = f"""
Return ONLY valid JSON.
No markdown.
No explanations.

{{ "score": 0, "criteria": {{
  "must_have_requirements": "○|△|×",
  "preferred_requirements": "○|△|×",
  "role_alignment": "○|△|×"
}} }}

【職務内容】
{job["job_context"][:1500]}
"""

    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        [prompt, *candidate_files],
        generation_config={"temperature": 0.3, "max_output_tokens": 900}
    )

    raw = response.text
    parsed = extract_json(raw)

    parsed["score"] = calculate_score(
        parsed.get("criteria", {}),
        job["seniority"]
    )

    return {"ok": True, "data": parsed, "raw": raw}


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

if uploaded_cvs:
    st.session_state.cvs = uploaded_cvs
    st.success(f"{len(uploaded_cvs)} CV(s) uploaded")

if st.session_state.cvs and jobs_file and st.button("Evaluate CVs"):
    jobs_df = pd.read_excel(jobs_file)
    jobs = get_available_jobs(jobs_df)

    candidate_files = [to_gemini_part(f) for f in st.session_state.cvs]
    candidate_seniority = detect_candidate_seniority_from_cv(candidate_files)

    st.session_state.candidate_files = candidate_files
    st.session_state.candidate_seniority = candidate_seniority

    results = []

    for job in jobs:
        result = ai_match_job(
            candidate_files,
            job,
            SELECTED_MODEL,
            candidate_seniority
        )

        results.append({
            "job": job,
            "score": result["data"]["score"],
            "criteria": result["data"]["criteria"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    st.session_state.results = [{
        "cv_name": "Combined Candidate Profile",
        "cv_type": "MULTI-DOC",
        "cv_files": [f.name for f in st.session_state.cvs],
        "results": results
    }]

# ----------------------------
# Results rendering
# ----------------------------

if st.session_state.results:
    st.subheader("CV Evaluation Results")

    for cv_block in st.session_state.results:
        best_job = cv_block["results"][0]

        st.success(
            f"Best match: **{best_job['job']['title']}** "
            f"({best_job['score']}%)"
        )
