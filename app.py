# =====================================================
# Imports
# =====================================================

import os
import io
import re
import json
import mimetypes

import streamlit as st
import pandas as pd
import google.generativeai as genai


# =====================================================
# UI-safe helpers
# =====================================================

def get_display_score(score: int, seniority: str) -> int:
    """UI-safe score display. Does NOT affect AI logic."""
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


# =====================================================
# Gemini setup
# =====================================================

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is NOT loaded. Check Streamlit Secrets.")
    st.stop()

st.success("Gemini API key loaded successfully.")


# =====================================================
# App config
# =====================================================

st.set_page_config(
    page_title="AI Resume Matcher (new)",
    layout="centered",
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

st.title("AI Resume Matcher (hello)")


# =====================================================
# Model selection
# =====================================================

try:
    available_models = [
        m.name
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
except Exception as e:
    st.error(f"Could not fetch models: {e}")
    available_models = []

MODEL_OPTIONS = {
    "Gemini 3 Flash (Preview)": "models/gemini-3-flash-preview"
}

selected_model_label = st.selectbox(
    "Select AI model",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
)

SELECTED_MODEL = MODEL_OPTIONS[selected_model_label]
st.caption(f"Using model: {SELECTED_MODEL}")


# =====================================================
# Instructions
# =====================================================

st.write("Upload a candidate CV to see which jobs are most likely to result in an offer.")
st.write(
    "📌 For best accuracy, upload CVs in DOCX or text-based PDF format. "
    "Scanned PDFs may reduce matching quality."
)


# =====================================================
# JSON helpers
# =====================================================

def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON found")

    return json.loads(text[start:end + 1])


def safe_parse_json(text: str) -> dict:
    return extract_json(text)


# =====================================================
# Candidate seniority detection
# =====================================================

def detect_candidate_seniority_from_cv(candidate_files):
    text_hint = ""

    for f in candidate_files:
        text_hint += f["data"][:8000].decode(errors="ignore")

    senior_signals = ["店長", "マネージャー", "責任者", "統括"]
    mid_signals = [
        "役職", "リーダー", "主任", "売上", "実績", "成果",
        "達成", "年収", "万円", "契約", "案件", "顧客",
        "営業", "不動産"
    ]

    if any(k in text_hint for k in senior_signals):
        return "SENIOR"

    numeric_mid_patterns = [
        r"[2-9]年",
        r"[2-9]年目",
        r"\d{3,4}万円",
        r"平均年収",
        r"年収\d{3,4}",
    ]

    if any(re.search(p, text_hint) for p in numeric_mid_patterns):
        return "MID"

    if any(k in text_hint for k in mid_signals):
        return "MID"

    return "ENTRY"


# =====================================================
# Job helpers
# =====================================================

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


# =====================================================
# Scoring
# =====================================================

def calculate_score(criteria: dict, seniority: str) -> int:
    weights = {"○": 1.0, "△": 0.6, "×": 0.0}

    raw = (
        weights.get(criteria.get("must_have_requirements"), 0) * 0.4
        + weights.get(criteria.get("preferred_requirements"), 0) * 0.3
        + weights.get(criteria.get("role_alignment"), 0) * 0.3
    )

    score = int(raw * 100)

    if seniority == "ENTRY":
        score = max(score, 35)
    elif seniority == "MID":
        score = max(score, 20)
    elif seniority == "SENIOR":
        score = max(score, 10)

    return min(score, 100)


# =====================================================
# AI matching
# =====================================================

def ai_match_job(candidate_files, job, model_name, candidate_seniority):
    total_bytes = sum(len(f["data"]) for f in candidate_files)

    prompt = f"""Return ONLY valid JSON. No markdown. No explanations. No text outside JSON.
All string values MUST be single-line.

あなたは、書類選考を担当する採用実務者です。

【求人レベル】 {job['seniority']}
【候補者レベル】 {candidate_seniority}

【評価ルール】
○：明確な直接経験
△：間接・限定的経験
×：根拠なし

【出力形式】
{{"score":0,"criteria":{{"must_have_requirements":"○|△|×","preferred_requirements":"○|△|×","role_alignment":"○|△|×"}}}}

【職務内容】
{job["job_context"][:1500]}
"""

    try:
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            [prompt, *candidate_files],
            generation_config={"temperature": 0.3, "max_output_tokens": 900},
        )

        raw = response.text
        parsed = extract_json(raw)

        parsed["score"] = calculate_score(parsed["criteria"], job["seniority"])

        if total_bytes < 2000:
            st.warning("⚠️ The uploaded document may contain little readable text.")

        return {"ok": True, "data": parsed, "raw": raw}

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "raw": None,
            "data": {
                "score": 0,
                "criteria": {
                    "must_have_requirements": "×",
                    "preferred_requirements": "×",
                    "role_alignment": "×",
                },
            },
        }


# =====================================================
# UI
# =====================================================

uploaded_cvs = st.file_uploader(
    "Upload CV files (PDF / DOCX / XLSX)",
    type=["pdf", "docx", "xlsx"],
    accept_multiple_files=True,
)

jobs_file = st.file_uploader(
    "Upload jobs Excel file",
    type=["xlsx"],
)

if uploaded_cvs:
    st.success("CV files uploaded")
    st.session_state.cvs = uploaded_cvs
    st.info(f"{len(uploaded_cvs)} CVs uploaded")

if uploaded_cvs and jobs_file and st.button("Evaluate CVs"):
    jobs_df = pd.read_excel(jobs_file)
    jobs = get_available_jobs(jobs_df)

    candidate_files = []
    for f in st.session_state.cvs:
        candidate_files.append(to_gemini_part(f))

    candidate_seniority = detect_candidate_seniority_from_cv(candidate_files)

    results = []

    for job in jobs:
        result = ai_match_job(candidate_files, job, SELECTED_MODEL, candidate_seniority)
        results.append({
            "job": job,
            "score": result["data"]["score"],
            "criteria": result["data"]["criteria"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    st.session_state.results = results

    st.success("Evaluation completed")

if st.session_state.results:
    st.subheader("CV Evaluation Results")

    for r in st.session_state.results:
        job = r["job"]
        score = get_display_score(r["score"], job["seniority"])

        st.markdown(f"### {job['title']}")
        st.write(f"**Estimated Offer Probability:** {score}%")
        st.divider()
