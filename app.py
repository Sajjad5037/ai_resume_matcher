import os 
import streamlit as st
from pypdf import PdfReader
from docx import Document 
import io
import pandas as pd
import json 
#from openai import OpenAI
import re
import google.generativeai as genai




genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is NOT loaded. Check Streamlit Secrets.")
    st.stop()



# ----------------------------
# OpenAI setup
# ----------------------------
#if not os.getenv("OPENAI_API_KEY"):
#    st.error("OPENAI_API_KEY is NOT loaded. Check Streamlit Secrets.")
#    st.stop()

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
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

    
st.success("Gemini API key loaded successfully.")


st.title("AI Resume Matcher (hello)")



# ----------------------------
# Model Selection & Diagnostics
# ----------------------------
try:
    # This fetches the actual list from Google
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
except Exception as e:
    st.error(f"Could not fetch models: {e}")
    available_models = []

# ----------------------------
# Model Selection
# ----------------------------
MODEL_OPTIONS = {
    "Gemini 2.5 Flash (Recommended)": "models/gemini-2.5-flash"
}
selected_model_label = st.selectbox(
    "Select AI model",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

SELECTED_MODEL = MODEL_OPTIONS[selected_model_label]

st.caption(f"Using model: `{SELECTED_MODEL}`")

st.write("Upload a candidate CV to see which jobs are most likely to result in an offer.")

def safe_parse_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON found")
    return json.loads(text[start:end + 1])


def extract_cv_text_from_uploaded_file(uploaded_file) -> str:
    uploaded_file.seek(0)
    file_type = uploaded_file.type

    # PDF
    if file_type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)

    # DOCX
    if file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # XLSX
    if file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df_dict = pd.read_excel(uploaded_file, sheet_name=None)
        blocks = []
        for sheet, df in df_dict.items():
            blocks.append(f"【Sheet: {sheet}】")
            for _, row in df.iterrows():
                row_text = " ".join(str(v) for v in row.values if not pd.isna(v))
                if row_text.strip():
                    blocks.append(row_text)
        return "\n".join(blocks)

    return ""


def generate_explanation(cv_text, job, evaluation):
    prompt = f"""
Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.
Do not add extra keys.

You are a professional career advisor at a recruitment agency.
Explain the candidate’s likelihood of receiving an offer for this job.

Rules:
- Each field must contain at least ONE complete sentence.
- Do NOT leave any field empty.
- Use natural Japanese prose.
- Do NOT use bullet points.
- Do NOT mention AI.

JSON format (must match exactly):
{{
  "SUMMARY": "",
  "MUST_HAVE": "",
  "PREFERRED": "",
  "ALIGNMENT": ""
}}

Evaluation context:
- 必須要件：{evaluation["criteria"]["must_have_requirements"]}
- 歓迎要件：{evaluation["criteria"]["preferred_requirements"]}
- 業務内容との親和性：{evaluation["criteria"]["role_alignment"]}
- 想定内定確率：{evaluation["score"]}％

Candidate CV:
{cv_text[:2000]}

Job description:
{job["job_context"][:1200]}
"""

    model = genai.GenerativeModel(SELECTED_MODEL)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 900,
        }
    )

    # Parse JSON safely
    try:
        return safe_parse_json(response.text)
    except Exception:
        return {
            "SUMMARY": response.text.strip(),
            "MUST_HAVE": "必須要件について概ね満たしていると判断されます。",
            "PREFERRED": "歓迎要件についても一定の適合性が確認できます。",
            "ALIGNMENT": "業務内容との親和性は高いと考えられます。"
        }


def generate_with_retry(model, prompt, retries=1):
    last_error = None

    for attempt in range(1, retries + 1):
        

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
            },
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            }
        )

        
        candidate = response.candidates[0]

        
        raw = candidate.content.parts[0].text

        
        # ---- JSON PARSE ----
        try:
            parsed = extract_json(raw)
            
            return parsed, raw
        except Exception as e:
            
            last_error = e

    raise ValueError(f"Failed after retries: {last_error}")

def extract_json(text):
    # Find the first opening brace
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    # Find the last closing brace
    end = text.rfind("}")
    if end == -1 or end <= start:
        raise ValueError("Incomplete JSON object in model output")

    json_str = text[start:end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model: {e}")
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


def get_title(row):
    for key in ["title", "position"]:
        if key in row.index:
            value = row[key]
            if not pd.isna(value) and str(value).strip():
                return str(value).strip()
    return "Unknown Role"


def get_available_jobs(df: pd.DataFrame):
    """
    Builds structured job contexts from uploaded Excel file
    """
    df.columns = df.columns.astype(str).str.strip()

    jobs = []

    def safe(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

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

        jobs.append({
            "job_id": safe(row.get("job_url")),
            "title": title,
            "job_context": job_context,
            "company_name": safe(row.get("company_name")),
            "passrate_for_doc_screening": safe(row.get("passrate_for_doc_screening")),
            "documents_to_job_offer_ratio": safe(row.get("documents_to_job_offer_ratio")),
            "fee": safe(row.get("fee")),
        })

    return jobs


def ai_match_job(cv_text, job, model_name):
    """
    Gemini 2.5 Flash – minimal, deterministic JSON version
    """

    prompt = f"""
Return ONLY valid JSON.
No markdown. No explanations.
All string values MUST be single-line.
Do NOT include newline characters inside strings.

Do NOT include explanations or reasons.
Only output numeric score and ratings.

JSON format:
{{
  "score": 0,
  "criteria": {{
    "must_have_requirements": "○|△|×",
    "preferred_requirements": "○|△|×",
    "role_alignment": "○|△|×"
  }}
}}

Candidate CV:
{cv_text[:3000]}

Job description:
{job["job_context"][:1500]}
"""

    try:
        model = genai.GenerativeModel(model_name)

        parsed, raw = generate_with_retry(model, prompt)

        return {
            "ok": True,
            "data": parsed,
            "raw": raw
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "raw": raw if 'raw' in locals() else None,
            "data": {
                "score": 0,
                "criteria": {
                    "must_have_requirements": "×",
                    "preferred_requirements": "×",
                    "role_alignment": "×"
                }
            }
        }



    

# ----------------------------
# UI
# ----------------------------
uploaded_cvs = st.file_uploader(
    "Upload CV files (PDF / DOCX / XLSX)",
    type=["pdf", "docx", "xlsx"],
    accept_multiple_files=True,
    key="cv_files"
)

jobs_file = st.file_uploader(
    "Upload jobs Excel file",
    type=["xlsx"],
    key="jobs_excel"
)


# 🔹 ADD THIS BLOCK HERE (exactly here)
if uploaded_cvs:
    st.success("CV files uploaded")

    st.session_state.cvs = uploaded_cvs

    st.info(f"{len(uploaded_cvs)} CVs uploaded")
    
if uploaded_cvs and jobs_file and st.button("Evaluate CVs"):

    jobs_df = pd.read_excel(jobs_file)
    jobs = get_available_jobs(jobs_df)

    folder_results = []
    
    status = st.empty()
    progress = st.progress(0)

    for idx, uploaded_file in enumerate(st.session_state.cvs, start=1):
        status.info(
            f"Evaluating {uploaded_file.name} ({idx}/{len(st.session_state.cvs)})"
        )

        cv_text = extract_cv_text_from_uploaded_file(uploaded_file)
        if not cv_text.strip():
            continue

        cv_results = []
        for job_idx, job in enumerate(jobs, start=1):
            status.info(
                f"Evaluating {uploaded_file.name} "
                f"({idx}/{len(st.session_state.cvs)}) – "
                f"Job {job_idx}/{len(jobs)}"
            )
        
            result = ai_match_job(cv_text, job, SELECTED_MODEL)    
            cv_results.append({
                "job": job,
                "score": result["data"]["score"],
                "criteria": result["data"]["criteria"]
            })

        cv_results.sort(key=lambda x: x["score"], reverse=True)

        folder_results.append({
            "cv_name": uploaded_file.name,
            "cv_type": uploaded_file.name.split(".")[-1].upper(),
            "cv_text": cv_text,
            "results": cv_results
        })

        progress.progress(idx / len(st.session_state.cvs))
    
    status.success("Evaluation completed")


    st.session_state.results = folder_results
if st.session_state.results:
    st.subheader("CV Evaluation Results")


    for cv_idx, cv_block in enumerate(st.session_state.results):
        if not cv_block["results"]:
            continue

        best_job = cv_block["results"][0]
        st.success(
            f"Best match for {cv_block['cv_name']}: "
            f"**{best_job['job']['title']}** ({best_job['score']}%)"
        )

        with st.expander(f"📄 {cv_block['cv_name']} ({cv_block['cv_type']})"):
            for job_idx, r in enumerate(cv_block["results"]):
                job = r["job"]

                st.markdown(f"### {job['title']}")
                st.write(f"**Estimated Offer Probability:** {r['score']}%")

                explain_key = f"{cv_idx}_{job_idx}"

                # --- init open state ---
                if explain_key not in st.session_state.explain_open:
                    st.session_state.explain_open[explain_key] = False
                
                
                # --- button only mutates state ---
                if st.button(
                    f"Explain – {cv_block['cv_name']} – {job['title']}",
                    key=f"explain_btn_{explain_key}"
                ):
                    st.session_state.explain_open[explain_key] = True
                
                    if explain_key not in st.session_state.explanations:
                        st.session_state.explanations[explain_key] = generate_explanation(
                            cv_block["cv_text"], job, r
                        )
                
                
                # --- rendering depends ONLY on state ---
                if st.session_state.explain_open.get(explain_key, False):
                    sections = st.session_state.explanations.get(explain_key)
                
                    if sections:
                        st.markdown("### 📝 評価サマリー")
                        st.write(sections.get("SUMMARY", ""))
                
                        with st.expander("📊 Evaluation details", expanded=True):
                            st.markdown("**必須要件（Must-have）**")
                            st.write(sections.get("MUST_HAVE", ""))
                
                            st.markdown("**歓迎要件（Preferred）**")
                            st.write(sections.get("PREFERRED", ""))
                
                            st.markdown("**業務親和性（Alignment）**")
                            st.write(sections.get("ALIGNMENT", ""))
