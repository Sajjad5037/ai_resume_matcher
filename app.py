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
import zipfile
from tempfile import TemporaryDirectory
from pathlib import Path

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
    st.session_state.results = None


st.success("Gemini API key loaded successfully.")


st.title("AI Resume Matcher (11:40)")



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

def extract_cvs_from_zip(uploaded_zip):
    if "cv_tmpdir" not in st.session_state:
        st.session_state.cv_tmpdir = TemporaryDirectory()

    tmpdir = Path(st.session_state.cv_tmpdir.name)

    zip_path = tmpdir / "cvs.zip"
    zip_path.write_bytes(uploaded_zip.read())

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmpdir)

    cv_paths = [
        path for path in tmpdir.rglob("*")
        if path.suffix.lower() in (".pdf", ".docx", ".xlsx")
    ]

    return cv_paths

def extract_text_from_excel(path: Path) -> str:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        return ""

    blocks = []

    for sheet_name, df in sheets.items():
        blocks.append(f"【Sheet: {sheet_name}】")

        for _, row in df.iterrows():
            row_text = " ".join(
                str(cell) for cell in row.values if not pd.isna(cell)
            )
            if row_text.strip():
                blocks.append(row_text)

    return "\n".join(blocks)

def extract_cv_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()

    # ✅ PDF
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
        except Exception:
            return ""

    # ✅ DOCX
    if suffix == ".docx":
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            return ""

    # ✅ XLSX
    if suffix == ".xlsx":
        return extract_text_from_excel(path)

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
            "max_output_tokens": 700,
        }
    )

    # Parse JSON safely
    try:
        return json.loads(response.text)
    except Exception:
        # Absolute fallback to prevent UI breakage
        return {
            "SUMMARY": response.text.strip(),
            "MUST_HAVE": "必須要件について大きな不足は見られません。",
            "PREFERRED": "歓迎要件についても一定の適合性が確認できます。",
            "ALIGNMENT": "業務内容との親和性は高いと判断されます。"
        }


def generate_with_retry(model, prompt, retries=2):
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
uploaded_zip = st.file_uploader(
    "Upload CV folder (ZIP containing PDF / DOCX / XLSX)",
    type=["zip"]
)
jobs_file = st.file_uploader(
    "Upload jobs Excel file",
    type=["xlsx"],
    key="jobs_excel"
)

if uploaded_zip:
    # 🔑 CLEAR OLD RESULTS WHEN A NEW ZIP IS UPLOADED
    st.session_state.results = None

    st.success("CV folder uploaded")

    cvs = extract_cvs_from_zip(uploaded_zip)
    if not cvs:
        st.error("No PDF, DOCX, or XLSX files found in the ZIP.")
        st.stop()

    st.info(f"{len(cvs)} CVs found in folder")

    if jobs_file and st.button("Evaluate CV Folder"):
        jobs_df = pd.read_excel(jobs_file)
        jobs = get_available_jobs(jobs_df)

        folder_results = []
        progress = st.progress(0)

        for idx, cv_path in enumerate(cvs, start=1):
            cv_text = extract_cv_text_from_path(cv_path)

            if not cv_text.strip():
                continue

            cv_results = []

            for job in jobs:
                result = ai_match_job(cv_text, job, SELECTED_MODEL)

                cv_results.append({
                    "job": job,
                    "score": result["data"]["score"],
                    "criteria": result["data"]["criteria"]
                })

            cv_results.sort(key=lambda x: x["score"], reverse=True)

            folder_results.append({
                "cv_name": cv_path.name,
                "cv_type": cv_path.suffix.upper().replace(".", ""),
                "cv_text": cv_text,
                "results": cv_results
            })

            progress.progress(idx / len(cvs))

        st.session_state.results = folder_results

    # ✅ RENDERING = OUTSIDE BUTTON
    if st.session_state.results:
        st.subheader("CV Folder Evaluation Results")
        for cv_block in st.session_state.results:
            if not cv_block["results"]:
                st.warning(f"No valid matches for {cv_block['cv_name']}")
                continue
        
            best_job = cv_block["results"][0]
        
            st.success(
                f"Best match for {cv_block['cv_name']}: "
                f"**{best_job['job']['title']}** ({best_job['score']}%)"
            )

    
        for cv_idx, cv_block in enumerate(st.session_state.results):
            with st.expander(f"📄 {cv_block['cv_name']} ({cv_block['cv_type']})"):
                for job_idx, r in enumerate(cv_block["results"]):

                    job = r["job"]
    
                    st.markdown(f"### {job['title']}")
                    st.write(f"**Estimated Offer Probability:** {r['score']}%")
    
                    st.caption(
                        f"""
                        **Company:** {job['company_name']}  
                        **Document pass rate:** {job['passrate_for_doc_screening']}  
                        **Offer rate:** {job['documents_to_job_offer_ratio']}  
                        **Fee:** {job['fee']}  
                        **Job link:** {job['job_id']}
                        """
                    )
    
                    if st.button(
                        f"Explain – {cv_block['cv_name']} – {job['title']}",
                        key=f"explain_{cv_idx}_{job_idx}"

                    ):
                        with st.spinner("Generating explanation..."):
                            sections = generate_explanation(
                                cv_block["cv_text"],  # ✅ correct CV
                                job,
                                r
                            )
    
                            st.write(sections.get("SUMMARY", ""))
    
                            with st.expander("Evaluation details"):
                                st.markdown("**Must-have requirements ○**")
                                st.write(sections.get("MUST_HAVE", ""))
    
                                st.markdown("**Preferred requirements ×**")
                                st.write(sections.get("PREFERRED", ""))
    
                                st.markdown("**Role alignment △**")
                                st.write(sections.get("ALIGNMENT", ""))
    
                    st.divider()


        
        
            

