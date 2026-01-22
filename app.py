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
    st.session_state.results = None


st.success("Gemini API key loaded successfully.")


st.title("AI Resume Matcher (9:40)")



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

def parse_explanation(text):
    sections = {}

    for key in ["SUMMARY", "MUST_HAVE", "PREFERRED", "ALIGNMENT"]:
        pattern = rf"{key}:(.*?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    return sections

def generate_explanation(cv_text, job, evaluation):
    prompt = f"""
あなたは人材紹介会社の優秀なキャリアアドバイザーです。

あなたがサポートしている求職者について、
以下の評価結果をもとに、なぜこの求人が内定しやすい／しにくいのかを説明してください。

評価結果：
① 必須要件：{evaluation["criteria"]["must_have_requirements"]}
② 歓迎要件：{evaluation["criteria"]["preferred_requirements"]}
③ 業務内容との親和性：{evaluation["criteria"]["role_alignment"]}
想定内定確率：{evaluation["score"]}％

説明ルール：
- 人事・キャリアアドバイザーとしての視点で説明する
- 箇条書きは使わない
- 自然な文章で、簡潔かつ具体的に書く
- AIという言葉は使わない

出力形式（必ず守る）：

SUMMARY:
全体的な内定可能性とその理由

MUST_HAVE:
必須要件についての判断理由

PREFERRED:
歓迎要件についての判断理由

ALIGNMENT:
業務内容との親和性についての判断理由

求職者情報：
{cv_text[:2000]}

求人情報：
{job["job_context"][:1200]}
"""

    model = genai.GenerativeModel(SELECTED_MODEL)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.5,
            "max_output_tokens": 700,
        }
    )

    return response.text.strip()


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
uploaded_file = st.file_uploader(
    "Upload candidate CV (PDF or DOCX)",
    type=["pdf", "docx"]
)
jobs_file = st.file_uploader(
    "Upload jobs Excel file",
    type=["xlsx"],
    key="jobs_excel"
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

    # ✅ BUTTON = COMPUTE ONLY
    if uploaded_file and jobs_file and st.button("Evaluate Candidate"):
        jobs_df = pd.read_excel(jobs_file)
        jobs = get_available_jobs(jobs_df)

        results = []
        progress = st.empty()
        total_jobs = len(jobs)

        for i, job in enumerate(jobs, start=1):
            progress.info(f"Evaluating job {i} of {total_jobs}: {job['title']}")
            result = ai_match_job(cv_text, job, SELECTED_MODEL)

            parsed = result["data"]

            results.append({
                "job": job,
                "score": parsed.get("score", 0),
                "criteria": parsed.get("criteria", {})
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        st.session_state.results = results

    # ✅ RENDERING = OUTSIDE BUTTON
    if st.session_state.results:
        st.subheader("Job Match Results")

        for r in st.session_state.results:
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
                f"Explain this evaluation – {job['title']}",
                key=f"explain_{job['job_id']}"
            ):
                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(cv_text, job, r)
                    sections = parse_explanation(explanation)

                    st.write(sections.get("SUMMARY", ""))

                    with st.expander("Evaluation details"):
                        st.markdown("**Must-have requirements ○**")
                        st.write(sections.get("MUST_HAVE", ""))

                        st.markdown("**Preferred requirements ×**")
                        st.write(sections.get("PREFERRED", ""))

                        st.markdown("**Role alignment △**")
                        st.write(sections.get("ALIGNMENT", ""))

            st.divider()

        best = st.session_state.results[0]
        st.success(
            f"Best match: **{best['job']['title']}** ({best['score']}%)"
        )
        
            

