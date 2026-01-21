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
    "Gemini 2.5 Pro (GPT-4o-mini equivalent)": "models/gemini-2.5-pro"
}
selected_model_label = st.selectbox(
    "Select AI model",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

SELECTED_MODEL = MODEL_OPTIONS[selected_model_label]

st.caption(f"Using model: `{SELECTED_MODEL}`")

st.write("Upload a candidate CV to see which jobs are most likely to result in an offer.")

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


def get_available_jobs():
    """
    Reads the client's Excel schema and builds
    a structured job context for AI evaluation.
    """
    df = pd.read_excel("jobs_new.xlsx")

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
    Gemini 3 Flash – strict JSON mode (NO truncation)
    """

    prompt = f"""
You must output VALID JSON ONLY.
Do not include explanations outside JSON.

Rules:
- Each text field must be ONE short sentence.
- Do NOT use commas inside strings.
- Do NOT use line breaks inside strings.
- Keep all values short.

JSON schema:
{{
  "score": 0,
  "summary_reason": "",
  "criteria": {{
    "must_have_requirements": {{ "rating": "○|△|×", "reason": "" }},
    "preferred_requirements": {{ "rating": "○|△|×", "reason": "" }},
    "role_alignment": {{ "rating": "○|△|×", "reason": "" }}
  }}
}}

Candidate CV:
{cv_text[:2000]}

Job description:
{job["job_context"][:1000]}
"""


    try:
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 800,
                
            }
        )
        if not response.candidates:
            raise ValueError("No candidates returned by Gemini")
        
        candidate = response.candidates[0]
        
        if not candidate.content or not candidate.content.parts:
            raise ValueError(
                f"Gemini returned no content. finish_reason={candidate.finish_reason}"
            )
        
        raw = candidate.content.parts[0].text

        if not raw:
            raise ValueError("Empty response from Gemini")

        parsed = extract_json(raw)


        return {
            "ok": True,
            "data": parsed,
            "raw": raw
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "raw": raw if "raw" in locals() else None,
            "data": {
                "score": 0,
                "summary_reason": "評価を完了できませんでした。",
                "criteria": {
                    "must_have_requirements": {"rating": "×", "reason": "評価失敗"},
                    "preferred_requirements": {"rating": "×", "reason": "評価失敗"},
                    "role_alignment": {"rating": "×", "reason": "評価失敗"},
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

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Extracting CV text..."):
        cv_text = extract_text(uploaded_file)

    if not cv_text.strip():
        st.error("Could not extract text from this file.")
        st.stop()

    st.subheader("Extracted CV Text")
    st.text_area("", cv_text, height=250)

    if st.button("Evaluate Candidate"):
        jobs = get_available_jobs()
        results = []

        progress = st.empty()
        total_jobs = len(jobs)

        for i, job in enumerate(jobs, start=1):
            progress.info(f"Evaluating job {i} of {total_jobs}: {job['title']}")
            result = ai_match_job(cv_text, job, SELECTED_MODEL)


            if not result["ok"]:
                st.error("AI MATCH ERROR")
                st.text(result["error"])
                if result["raw"]:
                    st.text(result["raw"][:3000])
            
            parsed = result["data"]


            results.append({
                "job": job,
                "score": result["data"]["score"],
                "reason": result["data"]["summary_reason"],
                "criteria": result["data"]["criteria"]
            })


        results.sort(key=lambda x: x["score"], reverse=True)

        st.subheader("Job Match Results")

        for r in results:
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

            st.write(r["reason"])

            with st.expander("Evaluation details"):
                for key, label in [
                    ("must_have_requirements", "Must-have requirements"),
                    ("preferred_requirements", "Preferred requirements"),
                    ("role_alignment", "Role alignment")
                ]:
                    st.write(label, r["criteria"][key]["rating"])
                    st.write(r["criteria"][key]["reason"])

            st.divider()

        best = results[0]
        st.success(
            f"Best match: **{best['job']['title']}** ({best['score']}%)"
        )
