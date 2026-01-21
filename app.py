import os
import streamlit as st
from pypdf import PdfReader
from docx import Document
import io
import pandas as pd
import json
from openai import OpenAI
import re

# ----------------------------
# OpenAI setup
# ----------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is NOT loaded. Check Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="centered"
)

st.success("OPENAI_API_KEY loaded successfully.")

st.title("AI Resume Matcher (9:11)")
st.write("Upload a candidate CV to see which jobs are most likely to result in an offer.")

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())
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


def ai_match_job(cv_text, job):
    """
    Uses OpenAI to evaluate CV vs job context.
    Pure logic function – NO Streamlit calls inside.
    """
    prompt = f"""
You are a senior career advisor at a professional recruitment agency with deep experience in CV screening and hiring decisions.

Your task is to evaluate how likely the following candidate is to receive a job offer for the specific role described below, based STRICTLY on the evidence present in the CV text provided.

IMPORTANT EVIDENCE & RATING RULES (MANDATORY):
- You MUST ground every evaluation in explicit evidence from the CV text.
- You are NOT allowed to state that a candidate lacks a skill, industry experience, or qualification if it is explicitly mentioned anywhere in the CV text.
- If the CV shows INDIRECT, TRANSFERABLE, or SALES-ADJACENT experience (e.g. negotiation, persuasion, stakeholder approval, client-facing work), you MUST acknowledge it explicitly.
- INDIRECT or TRANSFERABLE experience MUST be rated as △ (partial) — NOT ×.
- You may ONLY assign × when the CV contains NO relevant evidence at all, including indirect or transferable experience.
- If evidence exists but is informal, brief, or not part of a formal job title, describe it as “indirect” or “non-traditional” rather than “absent”.
- If you cannot find any relevant evidence, you MUST say: "No clear evidence found in the provided CV text".
- False negatives (claiming absence when indirect evidence exists) are considered a serious evaluation error and must be avoided.

Evaluate the candidate using the three criteria below.

For each criterion, assign a rating using:
○ = Strongly meets requirements through direct, role-specific experience  
△ = Partially meets requirements through indirect, transferable, or limited experience  
× = Does not meet requirements due to complete absence of relevant evidence  

CRITERIA:
1. Must-have requirements (required_experience)
2. Preferred requirements (desired_experience, target_candidate)
3. Role responsibility alignment (job_content)

FOR EACH CRITERION, YOU MUST:
- Quote or clearly reference the specific CV evidence used
- Explain how this evidence supports the chosen rating
- If experience is indirect or transferable, explicitly label it as such
- Include ONE sentence explaining how a recruiter could communicate this evaluation constructively to the candidate

AFTER EVALUATING ALL CRITERIA:
- Write a concise overall summary synthesizing strengths, limitations, and hiring risk
- Estimate the overall probability of receiving a job offer (0–100 percent), using realistic hiring standards for this role

OUTPUT FORMAT RULES (STRICT):
- Return ONLY valid JSON
- Do NOT include any text outside the JSON object
- Do NOT copy example values

Required JSON structure:

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
\"\"\"
{cv_text[:6000]}
\"\"\"

Job Information:
Title: {job["title"]}
Company: {job["company_name"]}
Job URL: {job["job_id"]}
Document Pass Rate: {job["passrate_for_doc_screening"]}
Offer Rate: {job["documents_to_job_offer_ratio"]}
Fee: {job["fee"]}

Job Description:
{job["job_context"]}
"""


    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You evaluate candidate-job fit objectively."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()

        # Remove markdown fences if present
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()

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
                "summary_reason": "AI service temporarily unavailable.",
                "criteria": {
                    "must_have_requirements": {"rating": "×", "reason": "Evaluation failed"},
                    "preferred_requirements": {"rating": "×", "reason": "Evaluation failed"},
                    "role_alignment": {"rating": "×", "reason": "Evaluation failed"},
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
            result = ai_match_job(cv_text, job)

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
