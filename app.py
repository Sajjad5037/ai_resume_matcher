import os
import streamlit as st
from pypdf import PdfReader
from docx import Document
import io
import pandas as pd
import json
from openai import OpenAI


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

st.title("AI Resume (4:32)Matcher")
st.write("Upload a candidate CV to see which jobs fit best.")


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
    Reads the client's real Excel schema and builds
    a structured job context for AI evaluation.
    """
    
    
    df = pd.read_excel("jobs.xlsx")

    
    # Normalize column names (VERY important)
    df.columns = (
        df.columns
          .astype(str)
          .str.strip()
    )

    jobs = []

    def safe(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    
    for _, row in df.iterrows():

        # ----------------------------
        # Title handling (robust)
        # ----------------------------
        title = get_title(row)

        # ----------------------------
        # Build job context
        # ----------------------------
        job_context_parts = [
            f"Job Title: {safe(row.get('title'))}",
            f"Position: {safe(row.get('position'))}",
            f"Industry: {safe(row.get('job_industry'))}",
            f"Job Type: {safe(row.get('job_type'))}",
            f"Location: {safe(row.get('location'))}",
            f"Job Description: {safe(row.get('job_content'))}",
            f"Required Experience: {safe(row.get('required_experience'))}",
            f"Desired Experience: {safe(row.get('desired_experience'))}",
            f"Experience Needed: {safe(row.get('experience_needed'))}",
            f"Education: {safe(row.get('education'))}",
            f"Eligibility Details: {safe(row.get('eligibility_details'))}",
            f"Foreigner Status: {safe(row.get('foreigner_status'))}",
            f"Age Range: {safe(row.get('age_lowest'))} - {safe(row.get('age_highest'))}",
            f"Target Candidate: {safe(row.get('target_candidate'))}",
        ]

        job_context = "\n".join(
            part for part in job_context_parts
            if part.split(": ", 1)[1]
        )

        # ----------------------------
        # Final job object
        # ----------------------------
        jobs.append({
            "job_id": safe(row.get("job_url")),
            "title": title,
            "job_context": job_context,
            "company_name": safe(row.get("company_name")),
            "annual_income": safe(row.get("annual_income")),
            "passrate_for_doc_screening": safe(row.get("passrate_for_doc_screening")),
            "documents_to_job_offer_ratio": safe(row.get("documents_to_job_offer_ratio")),
            "fee": safe(row.get("fee")),
        })


    return jobs



def ai_match_job(cv_text, job):
    """
    Uses OpenAI to evaluate CV vs full job context.
    """
    prompt = f"""
You are an excellent career advisor working at a professional recruitment agency.

Your task is to evaluate how well the following candidate fits the job below and estimate the likelihood that this candidate would receive an offer.

Evaluate the candidate using the following three criteria. For each criterion, use one of the following ratings: ○ (meets well), △ (partially meets), × (does not meet).

(1) Must-have requirements  
- Primarily refer to: required_experience  
- Determine whether the candidate satisfies the essential requirements for this role.

(2) Preferred requirements  
- Primarily refer to: desired_experience and target_candidate  
- Determine whether the candidate matches the preferred or ideal profile.

(3) Role responsibility alignment  
- Primarily refer to: job_content  
- Determine whether the candidate’s past experience is highly aligned with the actual responsibilities of the role.

For EACH of the three criteria:
- Provide the rating (○ / △ / ×)
- Provide a short explanation for your evaluation

Then, based on the overall assessment:
- Estimate the likelihood of the candidate receiving an offer for this job (0–100%)
- Explain the reasoning behind this estimate clearly and concisely

Return ONLY valid JSON in the following format (no markdown, no extra text):

{{
  "score": number between 0 and 100,
  "summary_reason": "overall explanation for the estimated offer likelihood",
  "criteria": {{
    "must_have_requirements": {{
      "rating": "○ | △ | ×",
      "reason": "short explanation"
    }},
    "preferred_requirements": {{
      "rating": "○ | △ | ×",
      "reason": "short explanation"
    }},
    "role_alignment": {{
      "rating": "○ | △ | ×",
      "reason": "short explanation"
    }}
  }}
}}

Candidate CV:
\"\"\"
{cv_text[:6000]}
\"\"\"

Job Information:
Title: {job["title"]}
Company Name: {job.get("company_name", "")}
Job URL: {job.get("job_id", "")}
Document Screening Pass Rate: {job.get("passrate_for_doc_screening", "")}
Offer Rate (Documents to Offer): {job.get("documents_to_job_offer_ratio", "")}
Fee: {job.get("fee", "")}

Job Description and Requirements:
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

        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        result = json.loads(raw)
        return int(result["score"]), result["summary_reason"]



    except Exception as e:
        return 0, "AI service temporarily unavailable."


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

    st.divider()

    if st.button("Evaluate Candidate"):
        jobs = get_available_jobs()
        results = []

        progress_text = st.empty()

        total_jobs = len(jobs)
        
        for i, job in enumerate(jobs, start=1):
            progress_text.info(f"Evaluating job {i} of {total_jobs}: {job['title']}")
            score, reason = ai_match_job(cv_text, job)


            results.append({
                "Job Title": job["title"],
                "Match Score (%)": score,
                "Reason": reason
            })

        if not results:
            st.warning("No jobs found to evaluate.")
            st.stop()

        results = sorted(
            results,
            key=lambda x: x["Match Score (%)"],
            reverse=True
        )

        st.subheader("Job Match Results")
        st.table(results)

        best = results[0]
        st.success(
            f"Best match: **{best['Job Title']}** ({best['Match Score (%)']}%)"
        )
