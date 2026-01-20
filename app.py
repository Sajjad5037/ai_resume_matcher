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

st.title("AI Resume (11:06)Matcher")
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


def get_available_jobs():
    """
    Reads the client's real Excel schema and builds
    a structured job context for AI evaluation.
    """
    df = pd.read_excel("jobs.xlsx")
    jobs = []

    for _, row in df.iterrows():

        def safe(value):
            if pd.isna(value):
                return ""
            return str(value).strip()

        # ----------------------------
        # Title handling (robust)
        # ----------------------------
        title = safe(row.get("title"))
        if not title:
            title = safe(row.get("position"))
        if not title:
            title = "Unknown Role"

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
        })

    return jobs



def ai_match_job(cv_text, job):
    """
    Uses OpenAI to evaluate CV vs full job context.
    """
    prompt = f"""
You are an experienced professional recruiter.

Evaluate how well the candidate fits the job below.

Return ONLY valid JSON in this format:
{{
  "score": number between 0 and 100,
  "reason": "short, clear explanation"
}}

Candidate CV:
\"\"\"
{cv_text[:6000]}
\"\"\"

Job Information:
Title: {job["title"]}

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
        return int(result["score"]), result["reason"]

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

        for job in jobs:
            with st.spinner(f"Evaluating {job['title']}..."):
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
