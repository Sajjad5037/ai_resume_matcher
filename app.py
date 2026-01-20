import streamlit as st
from pypdf import PdfReader
from docx import Document
import io
import re

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="centered"
)

st.title("AI Resume Matcher")
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
    Mock data for now.
    This will later be replaced by Google Apps Script API.
    """
    return [
        {
            "job_id": "J1",
            "title": "Backend Developer",
            "keywords": ["python", "api", "fastapi", "database", "postgresql"]
        },
        {
            "job_id": "J2",
            "title": "Frontend Developer",
            "keywords": ["react", "javascript", "css", "html"]
        },
        {
            "job_id": "J3",
            "title": "Data Analyst",
            "keywords": ["sql", "excel", "data", "analysis", "dashboard"]
        }
    ]


def simple_match_score(cv_text, job):
    """
    Placeholder logic.
    This WILL be replaced by AI later.
    """
    text = cv_text.lower()
    matches = sum(1 for kw in job["keywords"] if re.search(rf"\b{kw}\b", text))
    score = int((matches / len(job["keywords"])) * 100)
    return score


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
    st.text_area(
        label="",
        value=cv_text,
        height=250
    )

    st.divider()

    if st.button("Evaluate Candidate"):
        jobs = get_available_jobs()
        results = []

        for job in jobs:
            score = simple_match_score(cv_text, job)
            results.append({
                "Job Title": job["title"],
                "Match Score (%)": score
            })

        results = sorted(results, key=lambda x: x["Match Score (%)"], reverse=True)

        st.subheader("Job Match Results")
        st.table(results)

        best = results[0]
        st.success(
            f"Best match: **{best['Job Title']}** ({best['Match Score (%)']}%)"
        )
