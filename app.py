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
import mimetypes

def to_gemini_part(uploaded_file):
    uploaded_file.seek(0)

    # Guess MIME type from filename (more reliable than browser)
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)

    # Hard fallback for PDFs
    if uploaded_file.name.lower().endswith(".pdf"):
        mime_type = "application/pdf"

    # Final fallback (should almost never be used)
    if not mime_type:
        mime_type = uploaded_file.type

    return {
        "mime_type": mime_type,
        "data": uploaded_file.read(),
    }



 
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


st.title("AI Resume Matcher (old)")



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
    "Gemini 3 Flash (Preview)": "models/gemini-3-flash-preview"
}
selected_model_label = st.selectbox(
    "Select AI model",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

SELECTED_MODEL = MODEL_OPTIONS[selected_model_label]

st.caption(f"Using model: `{SELECTED_MODEL}`")

st.write("Upload a candidate CV to see which jobs are most likely to result in an offer.")
st.write(
    "📌 For best accuracy, upload CVs in DOCX or text-based PDF format. "
    "Scanned PDFs may reduce matching quality."
)


def extract_cv_text_from_uploaded_file(uploaded_file) -> str:
    uploaded_file.seek(0)
    file_type = uploaded_file.type

    # ----------------
    # PDF (SAFE)
    # ----------------
    if file_type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = []
        failed_pages = 0

        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text.append(page_text)
            except Exception:
                failed_pages += 1
                continue

        if failed_pages > 0:
            st.warning(
                f"⚠️ {uploaded_file.name}: {failed_pages} page(s) could not be read and were skipped."
            )

        return "\n".join(text)

    # ----------------
    # DOCX
    # ----------------
    if file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # ----------------
    # XLSX
    # ----------------
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


def safe_parse_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON found")
    return json.loads(text[start:end + 1])



def aggregate_candidate_cv_text(uploaded_files):
    combined_blocks = []
    filenames = []

    for file in uploaded_files:
        try:
            text = extract_cv_text_from_uploaded_file(file)
        except Exception:
            st.error(f"❌ Failed to process {file.name}")
            continue

        if text.strip():
            combined_blocks.append(f"\n\n--- {file.name} ---\n{text}")
            filenames.append(file.name)
        else:
            st.info(
                f"ℹ️ {file.name} contains little or no readable text."
            )

    return {
        "cv_text": "\n".join(combined_blocks),
        "filenames": filenames
    }

def generate_explanation(job, evaluation):


    score = evaluation["score"]

    # Create a prompt based on client requirements
    prompt = f"""
Return ONLY valid JSON.
Do not include markdown.
Do not include any text outside JSON.
Do not add extra keys.

あなたは、採用・書類選考の実務経験が豊富な人材アドバイザーです。
以下の「評価結果」のみを根拠として、
その判断理由を採用担当者向けに説明してください。

※ あなたは履歴書を再評価してはいけません。
※ 新しい判断や推測を行ってはいけません。
※ 下記の評価結果を説明・言語化することだけが目的です。

当該職種における候補者の内定可能性について、採用担当者向けに
客観的かつ丁寧な評価コメントを作成してください。

【必須ルール】
- 出力はすべて日本語で記述してください。
- 自然で客観的なビジネス日本語を使用してください。
- 箇条書きは使用せず、文章形式で記述してください。
- AI、モデル、システムに関する言及は禁止です。
- 各フィールドは必ず1文以上の完全な文章で記述してください。
- 内容が不明な場合でも、空欄にはせず、評価文として成立させてください。

【評価の前提】
- 評価は、提供されたCVに明示的に記載されている内容のみを根拠としてください。
- 推測や補完は行わないでください。
- 各評価は、採用担当者が社内共有できる説明として成立する内容にしてください。

【評価コンテキスト】
- 必須要件の評価：{evaluation["criteria"]["must_have_requirements"]}
- 歓迎要件の評価：{evaluation["criteria"]["preferred_requirements"]}
- 職務内容との適合性：{evaluation["criteria"]["role_alignment"]}
- 想定内定確率：{score}％

【出力JSON形式（厳守）】
{{
  "SUMMARY": "",
  "MUST_HAVE": "",
  "PREFERRED": "",
  "ALIGNMENT": ""
}}


【職務内容】
{job["job_context"][:1200]}
"""

    model = genai.GenerativeModel(SELECTED_MODEL)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 900,
        }
    )


    try:
        return safe_parse_json(response.text)

    except Exception:
        # Hard fallback in case of error
        if score == 0:
            return {
                "SUMMARY": "提供された履歴書の内容からは、当該職種において内定に至る可能性は現時点では低いと判断されます。",
                "MUST_HAVE": "必須要件に該当する明確な経験や根拠が履歴書上で確認できませんでした。",
                "PREFERRED": "歓迎要件についても、直接的な適合性は限定的であると考えられます。",
                "ALIGNMENT": "職務内容との直接的な一致は確認できず、業務適合性は低いと判断されます。"
            }

        return {
            "SUMMARY": "履歴書の内容を総合的に判断すると、一部に評価可能な要素はあるものの、内定可能性は限定的であると考えられます。",
            "MUST_HAVE": "必須要件については一部満たしている可能性はあるものの、十分な根拠は確認できませんでした。",
            "PREFERRED": "歓迎要件については限定的な適合性が確認できます。",
            "ALIGNMENT": "業務内容との親和性は一定程度確認できますが、決定的とは言えません。"
        }


def generate_with_retry(model, prompt, candidate_files, retries=1):
    last_error = None

    for attempt in range(1, retries + 1):
        response = model.generate_content(
            prompt,  # ✅ now defined via parameter
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 900,
            }
        )

        candidate = response.candidates[0]
        raw = candidate.content.parts[0].text

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


def ai_match_job(candidate_files, job, model_name):

    
    prompt = f"""
Return ONLY valid JSON.
No markdown.
No explanations.
No text outside JSON.

All string values MUST be single-line.
Do NOT include newline characters inside strings.

あなたは、書類選考を担当する採用実務者です。
以下の履歴書（CV）と職務内容を比較し、
CVに明示的に記載されている内容のみを根拠として評価してください。

【重要ルール】
- 推測や補完は禁止です。
- 間接的・汎用的な関連経験が確認できる場合は「△」としてください。
- CV に関連する根拠が一切確認できない場合のみ「×」としてください。
- 不足や未経験を断定してはいけません。

【評価記号】
○：直接的かつ職務関連性の高い経験が確認できる  
△：間接的・汎用的・限定的な関連経験が確認できる  
×：関連する経験や根拠が一切確認できない  

【出力JSON形式（厳守）】
{
  "score": 0,
  "criteria": {
    "must_have_requirements": "○|△|×",
    "preferred_requirements": "○|△|×",
    "role_alignment": "○|△|×"
  }
}

【スコア算出ルール】
- score は 0 から 100 の整数で返してください。
- 上記 criteria の評価結果を総合して score を算出してください。
- CV に明示的な根拠がほとんど確認できない場合は、低い score を返してください。

【職務内容】
{job["job_context"][:1500]}
"""



    try:
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
           [prompt, *candidate_files],
           generation_config={
               "temperature": 0.3,
               "max_output_tokens": 900,
           }
       )

        
        # Log candidate count and content parts
        if not response.candidates:
            raise ValueError("Gemini returned no candidates")
        
        content_parts = response.candidates[0].content.parts
        
        st.caption(
            f"🧠 Gemini response parts: {len(content_parts)} "
            f"(includes prompt + {len(candidate_files)} document(s))"
        )
        
        raw = response.text
        parsed = extract_json(raw)
        # Defensive normalization
        if not isinstance(parsed.get("score"), int):
            parsed["score"] = 0
        # 🔍 Heuristic warning: likely ingestion / readability issue
        if (
            parsed.get("score", 0) == 0 and
            all(v == "×" for v in parsed.get("criteria", {}).values())
        ):
            st.warning(
                "⚠️ The evaluation returned no matching signals. "
                "This may indicate the CV content was not fully readable by the model."
            )



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

    # 🔹 Aggregate ALL CVs into ONE candidate
    candidate_files = []

    st.subheader("📎 Document ingestion log")
    
    for f in st.session_state.cvs:
        part = to_gemini_part(f)
    
        candidate_files.append(part)
    
        # UI log (very important for debugging)
        st.code(
            {
                "filename": f.name,
                "detected_mime_type": part["mime_type"],
                "file_size_kb": round(len(part["data"]) / 1024, 2),
            },
            language="json"
        )


    cv_files = [f.name for f in st.session_state.cvs]
    st.session_state.candidate_files = candidate_files


    
    status.info("Evaluating candidate profile (combined documents)")
    
    cv_results = []
    
    for job_idx, job in enumerate(jobs, start=1):
        status.info(f"Evaluating Job {job_idx}/{len(jobs)}")
    
        result = ai_match_job(candidate_files, job, SELECTED_MODEL)

    
        cv_results.append({
            "job": job,
            "score": result["data"]["score"],
            "criteria": result["data"]["criteria"]
        })
    
    cv_results.sort(key=lambda x: x["score"], reverse=True)
    
    st.session_state.results = [{
        "cv_name": "Combined Candidate Profile",
        "cv_type": "MULTI-DOC",
        "cv_files": cv_files,
        "results": cv_results
    }]
    
    status.success("Evaluation completed")
    progress.progress(1.0)

    
    

    
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

        st.markdown(f"## 📄 {cv_block['cv_name']} ({cv_block['cv_type']})")
        st.divider()
               
        candidate_container = st.container()
        
        with candidate_container:
            st.markdown("**Uploaded Documents:**")
            for name in cv_block.get("cv_files", []):
                st.markdown(f"- {name}")
        
            st.caption(
                "※ アップロードされた履歴書ファイルは、AIが直接読み取り・評価しています。"
                " 事前のテキスト抽出やOCR処理は行っていません。"
            )

        
            for job_idx, r in enumerate(cv_block["results"]):
                job = r["job"]
        
                st.markdown(f"### {job['title']}")
                st.write(f"**Estimated Offer Probability:** {r['score']}%")
        
                cols = st.columns(3)
                cols[0].markdown(f"**会社名**<br>{job['company_name']}", unsafe_allow_html=True)
                cols[1].markdown(
                    f"**書類通過率**<br>{job['passrate_for_doc_screening']}%",
                    unsafe_allow_html=True
                )
                cols[2].markdown(
                    f"**内定率**<br>{job['documents_to_job_offer_ratio']}",
                    unsafe_allow_html=True
                )

        
                explain_key = f"{cv_idx}_{job_idx}"
        
                if explain_key not in st.session_state.explain_open:
                    st.session_state.explain_open[explain_key] = False
        
                if st.button(
                    f"分析詳細（この評価の理由） – {job['title']}",
                    key=f"explain_btn_{explain_key}"
                ):
                    st.session_state.explain_open[explain_key] = True

                    if explain_key not in st.session_state.explanations:
                        st.session_state.explanations[explain_key] = generate_explanation(job, r)

        
                if st.session_state.explain_open.get(explain_key, False):
                    sections = st.session_state.explanations.get(explain_key)
                    if sections:
                        st.markdown("### 📝 評価サマリー")
                        st.write(sections.get("SUMMARY", ""))
        
                        with st.expander("📊 評価詳細", expanded=True):

                            st.markdown("**必須要件（Must-have）**")
                            st.write(sections.get("MUST_HAVE", ""))
                            st.markdown("**歓迎要件（Preferred）**")
                            st.write(sections.get("PREFERRED", ""))
                            st.markdown("**業務親和性（Alignment）**")
                            st.write(sections.get("ALIGNMENT", ""))
        
        
        
                    
