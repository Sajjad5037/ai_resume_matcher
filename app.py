import os 
import streamlit as st

import io
import pandas as pd
import json 
#from openai import OpenAI
import re
import google.generativeai as genai


import mimetypes
st.write(
    "📌 For best accuracy, upload CVs in DOCX or text-based PDF format. "
    "Scanned PDFs may reduce matching quality."
)


def generate_full_assessment(candidate_files, job, model_name, candidate_seniority):
    """
    Single-pass advisor-style evaluation.
    CV + JD are read together.
    Explanation first, score last.
    UI-compatible output keys.
    """

    prompt = f"""
Return ONLY valid JSON.
No markdown.
No text outside JSON.

あなたは、キャリアアドバイザー兼採用担当者です。
以下の履歴書（CV）を丁寧に読み、
この求人に対して「なぜそう評価したのか」が
第三者にも分かるように説明してください。

【重要な前提】
- 評価は書類選考段階のものです。
- CVに明示的に記載されている内容のみを根拠にしてください。
- 推測や断定は禁止です。
- ENTRY求人では経験不足を否定的に扱ってはいけません。

【求人レベル】
{job["seniority"]}

【候補者レベル】
{candidate_seniority}

【職務内容】
{job["job_context"][:1500]}

【出力JSON形式（厳守）】
{{
  "SUMMARY": "",
  "MUST_HAVE": "",
  "PREFERRED": "",
  "ALIGNMENT": "",
  "score": 0
}}

【各項目の意味】
- SUMMARY：全体評価（なぜこのような判断になったか）
- MUST_HAVE：履歴書から確認できる主な強み・評価できる点
- PREFERRED：現時点で懸念となり得る点や不足している可能性のある要素
- ALIGNMENT：本求人との役割・期待値の適合性

【スコアについて】
- 0〜100 の整数で返してください
- 上記の評価内容と整合する数値にしてください
"""
    
    model = genai.GenerativeModel(model_name)

    content_parts = prompt


    
    response = model.generate_content(
        content_parts,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 900,
        }
    )


    raw = response.text
    parsed = extract_json(raw)
    return parsed

def get_display_score(score: int, seniority: str) -> int:
    """
    UI-safe score display.
    Does NOT affect AI logic.
    """
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
        mime_type = uploaded_file.type or "application/octet-stream"

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
st.subheader("✅ Available Gemini models for this project")
st.code(available_models)


# ----------------------------
# Model Selection
# ----------------------------
MODEL_OPTIONS = {
    "Gemini 1.5 Pro": "models/gemini-1.5-pro"
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




def safe_parse_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON found")
    return json.loads(text[start:end + 1])




def generate_explanation(job, evaluation, candidate_seniority):

    score = evaluation["score"]

    # 🔑 Core intent flag (THIS WAS MISSING)
    is_overqualified_for_entry = (
        job["seniority"] == "ENTRY"
        and candidate_seniority in ["MID", "SENIOR"]
    )

    prompt = f"""
Return ONLY valid JSON.
Do not include markdown.
Do not include any text outside JSON.
Do not add extra keys.

【求人レベル】
この求人は「{job['seniority']} レベル」の募集です。

【候補者レベル】
この候補者は「{candidate_seniority} レベル」と推定されます。

【スコアの意味（厳守）】
- 想定内定確率は書類選考段階での可能性を示します。
- ENTRY 求人において、経験不足を前提とした表現は禁止です。

【レベル差に関する表現ルール（最重要）】
- 候補者レベルが求人レベルを上回る場合：
  以下の表現のみを使用してください。
  ・役割期待の違い
  ・業務範囲・責任設計の相違
  ・ポジション特性とのミスマッチ

  以下の表現は絶対に使用してはいけません。
  ・育成
  ・学習
  ・成長次第
  ・判断材料が限られている

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
        generation_config={"temperature": 0.3, "max_output_tokens": 900}
    )

    try:
        result = safe_parse_json(response.text)
        # 🔍 DEBUG: confirm level pairing
        
        # 🚨 HARD OVERRIDE: ENTRY job + non-ENTRY candidate
        if is_overqualified_for_entry:

            result["ALIGNMENT"] = (
                "本求人はENTRYレベルの役割設計となっており、"
                "候補者の経験水準や期待役割との間に差異が見られます。"
                "業務範囲や責任設計の観点から、役割期待の整理が必要です。"
            )
        
            # Also clean MUST_HAVE if needed
            if "育成" in result.get("MUST_HAVE", ""):
                result["MUST_HAVE"] = (
                    "必須要件に関連する経験は確認できますが、"
                    "本求人で想定されている役割水準とは一部異なる可能性があります。"
                )
            if (
                "育成" in result.get("SUMMARY", "")
                or "成長" in result.get("SUMMARY", "")
            ):
                result["SUMMARY"] = (
                    "履歴書の内容から、当該職種に対して一定の可能性は見られますが、"
                    "本求人の役割設計との間に期待値の違いが見受けられます。"
                )
        
        return result
        

    except Exception:
        # ---------- FALLBACK (STRICT & INTENT-AWARE) ----------

        # ❌ Low score & non-entry → strict allowed
        if score < 20 and job["seniority"] != "ENTRY":
            return {
                "SUMMARY": "提供された履歴書の内容から、当該職種において現時点での内定可能性は限定的であると考えられます。",
                "MUST_HAVE": "必須要件に該当する職務経験や成果が、履歴書上では十分に確認できませんでした。",
                "PREFERRED": "歓迎要件についても、直接的な適合性を示す記載は限定的です。",
                "ALIGNMENT": "職務内容との直接的な一致は確認できず、役割要件との乖離が見られます。"
            }

        # ⚠️ Medium score
        elif score < 60:

            # ✅ ENTRY job but candidate is ABOVE entry → role mismatch
            if is_overqualified_for_entry:
                return {
                    "SUMMARY": "履歴書の内容から、当該職種に対して一定の可能性は見られるものの、役割設計との観点で慎重な検討が必要と考えられます。",
                    "MUST_HAVE": "必須要件に関連する経験や実績は確認できますが、本求人で想定されている役割水準とは一部乖離が見られます。",
                    "PREFERRED": "歓迎要件については活かせる要素が確認できますが、業務範囲や期待役割との整理が必要です。",
                    "ALIGNMENT": "本求人はENTRYレベルの役割設計となっており、候補者の経験水準とは役割期待の違いが生じている可能性があります。"
                }

            # ✅ True ENTRY candidate → growth framing allowed
            if candidate_seniority == "ENTRY" and job["seniority"] == "ENTRY":
                return {
                    "SUMMARY": "履歴書の内容から、当該職種において一定の検討余地があると考えられます。",
                    "MUST_HAVE": "必須要件については明確な経験の記載は限定的ですが、育成や学習によって補完可能な余地があります。",
                    "PREFERRED": "歓迎要件については一部に関連性が見られるものの、限定的な内容にとどまっています。",
                    "ALIGNMENT": "職務内容との親和性については、今後の育成過程を踏まえた評価が想定されます。"
                }

            # MID / SENIOR normal case
            return {
                "SUMMARY": "履歴書の内容から、当該職種において一定の可能性が見られます。",
                "MUST_HAVE": "必須要件に関連する実務経験や成果は確認できますが、求人で想定されている役割や期待との間に一部差異が見られます。",
                "PREFERRED": "歓迎要件については、職務に活かせる要素が一部確認できます。",
                "ALIGNMENT": "職務内容との適合性については、役割期待の違いを踏まえた検討が必要です。"
            }

        # ✅ High score → positive framing
        else:
            return {
                "SUMMARY": "履歴書の内容から、当該職種との適合性が一定程度確認でき、前向きに検討できる可能性があります。",
                "MUST_HAVE": "必須要件については、職務に関連する十分な経験や成果が確認できます。",
                "PREFERRED": "歓迎要件についても、評価可能な要素が含まれています。",
                "ALIGNMENT": "職務内容との親和性は比較的高く、業務への適応が期待されます。"
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
def detect_candidate_seniority_from_cv(candidate_files):
    # Document-based seniority inference is intentionally disabled
    # because we rely on Gemini's internal document understanding.
    return "ENTRY"


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
def calculate_score(criteria: dict, seniority: str) -> int:
    weights = {
        "○": 1.0,
        "△": 0.6,
        "×": 0.0,
    }

    raw = (
        weights.get(criteria.get("must_have_requirements"), 0) * 0.4 +
        weights.get(criteria.get("preferred_requirements"), 0) * 0.3 +
        weights.get(criteria.get("role_alignment"), 0) * 0.3
    )

    score = int(raw * 100)

    # 🎯 Seniority-based soft floors
    if seniority == "ENTRY":
        score = max(score, 35)
    elif seniority == "MID":
        score = max(score, 20)
    elif seniority == "SENIOR":
        score = max(score, 10)

    return min(score, 100)
    


def ai_match_job(candidate_files, job, model_name, candidate_seniority):

    


    

    
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

【求人レベル】
この求人は「{job['seniority']} レベル」に分類されます。

【候補者レベル】
この候補者は「{candidate_seniority} レベル」と推定されます。

【レベル差に関する重要ルール（必ず遵守）】
- 候補者レベルが求人レベルを上回る場合：
  経験不足として評価してはいけません。
  業界の違い、役割期待の違い、ポジションのミスマッチとして説明してください。
- 候補者レベルが求人レベルと同等または下回る場合：
  求人要件に対する経験の有無を通常どおり評価してください。

【レベル別評価方針】
- ENTRY 求人の場合：
  実務経験の欠如はマイナス評価にしてはいけません。
  学習・育成によって補完可能な要素は「△」として前向きに評価してください。
- MID / SENIOR 求人の場合：
  職務に直接関連する実務経験の有無をより重視してください。

【評価の基本方針】
- 評価は、CVに明示的に記載されている事実のみを根拠としてください。
- 推測、補完、過度な解釈は禁止です。
- 未記載の内容について、断定的な否定を行ってはいけません。

【評価ルール】
- 職務に直接関連する明確な経験が確認できる場合は「○」としてください。
- 間接的・汎用的・限定的な関連経験が確認できる場合は「△」としてください。
- CV上に関連する根拠が一切確認できない場合のみ「×」としてください。

【重要な補足ルール（必ず遵守）】
- ENTRY 求人においては、
  必須要件における直接経験の欠如をマイナス評価として扱ってはいけません。
- 明確な不一致が確認できない限り、
  職務内容との適合性を「低い（×）」と断定してはいけません。
- 経験が十分に確認できる場合、
  「判断材料が限られている」「育成前提」などの表現を使用してはいけません。

【評価記号の定義】
○：直接的かつ職務関連性の高い経験が確認できる  
△：間接的・汎用的・限定的な関連経験、または育成前提で評価可能  
×：関連する経験や根拠が一切確認できない  

【出力JSON形式（厳守）】
{{
  "score": 0,
  "criteria": {{
    "must_have_requirements": "○|△|×",
    "preferred_requirements": "○|△|×",
    "role_alignment": "○|△|×"
  }}
}}

【スコア算出ルール】
- score は 0 から 100 の整数で返してください。
- criteria の評価結果を総合して score を算出してください。
- ENTRY 求人においては、
  「△」が多い場合でも score を極端に低く設定してはいけません。
- 明確な不一致（×）が複数確認される場合のみ、低スコアを検討してください。

【職務内容】
{job["job_context"][:1500]}
"""




    try:
        model = genai.GenerativeModel(model_name)

        content_parts = [prompt, *candidate_files]
        
        response = model.generate_content(
            content_parts,
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
        
        criteria = parsed.get("criteria", {})

        parsed["score"] = calculate_score(
            criteria=criteria,
            seniority=job["seniority"]
        )


        # 🔍 Heuristic warning: likely ingestion / readability issue
        # Case 1: Document likely unreadable
        if st.session_state.get("total_cv_bytes", 0) < 2000:

            st.warning(
                "⚠️ The uploaded document may contain little readable text "
                "(e.g. scanned or image-based PDF)."
            )
        
        # Case 2: Valid evaluation, but no match
        elif (
            parsed.get("score", 0) == 0 and
            all(v == "×" for v in parsed.get("criteria", {}).values())
        ):
            st.info(
                "ℹ️ No matching signals were found for this role. "
                "This likely reflects a genuine CV–job mismatch."
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
    "Upload CV files (PDF / DOCX)",
    type=["pdf", "docx"],
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
    total_bytes = 0

    st.subheader("📎 Document ingestion log")

    for f in st.session_state.cvs:
        total_bytes += f.size
        part = to_gemini_part(f)
        candidate_files.append(part)
    
        st.code(
            {
                "filename": f.name,
                "detected_mime_type": mimetypes.guess_type(f.name)[0],
                "file_size_kb": round(f.size / 1024, 2),
            },
            language="json"
        )


    st.session_state.total_cv_bytes = total_bytes

    cv_files = [f.name for f in st.session_state.cvs]
    candidate_seniority = detect_candidate_seniority_from_cv(candidate_files)
    st.session_state.candidate_seniority = candidate_seniority

    st.session_state.candidate_files = candidate_files
    


    
    status.info("Evaluating candidate profile (combined documents)")
    
    cv_results = []
    
    for job_idx, job in enumerate(jobs, start=1):
        status.info(f"Evaluating Job {job_idx}/{len(jobs)}")
    
        result = ai_match_job(
            candidate_files,
            job,
            SELECTED_MODEL,
            candidate_seniority
        )


    
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
                "※ アップロードされた履歴書は、AIが内容を直接理解し評価しています。"
                " 追加の事前処理は不要な設計のため、迅速に分析結果を提供できます。"
            )


        
            for job_idx, r in enumerate(cv_block["results"]):
                job = r["job"]
        
                st.markdown(f"### {job['title']}")
                display_score = get_display_score(
                    r["score"],
                    job["seniority"]
                )
                
                st.write(f"**Estimated Offer Probability:** {display_score}%")

        
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
                        st.session_state.explanations[explain_key] = generate_full_assessment(
                            st.session_state.candidate_files,
                            job,
                            SELECTED_MODEL,
                            st.session_state.candidate_seniority
                        )



        
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
        
        
        
                    
