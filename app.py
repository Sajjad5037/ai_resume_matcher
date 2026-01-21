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
あなたは、採用・書類選考の実務経験が豊富なプロフェッショナルな人材アドバイザーです。

以下の候補者が、この職種において内定を獲得できる可能性について、
**提供された履歴書（CV）の記載内容のみ**を根拠として、客観的かつ丁寧に評価してください。

【言語要件（必須）】
- 評価内容はすべて日本語で記述してください。
- 英語は一切使用しないでください。
- 採用担当者が社内で共有する評価コメントを想定した、自然で客観的なビジネス日本語を使用してください。
- JSONのキー名は変更せず、値（score以外の文章）はすべて日本語で記述してください。

【重要な評価ルール（必須）】
- すべての評価は、CVに明示的に記載されている事実・経験のみを根拠としてください。
- CVに記載されているスキル・業界経験・資格について、「不足している」「経験がない」と断定してはいけません。
- 交渉、説得、関係者調整、顧客対応などの**間接的・汎用的・営業関連（セールスアジャセント）な経験**が確認できる場合は、必ず明示的に言及してください。
- 間接的または汎用的な経験が存在する場合、その評価は必ず「△」とし、「×」を付けてはいけません。
- 「×」を付けられるのは、直接的・間接的を問わず、関連する経験や根拠がCV上に**一切確認できない場合のみ**です。
- 経験が短期間、非公式、職務名に含まれない場合でも、「間接的」「非典型的な経験」として評価し、「経験なし」と表現してはいけません。
- 関連する根拠が見つからない場合は、必ず「提供されたCV上では明確な根拠を確認できません」と記述してください。
- 間接的な経験を見落として「経験がない」と評価することは重大な評価ミスと見なされます。

以下の3つの観点で評価してください。

【評価記号の定義】
○：職務要件を十分に満たす、直接的かつ職務関連性の高い経験が確認できる  
△：直接的ではないが、汎用的・間接的・限定的な関連経験が確認できる  
×：関連する経験や根拠が一切確認できない  

【評価項目】
1. 必須要件（required_experience）
2. 歓迎要件（desired_experience, target_candidate）
3. 職務内容との適合性（job_content）

【各評価項目について必ず行うこと】
- CV内の具体的な記載内容（職務、行動、成果、エピソード）を明示または要約する
- その経験が評価につながる理由を説明する（なぜ○/△/×なのか）
- 採用判断において、その経験がどのような意味・影響を持つかを説明する
- 間接的・汎用的な経験である場合は、その旨を明確に記述する
- 採用担当者が候補者に対して、どのように前向きかつ建設的に伝えられるかを1文で記載する

※ 各評価理由は、内容の重複を避けつつ、原則として2〜3文程度で簡潔にまとめてください。

【総合評価】
- 各評価項目を踏まえ、候補者の強み・制約・育成余地・採用上のリスクを統合的に整理してください。
- 単なる要約ではなく、「なぜその評価になるのか」が分かる説明を含めてください。
- 当該職種における内定獲得確率を0〜100％で推定してください（現実的な採用基準に基づく）。

【出力形式（厳守）】
- 出力は有効なJSONのみとする
- JSON以外の文章を一切含めない
- 以下の構造例の値をコピーしないこと

必要なJSON構造：

{{
  "score": 0,
  "summary_reason": "",
  "criteria": {{
    "must_have_requirements": {{ "rating": "○|△|×", "reason": "" }},
    "preferred_requirements": {{ "rating": "○|△|×", "reason": "" }},
    "role_alignment": {{ "rating": "○|△|×", "reason": "" }}
  }}
}}

【候補者の履歴書（CV）】
\"\"\"
{cv_text[:6000]}
\"\"\"

【求人情報】
職種名：{job["title"]}
企業名：{job["company_name"]}
求人URL：{job["job_id"]}
書類通過率：{job["passrate_for_doc_screening"]}
内定率：{job["documents_to_job_offer_ratio"]}
紹介手数料：{job["fee"]}

【職務内容】
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
