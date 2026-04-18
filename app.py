# app.py
import os
from dotenv import load_dotenv

# 1. 먼저 환경 변수를 로드해야 함
load_dotenv() 

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 2. 환경 변수가 로드된 '후'에 내부 모듈들을 불러와야 에러가 나지 않습니다.
from etf_rag_system.data.dataset import etf_data, eval_queries
from etf_rag_system.retrieval.router import smart_router, hybrid_search, bm25_search, llm_rerank, score_filter
from etf_rag_system.recommendation.risk import extract_profile, rule_based_filter, llm_recommend, analyze_risk, explain_risks, asdict
from etf_rag_system.evaluation.metrics import compute_bleu, rouge_n, rouge_l, simple_bertscore
from etf_rag_system.evaluation.llm_judge import llm_judge, criteria_evaluation

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ================================
# Pydantic 스키마 추가: 숨은 문맥 딥다이브 추출용
# ================================
class AdvancedContext(BaseModel):
    current_portfolio: str = Field(description="현재 보유 중인 자산이나 투자 전략 (예: S&P500, 개별주 위주 등). 모르면 '해당 없음'")
    specific_request: str = Field(description="사용자가 특별히 요청한 조건 (예: 1개만 딱 추천, 배당 ETF, 채권형 등). 모르면 '해당 없음'")
    risk_tolerance: str = Field(description="입력된 문맥상 파악된 진짜 위험 선호도: '안정형', '중립형', '공격형' 중 반드시 하나 선택")
    summary: str = Field(description="사용자 상황을 종합한 자연스러운 요약 (2~3문장)")

# ================================
# 1. ETF 추천 리포트 파이프라인 (Weekend 3) - 로직 개선
# ================================
def recommend_pipeline(user_text):
    # 1. 기존 모듈 호환성을 위해 기본 프로필 추출 (내부 데이터 구조 유지)
    profile = extract_profile(user_text)
    
    # 2. LLM + Pydantic으로 디테일한 문맥(요청사항, 현재 포트폴리오) 추가 추출
    context_prompt = PromptTemplate.from_template(
        """당신은 전문적인 자산 관리사입니다. 
사용자의 입력 텍스트를 분석하여 투자자 상태를 정확하게 추출하세요.
단순 기계적 계산보다는 사용자가 '왜' 이런 요청을 했는지 의도를 파악하는 것이 핵심입니다.
[사용자 입력]
{user_text}"""
    )
    structured_llm = llm.with_structured_output(AdvancedContext)
    advanced_info = (context_prompt | structured_llm).invoke({"user_text": user_text})
    
    # 3. 기존 프로필에 '똑똑해진 성향' 덮어쓰기 (기존 필터링 로직이 잘 돌게 하기 위함)
    profile.risk_tolerance = advanced_info.risk_tolerance
    
    # 4. 기존 파이프라인 정상 수행
    candidates = rule_based_filter(profile.risk_tolerance, etf_data)
    recs = llm_recommend(profile, candidates, top_k=3)
    warnings = analyze_risk(profile, recs, etf_data)
    risk_msg = explain_risks(profile, warnings)
    
    # 5. 리포트 생성 (추출해 둔 advanced_info를 프롬프트에 주입하여 똑똑하게 대답하도록 강제)
    rec_text = "\n".join([f"| {r['name']} | {r.get('allocation', '?')}% | {r.get('reason', '')} |" for r in recs])
    
    prompt = f"""ETF 추천 최종 리포트를 마크다운으로 작성해주세요.

[투자자 분석 데이터]
- 핵심 상황 요약: {advanced_info.summary}
- 현재 포트폴리오: {advanced_info.current_portfolio}
- 특별 요청사항: {advanced_info.specific_request}

[시스템 추천 초안 (최대 3개)]
{rec_text}
리스크: {risk_msg}

[작성 지시사항]
- 사용자의 특별 요청사항(예: "1개만 추천해달라")이 있다면, 시스템 추천 초안 중에서 가장 조건에 맞는 것만 남기고 나머지는 버리세요.
- 구조: 
  1) 🧑‍💼 투자자 요약 (위 '핵심 상황 요약'과 '현재 포트폴리오'를 엮어 사용자 맞춤형으로 자연스럽게 작성)
  2) 📊 맞춤형 ETF 추천 (위 테이블 양식을 지키되, 요청사항에 맞춰 개수를 조절해서 표기)
  3) ⚠️ 리스크 고지 (시스템의 리스크 내용을 기반으로 작성)
"""
    
    report = llm.invoke([{"role": "system", "content": "전문 금융 어드바이저."}, {"role": "user", "content": prompt}]).content
    return report

# ================================
# 2. 품질 평가 대시보드 (Weekend 2)
# ================================
def evaluate_query_extended(query, reference=""):
    initial = hybrid_search(query, k=7)
    reranked = llm_rerank(query, initial, top_k=3)
    filtered = score_filter(reranked, method="dynamic")

    context = "\n".join([f"[{doc.metadata['name']}] {doc.page_content}" for doc, score in filtered])
    answer = llm.invoke([
        SystemMessage(content="ETF 전문가입니다. 검색된 문서만을 근거로 답변하세요."),
        HumanMessage(content=f"참고 문서:\n{context}\n\n질문: {query}")
    ]).content

    search_info = "📋 검색 결과:\n" + "\n".join([f"  [{doc.metadata['name']}] score={score:.4f}" for doc, score in filtered])
    metrics_info = "📊 참조 답변을 입력하면 자동 평가가 표시됩니다."
    if reference.strip():
        b4 = compute_bleu(reference, answer, max_n=4)
        r1 = rouge_n(reference, answer, 1)["f1"]
        rl = rouge_l(reference, answer)["f1"]
        bs = simple_bertscore(reference, answer)
        gate = "🟢 PASS" if bs >= 0.7 else "🔴 FAIL"
        metrics_info = f"📊 자동 평가:\n  BLEU-4: {b4:.4f}\n  ROUGE-1: {r1:.4f}\n  ROUGE-L: {rl:.4f}\n  BERTScore: {bs:.4f}\n\n품질 게이트: {gate}"

    judge = llm_judge(query, answer, context)
    judge_total = judge.get("총점", 0)
    judge_info = f"🧑‍⚖️ LLM-as-Judge:\n  총점: {judge_total}/25\n  피드백: {judge.get('피드백', 'N/A')[:200]}\n  게이트: {'🟢 PASS' if judge_total >= 15 else '🔴 FAIL'}"

    crit = criteria_evaluation(query, answer, context)
    crit_info = "📋 Criteria 평가:\n" + "\n".join([f"  {n}: {r['score']}/5 — {r['reason'][:40]}" for n, r in crit["criteria"].items()])
    crit_info += f"\n\n  가중 점수: {crit['weighted_score']:.2f}/5.00 ({crit['normalized']:.0f}점)"

    return answer, search_info, metrics_info, judge_info, crit_info

# ================================
# 3. 검색 엔진 비교 (Weekend 1)
# ================================
def full_comparison(query, top_k):
    top_k = int(top_k)
    output = f"🔍 질의: {query}\n{'='*60}\n\n"
    
    # 1. FAISS
    from etf_rag_system.retrieval.vectorstore import get_vectorstore
    faiss_res = get_vectorstore().similarity_search_with_score(query, k=top_k)
    output += "📌 FAISS 벡터 검색:\n" + "\n".join([f"  {i}. [{s:.4f}] {d.metadata['name']}" for i, (d, s) in enumerate(faiss_res, 1)])
    
    # 2. BM25
    bm25_res = bm25_search(query, top_k)
    output += "\n\n📌 BM25 키워드 검색 (Kiwi):\n" + "\n".join([f"  {i}. [{s:.4f}] {d.metadata['name']}" for i, (d, s) in enumerate(bm25_res, 1)])
    
    # 3. Hybrid
    hybrid_res = hybrid_search(query, alpha=0.5, k=top_k)
    output += "\n\n📌 하이브리드 검색 (α=0.5):\n" + "\n".join([f"  {i}. [{s:.4f}] {d.metadata['name']}" for i, (d, s) in enumerate(hybrid_res, 1)])
    
    return output

# ================================
# Gradio UI 구성
# ================================
with gr.Blocks(title="스마트 RAG 기반 ETF 추천 시스템") as demo:
    gr.Markdown("# 📈 ETF 추천 및 평가 통합 시스템")
    
    with gr.Tab("1. 맞춤형 ETF 추천 (엔드투엔드)"):
        user_input = gr.Textbox(label="투자 성향 입력", placeholder="예: 은퇴 준비 중인 55세입니다. 월 200만원 여유가 있고 안정적인 배당이 중요해요.")
        rec_btn = gr.Button("리포트 생성")
        rec_out = gr.Markdown()
        rec_btn.click(recommend_pipeline, inputs=user_input, outputs=rec_out)

    with gr.Tab("2. LLM-as-Judge 품질 평가"):
        with gr.Row():
            q_input = gr.Textbox(label="질문", placeholder="ETF 관련 질문을 입력하세요")
            ref_input = gr.Textbox(label="참조 정답 (선택)", placeholder="자동 평가(BLEU 등)를 위한 정답 텍스트")
        eval_btn = gr.Button("평가 수행")
        with gr.Row():
            ans_out = gr.Textbox(label="📝 생성된 답변")
            search_out = gr.Textbox(label="🔍 검색 결과")
        with gr.Row():
            metric_out = gr.Textbox(label="📊 자동 메트릭")
            judge_out = gr.Textbox(label="🧑‍⚖️ LLM 평가")
            crit_out = gr.Textbox(label="📋 Criteria 다차원 평가")
        eval_btn.click(evaluate_query_extended, inputs=[q_input, ref_input], outputs=[ans_out, search_out, metric_out, judge_out, crit_out])

    with gr.Tab("3. 검색 엔진 비교 (FAISS vs BM25 vs Hybrid)"):
        with gr.Row():
            comp_q = gr.Textbox(label="검색 질의", placeholder="예: 배당 수익률 높은 안전한 ETF")
            comp_k = gr.Slider(1, 10, value=5, step=1, label="결과 수 (K)")
        comp_btn = gr.Button("비교 검색")
        comp_out = gr.Textbox(label="엔진별 검색 결과 비교", lines=20)
        comp_btn.click(full_comparison, inputs=[comp_q, comp_k], outputs=comp_out)

if __name__ == "__main__":
    demo.launch()