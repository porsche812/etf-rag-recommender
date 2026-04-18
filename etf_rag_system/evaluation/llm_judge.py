# etf_rag_system/evaluation/llm_judge.py
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
from etf_rag_system.evaluation.metrics import compute_bleu, rouge_n, rouge_l, simple_bertscore

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_judge(query, answer, context, criteria=None):
    if criteria is None:
        criteria = {"정확성": "답변이 사실과 일치하는가?", "관련성": "질문 의도와 부합하는가?", "완전성": "핵심정보 포함?", "안전성": "위험 고지?", "명확성": "이해하기 쉬운가?"}
    c_text = "\n".join([f"- {k}: {v} (1-5)" for k, v in criteria.items()])
    resp = llm.bind(response_format={"type": "json_object"}).invoke([
        SystemMessage(content="답변 품질 평가 전문가. JSON으로 응답."),
        HumanMessage(content=f"질문: {query}\n문서: {context[:500]}\n답변: {answer}\n평가기준:\n{c_text}\nJSON: {{\"scores\": {{\"정확성\": 4}}, \"총점\": 20, \"피드백\": \"...\"}}")
    ])
    try: return json.loads(resp.content)
    except: return {"error": "파싱 실패"}

def criteria_evaluation(query, answer, context):
    criteria = {
        "사실_정확성": {"desc": "답변이 사실과 일치?", "weight": 0.30},
        "투자_적합성": {"desc": "투자 목적에 적합?", "weight": 0.25},
        "리스크_고지": {"desc": "위험 고지 여부?", "weight": 0.20},
        "정보_완전성": {"desc": "수수료, 수익률 등 정보 포함?", "weight": 0.15},
        "표현_명확성": {"desc": "전문 용어 설명 여부?", "weight": 0.10},
    }
    results = {}
    for name, info in criteria.items():
        resp = llm.bind(response_format={"type": "json_object"}).invoke([
            SystemMessage(content="품질 평가 전문가. JSON 응답."),
            HumanMessage(content=f"기준: {info['desc']}\n질문: {query}\n답변: {answer}\n1-5점 평가. JSON: {{\"score\": 4, \"reason\": \"\"}}")
        ])
        try:
            r = json.loads(resp.content)
            results[name] = {"score": r["score"], "reason": r.get("reason", ""), "weight": info["weight"]}
        except: results[name] = {"score": 0, "reason": "실패", "weight": info["weight"]}
    weighted_total = sum(r["score"] * r["weight"] for r in results.values())
    return {"criteria": results, "weighted_score": weighted_total, "normalized": weighted_total / 5.0 * 100}

class ETFEvaluationPipeline:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {"ROUGE-1": 0.3, "BERTScore": 0.7, "LLM_Judge": 15}
        self.results = []
    def evaluate_single(self, query, answer, reference, context):
        b4 = compute_bleu(reference, answer, 4)
        r1 = rouge_n(reference, answer, 1)["f1"]
        rl = rouge_l(reference, answer)["f1"]
        bs = simple_bertscore(reference, answer)
        judge = llm_judge(query, answer, context)
        jt = judge.get("총점", 0)
        gates = {"ROUGE-1": r1 >= self.thresholds["ROUGE-1"], "BERTScore": bs >= self.thresholds["BERTScore"], "LLM_Judge": jt >= self.thresholds["LLM_Judge"]}
        status = "🔴 FAIL" if not gates["BERTScore"] else ("🟡 WARN" if not all(gates.values()) else "🟢 PASS")
        res = {"query": query[:30], "BLEU-4": b4, "ROUGE-1": r1, "BERTScore": bs, "LLM_Judge": jt, "status": status}
        self.results.append(res)
        return res