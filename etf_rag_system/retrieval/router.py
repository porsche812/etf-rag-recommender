# etf_rag_system/retrieval/router.py
import re
import json
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from etf_rag_system.retrieval.vectorstore import get_vectorstore, get_bm25
from etf_rag_system.retrieval.tokenizer import kiwi_tokenize

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
vectorstore = get_vectorstore()
bm25, documents = get_bm25()

BRANDS = ["KODEX", "TIGER", "ACE", "ARIRANG", "HANARO", "SOL", "KBSTAR"]

def bm25_search(query, k=5):
    tokens = kiwi_tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(documents[i], scores[i]) for i in top_idx if scores[i] > 0]

def hybrid_search(query, alpha=0.5, k=5):
    """벡터 + BM25 하이브리드 (Min-Max 정규화)"""
    vec_results = vectorstore.similarity_search_with_score(query, k=20)
    vec_scores = {doc.metadata["name"]: 1/(1+score) for doc, score in vec_results}

    tokens = kiwi_tokenize(query)
    bm25_raw = bm25.get_scores(tokens)
    bm25_scores = {documents[i].metadata["name"]: bm25_raw[i] for i in range(len(documents))}

    def minmax(d):
        if not d: return {}
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        return {k: (v-mn)/(mx-mn+1e-8) for k, v in d.items()}

    vn, bn = minmax(vec_scores), minmax(bm25_scores)
    combined = {n: alpha*vn.get(n,0) + (1-alpha)*bn.get(n,0) for n in set(vn)|set(bn)}
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    final_docs = []
    for name, score in sorted_results:
        for doc in documents:
            if doc.metadata["name"] == name:
                final_docs.append((doc, score))
                break
    return final_docs

def filtered_search(query, filters=None, k=5, fetch_k=20):
    results = vectorstore.similarity_search_with_score(query, k=fetch_k)
    if not filters: return results[:k]
    filtered = []
    for doc, score in results:
        m = doc.metadata
        ok = True
        for key, val in filters.items():
            if isinstance(val, dict):
                if "less_than" in val and m.get(key, float("inf")) >= val["less_than"]: ok = False
                if "greater_than" in val and m.get(key, 0) <= val["greater_than"]: ok = False
            else:
                if m.get(key) != val: ok = False
        if ok: filtered.append((doc, score))
    return filtered[:k]

def detect_intent(query):
    if re.search(r"\d+\s*%|이상|이하|초과|미만", query): return "조건"
    for b in BRANDS:
        if b in query.upper(): return "키워드"
    return "의미"

def extract_filters(query):
    prompt = f"사용자 쿼리에서 ETF 검색 필터를 추출하세요.\n쿼리: {query}\n필터: category, risk_level, expense_ratio({{'less_than':숫자}}), dividend_yield({{'greater_than':숫자}})\n순수 JSON만. 해당 없으면 {{}}"
    resp = llm.invoke([{"role": "user", "content": prompt}]).content
    try: return json.loads(resp)
    except:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        return json.loads(m.group()) if m else {}

def smart_router(query, k=5):
    intent = detect_intent(query)
    if intent == "조건":
        filters = extract_filters(query)
        return filtered_search(query, filters=filters, k=k)
    elif intent == "키워드":
        return bm25_search(query, k=k)
    else:
        return hybrid_search(query, alpha=0.5, k=k)

def llm_rerank(query, search_results, top_k=3):
    """LLM으로 검색 결과 질의 관련성 재정렬 (Weekend 2)"""
    doc_list = "\n".join([f"[{i}] {doc.metadata['name']}: {doc.page_content[:150]}" for i, (doc, score) in enumerate(search_results)])
    response = llm.bind(response_format={"type": "json_object"}).invoke([
        SystemMessage(content="ETF 검색 결과를 질의 관련성 순으로 정렬하는 전문가입니다."),
        HumanMessage(content=f"질문: {query}\n검색 결과:\n{doc_list}\n각 문서에 1-10점 관련성 부여. JSON 형식: {{\"rankings\": [{{\"index\": 0, \"score\": 9, \"reason\": \"이유\"}}]}}")
    ])
    try:
        result = json.loads(response.content)
        rankings = result.get("rankings", [])
        rankings.sort(key=lambda x: x.get("score", 0), reverse=True)
        return [(search_results[r["index"]][0], r["score"]) for r in rankings[:top_k] if 0 <= r["index"] < len(search_results)]
    except:
        return search_results[:top_k]

def score_filter(results, method="dynamic"):
    """스코어 기반 필터링 (Weekend 2)"""
    if not results: return results
    scores = [r[1] for r in results]
    if method == "fixed": threshold = 5.0
    elif method == "dynamic": threshold = np.mean(scores) - np.std(scores)
    elif method == "gap":
        gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        threshold = scores[np.argmax(gaps) + 1] + 0.01 if gaps else 0
    else: threshold = 0
    filtered = [r for r in results if r[1] >= threshold]
    return filtered if filtered else results[:1]
