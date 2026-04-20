# ==============================================================================
# [router.py 핵심 기능 요약]
# 1. 사용자의 질문 패턴을 분석해 검색 의도를 "조건", "키워드", "의미"로 분류
# 2. 문맥을 찾는 벡터 검색, 단어를 찾는 BM25, 둘을 합친 하이브리드, 조건을 거르느 필터링 검색 
# 3. LLM으로 검색 결과 관련성 순으로 rerank, 쓸모없는 정보 걸러내는 score_filter
# ==============================================================================

# 정규식 모듈. 사용자 질문에서 특정 패턴 찾는데 사용
import re 

# json 모듈. LLM에서 추출한 필터 정보를 JSON으로 파싱하는데 사용
import json 

# 배열 계산 및 통계 처리(정렬, 평균, 표준편차 등) 사용
import numpy as np

# Langchain에서 OpenAI의 챗봇 모델을 사용하기 위한 클래스
from langchain_openai import ChatOpenAI

# LLM에게 역할을 부여하거나 사용자의 질문을 전달하기 위한 메시지 클래스
from langchain_core.messages import SystemMessage, HumanMessage

# 벡터 기반 DB(의미 기반 검색)와 BM25(키워드 기반 검색) 객체 불러오는 커스텀 함수
from etf_rag_system.retrieval.vectorstore import get_vectorstore, get_bm25

# 한국어 형태소 분석기인 Kiwi를 활용해 질문을 단어 단위로 쪼개는 함수. BM25 검색에 사용
from etf_rag_system.retrieval.tokenizer import kiwi_tokenize

# LLM 인스턴스 생성. GPT-4o-mini 모델 사용, 온도는 0으로 설정해 일관된 답변 유도
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 전역 변수로 벡터스토어와 BM25 객체, 그리고 문서 리스트를 불러옴. 검색 함수에서 재사용
vectorstore = get_vectorstore()
bm25, documents = get_bm25()

# ETF 브랜드 리스트. 사용자의 질문에서 특정 브랜드가 언급되면 키워드 검색으로 분류하는데 활용
BRANDS = ["KODEX", "TIGER", "ACE", "ARIRANG", "HANARO", "SOL", "KBSTAR"]

"""
[키워드 검색]
사용자 질문에 포함된 단어가 문서에 얼마나 자주 등장하는지(TF-IDF 기반)를 계산하며 찾아냄
'KODEX 레버리지'처럼 고유명사나 특정 키워드 언급이 있는 경우 효과적
"""
# k는 반환할 상위 결과 수
def bm25_search(query, k=5): 
    # 질문을 형태소 분석기로 토큰화. 
    tokens = kiwi_tokenize(query)

    # 쪼개진 단어들을 BM25 모델에 넣어 각 문서와의 관련성 점수를 배열 형태로 받음
    scores = bm25.get_scores(tokens)

    # 점수가 가장 높은 순서대로 내림차순 정렬하여 상위 k개의 인덱스를 가져옴
    top_idx = np.argsort(scores)[::-1][:k]

    # 점수가 0보다 큰 문서들만 반환. 
    return [(documents[i], scores[i]) for i in top_idx if scores[i] > 0]

"""
[하이브리드 검색 (벡터 검색 + BM25)]
문맥적 '의미'를 찾는 벡터 검색과, 정확한 '단어'를 찾는 BM25의 장점을 합친 메서드
둘의 점수 산정 기준(Scale)이 완전히 다르기 때문에(하나는 코사인 유사도, 하나는 빈도수), 
이를 0~1 사이로 정규화(Min-Max)하여 합침
"""
def hybrid_search(query, alpha=0.5, k=5):
    # 1. 벡터 검색을 먼저 수행 20개 가져옴
    vec_results = vectorstore.similarity_search_with_score(query, k=20)
    # Langchain의 vectorstore는 점수가 낮을수록(거리) 좋은 경우가 많으므로 역수(1/(1+score))를 취해 점수가 클수록 좋은 것으로 맞춰줍니다.
    # 문서를 식별하기 위해 문서의 메타데이터 중 "name"(ETF 이름)을 키값으로 사용합니다.
    vec_scores = {doc.metadata["name"]: 1/(1+score) for doc, score in vec_results}

    # 2. BM25 키워드 검색을 수행합니다. 
    tokens = kiwi_tokenize(query)
    bm25_raw = bm25.get_scores(tokens)
    # 전체 문서에 대해 이름-점수 쌍의 딕셔너리를 만듭니다.
    bm25_scores = {documents[i].metadata["name"]: bm25_raw[i] for i in range(len(documents))}

    # 3. 내부 함수: 0~1 사이로 점수를 압축(Min-Max 정규화)하는 헬퍼 함수입니다.
    def minmax(d):
        if not d: return {}
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        # 분모가 0이 되는 것을 방지하기 위해 1e-8(매우 작은 수)을 더해줍니다.
        return {k: (v-mn)/(mx-mn+1e-8) for k, v in d.items()}

    # 벡터 점수와 BM25 점수를 각각 정규화합니다.
    vn, bn = minmax(vec_scores), minmax(bm25_scores)
    
    # 4. alpha 값(비율)에 따라 두 점수를 합산합니다. alpha가 0.5면 5:5 비율입니다.
    # set(vn)|set(bn)은 두 검색 결과에 등장한 모든 ETF 이름의 합집합을 구하는 합집합 연산입니다.
    combined = {n: alpha*vn.get(n,0) + (1-alpha)*bn.get(n,0) for n in set(vn)|set(bn)}
    
    # 5. 합산된 점수를 기준으로 내림차순 정렬하고 상위 k개를 자릅니다.
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # 6. 최종 점수에 맞춰 실제 문서 객체(doc)를 찾아 매핑하여 반환합니다.
    final_docs = []
    for name, score in sorted_results:
        for doc in documents:
            if doc.metadata["name"] == name:
                final_docs.append((doc, score))
                break
    return final_docs


def filtered_search(query, filters=None, k=5, fetch_k=20):
    """
    [메타데이터 필터링 검색기]
    "수수료가 0.5% 이하인 ETF 찾아줘" 같은 조건부 질문이 들어왔을 때, 
    일반 검색을 먼저 하고 결과물에서 메타데이터(조건)를 검사해 걸러내는 역할을 합니다.
    """
    # 필터링 과정에서 탈락하는 문서가 생길 수 있으므로, 최종 k개보다 넉넉하게 fetch_k(20)개를 먼저 가져옵니다.
    results = vectorstore.similarity_search_with_score(query, k=fetch_k)
    
    # 필터 조건이 없으면 굳이 아래 로직을 탈 필요 없이 바로 k개만 반환합니다.
    if not filters: return results[:k]
    
    filtered = []
    for doc, score in results:
        m = doc.metadata  # 문서에 저장된 부가 정보들 (카테고리, 수수료율, 배당률 등)
        ok = True
        
        # filters 딕셔너리에 담긴 조건들을 하나씩 검사합니다.
        for key, val in filters.items():
            if isinstance(val, dict):
                # 값이 'less_than'이나 'greater_than' 같은 범위 조건(Dict 형태)일 때의 로직입니다.
                if "less_than" in val and m.get(key, float("inf")) >= val["less_than"]: ok = False
                if "greater_than" in val and m.get(key, 0) <= val["greater_than"]: ok = False
            else:
                # 범위가 아니라 정확히 일치해야 하는 조건(예: 카테고리="IT")일 때의 로직입니다.
                if m.get(key) != val: ok = False
                
        # 모든 조건을 통과(ok == True)한 문서만 추가합니다.
        if ok: filtered.append((doc, score))
        
    # 필터링을 통과한 애들 중 상위 k개만 반환합니다.
    return filtered[:k]


def detect_intent(query):
    """
    [의도 파악 라우터 (Rule-based)]
    비싼 LLM을 호출하기 전에, 정규표현식이나 키워드 매칭 등 '가벼운 규칙'을 이용해 사용자의 질문 의도를 빠르게 분류합니다.
    이렇게 하면 비용을 절감하고 응답 속도를 극대화할 수 있습니다.
    """
    # 질문에 숫자+%, 혹은 대소 비교 단어(이상, 이하 등)가 들어가면 필터링이 필요한 "조건" 질문으로 봅니다.
    if re.search(r"\d+\s*%|이상|이하|초과|미만", query): return "조건"
    
    # 질문에 특정 운용사 브랜드명이 들어가면 콕 찝어 검색하는 "키워드" 질문으로 봅니다.
    for b in BRANDS:
        if b in query.upper(): return "키워드"
        
    # 위 조건에 모두 해당하지 않으면 포괄적인 "의미" 검색으로 분류합니다.
    return "의미"


def extract_filters(query):
    """
    [필터 조건 추출기 (LLM-based)]
    사용자의 자연어 질문에서 '필터링 기준'을 뽑아내 기계가 이해할 수 있는 JSON 형태로 변환합니다.
    """
    # LLM에게 줘야 할 지시사항(Prompt)입니다. 반드시 JSON으로 달라고 신신당부합니다.
    prompt = f"사용자 쿼리에서 ETF 검색 필터를 추출하세요.\n쿼리: {query}\n필터: category, risk_level, expense_ratio({{'less_than':숫자}}), dividend_yield({{'greater_than':숫자}})\n순수 JSON만. 해당 없으면 {{}}"
    
    # LLM을 호출하여 답변을 받습니다.
    resp = llm.invoke([{"role": "user", "content": prompt}]).content
    
    try: 
        # LLM이 말 잘 듣고 순수 JSON 텍스트만 줬다면 바로 딕셔너리로 변환하여 반환합니다.
        return json.loads(resp)
    except:
        # LLM이 말 안 듣고 마크다운(```json ... ```) 같은 걸 붙여서 에러가 났을 때를 대비한 안전망(Fallback)입니다.
        # 정규표현식으로 중괄호 { } 내부의 데이터만 강제로 뽑아내어 파싱합니다.
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        return json.loads(m.group()) if m else {}


def smart_router(query, k=5):
    """
    [스마트 라우터 메인 컨트롤러]
    파악된 질문의 의도(Intent)에 따라 가장 적절한 검색 함수로 작업을 분배(Routing)하는 총괄 함수입니다.
    이런 구조를 'Dynamic Routing'이라고 부르며, RAG의 정확도를 높이는 핵심 기법입니다.
    """
    intent = detect_intent(query)
    
    if intent == "조건":
        # 조건부 질문이면 LLM을 통해 조건을 JSON으로 뽑아낸 뒤 필터링 검색에 태웁니다.
        filters = extract_filters(query)
        return filtered_search(query, filters=filters, k=k)
    elif intent == "키워드":
        # 고유명사가 중요한 질문이면 BM25 키워드 검색만 태웁니다. (엉뚱한 의미 검색 결과 배제)
        return bm25_search(query, k=k)
    else:
        # 일반적인 질문이면 가장 성능이 무난하게 좋은 하이브리드(의미+키워드) 검색을 태웁니다.
        return hybrid_search(query, alpha=0.5, k=k)


def llm_rerank(query, search_results, top_k=3):
    """
    [LLM 기반 재정렬 (Reranking)]
    검색기들이 가져온 1차 결과물의 순서가 사용자의 의도와 맞지 않을 수 있습니다.
    이때 가장 똑똑한 LLM에게 "사용자 질문이랑 이 문서들이 얼마나 잘 맞는지 1~10점 매겨서 다시 줄세워봐"라고 시키는 고급 기법입니다.
    """
    # 검색된 문서들의 메타데이터와 본문 앞부분만 잘라서 평가용 텍스트 리스트로 만듭니다. (토큰 절약 목적)
    doc_list = "\n".join([f"[{i}] {doc.metadata['name']}: {doc.page_content[:150]}" for i, (doc, score) in enumerate(search_results)])
    
    # response_format을 json_object로 강제하여, LLM이 무조건 지정한 JSON 스키마로 답변을 주도록 유도합니다.
    response = llm.bind(response_format={"type": "json_object"}).invoke([
        SystemMessage(content="ETF 검색 결과를 질의 관련성 순으로 정렬하는 전문가입니다."),
        HumanMessage(content=f"질문: {query}\n검색 결과:\n{doc_list}\n각 문서에 1-10점 관련성 부여. JSON 형식: {{\"rankings\": [{{\"index\": 0, \"score\": 9, \"reason\": \"이유\"}}]}}")
    ])
    
    try:
        # LLM이 매긴 점수를 바탕으로 JSON을 파싱합니다.
        result = json.loads(response.content)
        rankings = result.get("rankings", [])
        
        # LLM이 부여한 점수(score)를 기준으로 다시 내림차순 정렬합니다.
        rankings.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 정렬된 인덱스대로 원래의 문서 객체를 찾아 상위 top_k개만 새롭게 구성하여 반환합니다.
        return [(search_results[r["index"]][0], r["score"]) for r in rankings[:top_k] if 0 <= r["index"] < len(search_results)]
    except:
        # LLM이 뻗거나 JSON 파싱에 실패하면, 아쉬운 대로 원래 1차 검색기의 결과 상위 top_k를 그냥 반환합니다 (안전망).
        return search_results[:top_k]


def score_filter(results, method="dynamic"):
    """
    [스코어 기반 노이즈 제거 필터]
    검색을 5개 해왔는데, 사실 1~2개만 관련 있고 3~5위는 전혀 상관없는 내용일 수 있습니다.
    상관없는 내용이 프롬프트에 들어가면 답변이 엉뚱해지므로(환각 발생), 점수가 낮은 '꼬리'를 잘라내는 기능입니다.
    """
    if not results: return results
    
    scores = [r[1] for r in results]
    
    # 문턱값(Threshold)을 설정하는 다양한 방법들입니다.
    if method == "fixed": 
        # 1. 고정값: 무조건 5점 이상만 통과 (하드코딩이라 유연성이 떨어짐)
        threshold = 5.0
    elif method == "dynamic": 
        # 2. 동적 평균: 검색된 점수들의 (평균 - 표준편차)를 커트라인으로 잡습니다. 
        # 전반적인 수준에 맞춰 유연하게 가장 낮은 티어의 문서를 버립니다.
        threshold = np.mean(scores) - np.std(scores)
    elif method == "gap":
        # 3. 갭(격차) 방식: 1위와 2위, 2위와 3위 사이의 점수 차이를 계산하고,
        # 점수가 가장 '뚝' 떨어지는 구간을 찾아 그 밑으로는 전부 버리는 스마트한 방법입니다.
        gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        threshold = scores[np.argmax(gaps) + 1] + 0.01 if gaps else 0
    else: 
        threshold = 0
        
    # 문턱값을 넘는 문서만 모읍니다.
    filtered = [r for r in results if r[1] >= threshold]
    
    # 필터링했더니 다 잘려나가서 남은 게 없다면, 최소한 1등 문서 하나라도 반환하게 만듭니다.
    return filtered if filtered else results[:1]
