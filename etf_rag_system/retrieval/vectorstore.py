# etf_rag_system/retrieval/vectorstore.py

# ==============================================================================
# [vectorstore.py 핵심 기능 요약]
# 1. 벡터 DB 구축 (FAISS): 문맥과 '의미'를 이해하는 검색을 위해 텍스트를 숫자 벡터로 변환하여 저장합니다.
# 2. 키워드 인덱스 구축 (BM25): '단어'가 정확히 일치하는 문서를 찾기 위해 전통적인 검색 엔진의 색인을 만듭니다.
# 3. 글로벌 로드(Global Load): 서버가 뜰 때 DB를 한 번만 메모리에 올려, 사용자가 질문할 때마다 지연(Lag)이 발생하지 않도록 합니다.
# 4. 다양성 검색 (MMR): 비슷한 내용만 중복해서 검색되는 것을 막고, 다양한 종류의 ETF를 골고루 가져오는 기능을 제공합니다.
# ==============================================================================

import os
import numpy as np

# [FAISS] 페이스북(Meta)에서 만든 엄청나게 빠른 고성능 유사도 검색(벡터 DB) 라이브러리입니다.
# 메모리 기반으로 동작하기 때문에 지금처럼 수백~수천 개 규모의 문서를 다루는 RAG 프로토타입에서 최고의 속도를 냅니다.
from langchain_community.vectorstores import FAISS as LangchainFAISS

# [OpenAI Embeddings] 사람이 쓴 자연어 텍스트를 기계가 이해하는 숫자 배열(벡터)로 바꿔주는 번역기 역할을 합니다.
from langchain_openai import OpenAIEmbeddings

# [BM25] 엘라스틱서치(Elasticsearch) 같은 대형 검색 엔진의 뼈대가 되는 키워드 검색 알고리즘입니다.
from rank_bm25 import BM25Okapi

# 앞서 우리가 정성들여 텍스트 해상도를 높여 만든 "ETF Document 객체 덩어리들"을 가져옵니다.
from etf_rag_system.data.dataset import get_documents

# Kiwi 형태소 분석기를 이용해 문장을 단어로 예쁘게 쪼개는 함수를 가져옵니다.
from etf_rag_system.retrieval.tokenizer import kiwi_tokenize

# 1. 검색 대상이 될 문서 200여 개를 통째로 불러옵니다.
documents = get_documents()

# 2. 임베딩 모델 설정
# "text-embedding-3-small" 모델은 OpenAI의 최신 임베딩 모델로, 비용은 매우 저렴하면서도 이전 세대(ada-002)보다 한국어 성능이 훨씬 좋습니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ==========================================
# [Vectorstore 초기화 (의미 기반 검색용)]
# - 왜 파일 맨 바닥(Global)에서 초기화하나요?: 
# 이 작업은 텍스트를 벡터로 쪼개고 색인하는 과정을 거치므로 시간이 약간 걸립니다.
# 사용자가 질문을 던질 때마다 이 짓을 하면 챗봇이 너무 느려지므로, 파이썬 파일이 처음 실행될 때 "딱 한 번만" 만들어두고 돌려쓰기 위함입니다.
# ==========================================
vectorstore = LangchainFAISS.from_documents(documents, embeddings)


# ==========================================
# [BM25 초기화 (키워드 기반 검색용)]
# ==========================================
# 1. 말뭉치(Corpus) 만들기: 모든 문서의 본문(page_content)을 형태소 단위로 갈기갈기 찢어서 리스트로 만듭니다.
# 예: ["kodex", "200", "분류", "국내주식", ...]
corpus = [kiwi_tokenize(doc.page_content) for doc in documents]

# 2. 찢어둔 단어 뭉치를 BM25 알고리즘에 먹여서 검색 색인을 완성합니다.
# 이제 "2차전지"라는 질문이 들어오면, 이 단어가 어느 문서에 가장 알차게 들어있는지 0.001초 만에 계산할 수 있습니다.
bm25 = BM25Okapi(corpus)


def get_vectorstore():
    """
    미리 만들어둔 벡터 DB 객체를 반환합니다.
    router.py의 hybrid_search나 filtered_search에서 이 함수를 호출해 DB를 꺼내 씁니다.
    """
    return vectorstore


def get_bm25():
    """
    미리 만들어둔 BM25 검색기와 원본 문서 리스트를 반환합니다.
    단순 점수 배열만 반환하는 BM25의 특성상, "이 점수가 어떤 문서의 점수인지" 매핑하기 위해 documents도 같이 넘겨주는 것이 핵심입니다.
    """
    return bm25, documents


def mmr_search(query, k=5, fetch_k=10):
    """
    [MMR (Maximal Marginal Relevance) 검색]
    사용자가 "배당금 많이 주는 ETF 찾아줘"라고 했을 때, 
    일반 벡터 검색을 하면 1등부터 5등까지 전부 '비슷비슷한 미국 고배당 ETF'만 나올 수 있습니다. (정보의 획일화 현상)
    
    이럴 때 MMR을 쓰면:
    1. 일단 넉넉하게 10개(fetch_k)를 찾은 다음,
    2. 질문과 관련성이 높으면서도, "서로의 내용이 최대한 겹치지 않는(다양한)" 문서 5개(k)를 골라냅니다.
    -> 결과적으로 미국 배당, 국내 배당, 채권 배당 등 다양한 선택지를 사용자에게 제안할 수 있게 만드는 아주 훌륭한 기법입니다.
    """
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)