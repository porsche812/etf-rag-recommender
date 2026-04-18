# 📈 스마트 RAG 기반 맞춤형 ETF 추천 시스템 (ETF RAG Recommender)

자연어 질의를 통해 사용자의 투자 성향과 목표를 파악하고, 최적의 금융 상품(ETF)을 추천해주는 **검색 증강 생성(RAG) 기반 챗봇 대시보드**입니다. 단순한 벡터 검색을 넘어 하이브리드 검색(Hybrid Search), 쿼리 의도 라우팅(Query Intent Routing), 투자 리스크 분석 파이프라인을 적용하여 높은 정확도와 신뢰성을 제공합니다.

---

## 🚀 핵심 기능 (Key Features)

* **🔍 하이브리드 검색 (Hybrid Search)**
  * `FAISS`를 활용한 의미 기반 벡터 검색과 `BM25` + `Kiwi 형태소 분석기`를 결합한 키워드 검색을 앙상블하여 검색(Retrieval) 정확도를 극대화했습니다.
* **🔀 스마트 쿼리 라우터 (Smart Query Routing)**
  * 사용자의 질문 의도를 분석하여 조건 검색(수익률, 수수료 등), 키워드 검색(브랜드명), 의미 검색으로 자동 분기(Routing)합니다.
* **⚠️ 맞춤형 리스크 진단 (Risk Analysis)**
  * 사용자의 자연어 입력에서 투자자 프로필(위험 성향, 목표, 투자 기간 등)을 추출하고, 추천된 ETF 포트폴리오와의 적합성을 분석하여 위험(집중 투자, 위험도 불일치 등)을 사전에 경고합니다.
* **⚖️ LLM-as-Judge 및 품질 평가 파이프라인**
  * 자체 구축한 평가 모듈을 통해 RAG 파이프라인의 검색 및 생성 품질을 정량적(BLEU, ROUGE, BERTScore) 및 정성적(LLM-as-Judge, 기준 기반 평가)으로 검증했습니다.

---

## 🛠 기술 스택 (Tech Stack)

* **Language**: Python 3.11
* **AI / RAG**: LangChain, OpenAI API (gpt-4o-mini, text-embedding-3-small)
* **Search / Vector DB**: FAISS, rank_bm25
* **NLP**: Kiwipiepy (금융 도메인 특화 토큰화 처리)
* **UI**: Gradio (웹 대시보드 구현)
* **Data Processing**: Pandas, NumPy

---

## 📂 프로젝트 구조 (Project Architecture)

```text
etf-rag-recommender/
├── app.py                     # 메인 실행 파일 (Gradio 대시보드 및 파이프라인)
├── requirements.txt           # 패키지 의존성
├── .env                       # API Key 설정 파일
└── etf_rag_system/            # 핵심 RAG 및 추천 시스템 패키지
    ├── data/                  # ETF 데이터 스냅샷 및 로더 모듈
    ├── retrieval/             # 하이브리드 검색, 형태소 분석, 쿼리 라우터 모듈
    ├── recommendation/        # 투자자 프로필 추출 및 리스크 분석 모듈
    └── evaluation/            # LLM-as-Judge 및 정량 평가 스크립트