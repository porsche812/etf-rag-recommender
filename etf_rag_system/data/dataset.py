import pandas as pd
import random

# ==========================================
# 1. 기존 데이터 세분화 (Weekend 3)
# - 추가된 특징: region(국가지역), investment_style(투자스타일), distribution_freq(배당주기), currency_hedge(환헤지여부)
# ==========================================
etf_data = [
    {"name": "KODEX 200", "category": "국내주식", "region": "국내", "investment_style": "인덱스", "distribution_freq": "분기배당", "currency_hedge": "환노출",
     "expense_ratio": 0.15, "return_1y": 8.5, "risk_level": "중간", "dividend_yield": 1.8, "volatility": 15.2,
     "keywords": ["코스피", "대형주", "인덱스", "분산투자", "국내주식"],
     "description": "KOSPI 200 지수를 추종하는 대표 국내 ETF. 대형주 중심 분산투자에 적합합니다."},
    
    {"name": "KODEX 미국S&P500TR", "category": "해외주식", "region": "미국", "investment_style": "인덱스", "distribution_freq": "TR(재투자)", "currency_hedge": "환노출",
     "expense_ratio": 0.05, "return_1y": 25.3, "risk_level": "중간", "dividend_yield": 0.0, "volatility": 18.5,
     "keywords": ["S&P500", "미국", "대형주", "패시브", "해외주식", "복리"],
     "description": "미국 S&P500 지수의 총수익(TR)을 추종. 배당금을 자동 재투자하여 장기 복리 효과를 극대화합니다."},
     
    {"name": "ACE 미국배당다우존스", "category": "배당", "region": "미국", "investment_style": "배당성장", "distribution_freq": "월배당", "currency_hedge": "환노출",
     "expense_ratio": 0.01, "return_1y": 12.1, "risk_level": "낮음", "dividend_yield": 3.5, "volatility": 10.3,
     "keywords": ["미국배당", "다우존스", "고배당", "월배당", "안정", "SCHD"],
     "description": "미국 고배당 대형주 중심. 최저 수수료(0.01%)로 매월 안정적인 현금흐름(배당수익)을 창출합니다."},
     
    {"name": "TIGER 2차전지테마", "category": "테마", "region": "국내", "investment_style": "성장주", "distribution_freq": "연배당", "currency_hedge": "환노출",
     "expense_ratio": 0.45, "return_1y": -15.2, "risk_level": "높음", "dividend_yield": 0.0, "volatility": 35.7,
     "keywords": ["2차전지", "배터리", "성장주", "고위험", "테마"],
     "description": "2차전지·배터리 관련 기업에 집중 투자. 고성장 기대 대신 높은 변동성을 감수해야 합니다."},
     
    {"name": "TIGER 국고채10년", "category": "채권", "region": "국내", "investment_style": "장기채", "distribution_freq": "분기배당", "currency_hedge": "환노출",
     "expense_ratio": 0.07, "return_1y": 4.2, "risk_level": "낮음", "dividend_yield": 2.8, "volatility": 5.1,
     "keywords": ["국고채", "10년", "안전자산", "금리", "채권"],
     "description": "한국 10년 국고채 지수를 추종하는 안전자산 ETF. 금리 하락기에 자본 차익을 기대할 수 있습니다."},
     
    {"name": "KODEX 골드선물(H)", "category": "원자재", "region": "글로벌", "investment_style": "대체투자", "distribution_freq": "배당없음", "currency_hedge": "환헤지",
     "expense_ratio": 0.68, "return_1y": 18.7, "risk_level": "중간", "dividend_yield": 0.0, "volatility": 20.4,
     "keywords": ["금", "골드", "인플레이션", "안전자산", "원자재"],
     "description": "금 선물 가격을 추종하며 환헤지 적용. 인플레이션 방어 및 포트폴리오 분산용입니다."},
     
    {"name": "KODEX 미국나스닥100TR", "category": "해외주식", "region": "미국", "investment_style": "성장주", "distribution_freq": "TR(재투자)", "currency_hedge": "환노출",
     "expense_ratio": 0.05, "return_1y": 32.1, "risk_level": "높음", "dividend_yield": 0.0, "volatility": 25.3,
     "keywords": ["나스닥100", "미국", "기술주", "성장", "해외주식"],
     "description": "나스닥100 기술주 중심 총수익 ETF. 높은 성장성과 높은 변동성을 동반합니다."},
     
    {"name": "TIGER 미국반도체", "category": "섹터", "region": "미국", "investment_style": "모멘텀", "distribution_freq": "분기배당", "currency_hedge": "환노출",
     "expense_ratio": 0.49, "return_1y": 45.0, "risk_level": "높음", "dividend_yield": 0.0, "volatility": 38.2,
     "keywords": ["반도체", "미국", "필라델피아", "기술", "섹터"],
     "description": "미국 필라델피아 반도체 지수 추종. AI/반도체 수요 수혜 기대되나 변동성 극심합니다."},
     
    {"name": "KODEX 단기채권PLUS", "category": "채권", "region": "국내", "investment_style": "단기자금", "distribution_freq": "연배당", "currency_hedge": "환노출",
     "expense_ratio": 0.03, "return_1y": 3.8, "risk_level": "낮음", "dividend_yield": 3.2, "volatility": 1.5,
     "keywords": ["단기채", "파킹통장", "저위험", "현금성", "채권"],
     "description": "단기 채권 중심의 초저위험 ETF. 파킹통장 대용으로 현금성 자산을 안정적으로 운용합니다."},
     
    {"name": "TIGER 리츠부동산", "category": "리츠", "region": "국내", "investment_style": "인컴수익", "distribution_freq": "월배당", "currency_hedge": "환노출",
     "expense_ratio": 0.29, "return_1y": 6.5, "risk_level": "중간", "dividend_yield": 4.8, "volatility": 12.8,
     "keywords": ["리츠", "부동산", "월배당", "인프라", "실물자산"],
     "description": "국내 리츠·부동산 인프라에 투자. 높은 월배당(4.8%)과 실물자산 분산 효과가 있습니다."},
     
    {"name": "KODEX 200TR", "category": "국내주식", "region": "국내", "investment_style": "인덱스", "distribution_freq": "TR(재투자)", "currency_hedge": "환노출",
     "expense_ratio": 0.12, "return_1y": 9.1, "risk_level": "중간", "dividend_yield": 0.0, "volatility": 14.9,
     "keywords": ["코스피200", "TR", "총수익", "인덱스", "국내주식"],
     "description": "KODEX 200의 총수익(TR) 버전. 배당 재투자 효과를 포함하여 장기 성과가 우수합니다."},
     
    {"name": "TIGER 고배당저변동", "category": "배당", "region": "국내", "investment_style": "방어주", "distribution_freq": "분기배당", "currency_hedge": "환노출",
     "expense_ratio": 0.30, "return_1y": 7.2, "risk_level": "낮음", "dividend_yield": 5.2, "volatility": 8.7,
     "keywords": ["고배당", "저변동", "방어적", "배당", "안정"],
     "description": "고배당+저변동성 종목을 선별. 배당수익률 5.2%로 국내 방어적 포트폴리오에 적합합니다."}
]

# ==========================================
# 2. 고도화된 RAG 테스트용 더미 데이터 생성 로직 (200개)
# ==========================================
def generate_mock_etfs(count=200):
    brands = ["KODEX", "TIGER", "ACE", "KBSTAR", "ARIRANG", "SOL"]
    
    # 세분화된 테마 풀 (배당 안에서도 커버드콜, 배당성장, 고배당 등으로 분리)
    themes = [
        {"name": "미국배당+7%프리미엄다우존스", "cat": "배당", "reg": "미국", "style": "커버드콜", "freq": "월배당", "desc": "SCHD 지수에 콜옵션 매도 전략을 결합하여 주가 상승을 제한하는 대신 연 7% 이상의 높은 월배당을 지급합니다.", "keys": ["커버드콜", "다우존스", "고배당", "월배당", "인컴"]},
        {"name": "글로벌비만치료제", "cat": "테마", "reg": "글로벌", "style": "메가트렌드", "freq": "연배당", "desc": "노보노디스크, 일라이릴리 등 글로벌 비만 치료제 및 헬스케어 혁신 기업에 집중 투자합니다.", "keys": ["비만치료제", "바이오", "헬스케어", "제약", "글로벌"]},
        {"name": "인도타타그룹", "cat": "해외주식", "reg": "신흥국", "style": "대형주", "freq": "분기배당", "desc": "인도 경제 성장을 주도하는 1위 기업 집단인 타타그룹 핵심 계열사에 선별 투자합니다.", "keys": ["인도", "타타", "신흥국", "고성장", "해외주식"]},
        {"name": "미국30년국채프리미엄", "cat": "채권", "reg": "미국", "style": "커버드콜", "freq": "월배당", "desc": "미국 장기채권에 투자하며 커버드콜 전략을 통해 금리 변동성에 방어하며 높은 월 배당을 창출합니다.", "keys": ["미국채", "30년", "장기채", "커버드콜", "월배당"]},
        {"name": "CD금리액티브(합성)", "cat": "파킹형", "reg": "국내", "style": "단기자금", "freq": "TR(재투자)", "desc": "매일 CD(양도성예금증서) 91일물 금리만큼 수익이 복리로 쌓이는 파킹형 ETF입니다.", "keys": ["파킹통장", "금리", "CD금리", "초안전", "현금대용"]},
        {"name": "유럽명품TOP10", "cat": "테마", "reg": "유럽", "style": "가치주", "freq": "연배당", "desc": "LVMH, 에르메스 등 압도적인 가격 결정력을 가진 유럽 럭셔리 명품 기업 10개에 압축 투자합니다.", "keys": ["명품", "유럽", "럭셔리", "소비재", "가치주"]},
        {"name": "글로벌AI반도체핵심", "cat": "섹터", "reg": "글로벌", "style": "성장주", "freq": "분기배당", "desc": "엔비디아, TSMC 등 인공지능 연산에 필수적인 글로벌 반도체 밸류체인에 집중 투자합니다.", "keys": ["AI", "반도체", "엔비디아", "성장주", "글로벌"]},
        {"name": "코스닥150선물인버스", "cat": "파생형", "reg": "국내", "style": "헤지용", "freq": "배당없음", "desc": "코스닥 150 지수 하락 시 수익이 발생하는 인버스 상품. 시장 하락에 대한 단기 헤지용으로 적합합니다.", "keys": ["인버스", "하락장", "코스닥", "헤지", "파생"]},
        {"name": "미국달러SOFR금리", "cat": "파킹형", "reg": "미국", "style": "단기자금", "freq": "TR(재투자)", "desc": "미국 무위험 지표금리(SOFR)를 추종하며 달러 환차익과 매일 쌓이는 이자를 동시에 누립니다.", "keys": ["달러", "SOFR", "환테크", "파킹통장", "안전자산"]},
        {"name": "일본반도체소부장", "cat": "섹터", "reg": "일본", "style": "가치성장", "freq": "분기배당", "desc": "글로벌 반도체 공정에 필수적인 일본의 소재, 부품, 장비 독점 기업들에 투자합니다.", "keys": ["일본", "엔화", "소부장", "반도체", "장비"]}
    ]
    
    generated = []
    for i in range(count):
        brand = random.choice(brands)
        theme = random.choice(themes)
        
        # 특징별 리스크 및 데이터 차등화
        hedge_status = random.choice(["환노출", "환헤지(H)"]) if theme["reg"] != "국내" else "환노출"
        
        if theme["cat"] in ["파킹형", "채권"]:
            risk = "매우낮음" if theme["cat"] == "파킹형" else "낮음"
            vol = round(random.uniform(0.5, 6.0), 1)
            ret = round(random.uniform(3.0, 5.5), 1)
            div = round(random.uniform(0.0, 8.0), 1) if theme["freq"] != "TR(재투자)" else 0.0
            exp = round(random.uniform(0.01, 0.07), 2)
        elif theme["style"] == "커버드콜":
            risk = "중간"
            vol = round(random.uniform(8.0, 12.0), 1)
            ret = round(random.uniform(-5.0, 10.0), 1)
            div = round(random.uniform(7.0, 12.0), 1) # 커버드콜은 배당률이 극단적으로 높음
            exp = round(random.uniform(0.3, 0.5), 2)
        else:
            risk = "높음"
            vol = round(random.uniform(18.0, 45.0), 1)
            ret = round(random.uniform(-15.0, 60.0), 1)
            div = round(random.uniform(0.0, 2.0), 1)
            exp = round(random.uniform(0.2, 0.6), 2)
            
        suffix = ""
        if hedge_status == "환헤지(H)": suffix += "(H)"
        if theme["freq"] == "TR(재투자)": suffix += " TR"
        
        etf_name = f"{brand} {theme['name']}{suffix}"
        if any(e["name"] == etf_name for e in generated + etf_data):
            etf_name = f"{etf_name} {random.randint(1, 99)}"

        generated.append({
            "name": etf_name,
            "category": theme["cat"],
            "region": theme["reg"],
            "investment_style": theme["style"],
            "distribution_freq": theme["freq"],
            "currency_hedge": hedge_status,
            "expense_ratio": exp,
            "return_1y": ret,
            "risk_level": risk,
            "dividend_yield": div,
            "volatility": vol,
            "keywords": theme["keys"],
            "description": theme["desc"]
        })
        
    return generated

# 총 200개 생성하여 기존 리스트에 병합 (총 212개)
etf_data.extend(generate_mock_etfs(200))


# ==========================================
# 3. 확장된 평가 질의
# ==========================================
eval_queries = [
    {"query": "안정적인 배당 ETF를 추천해주세요", "reference": "ACE 미국배당다우존스 ETF를 추천합니다. 미국 고배당 대형주 중심이며 최저 수수료(0.01%)로 안정적 배당수익을 추구합니다.", "relevant": ["ACE 미국배당다우존스", "TIGER 고배당저변동"]},
    {"query": "월배당이면서 수익률이 7% 이상 되는 커버드콜 상품 있어?", "reference": "미국배당+7%프리미엄다우존스 커버드콜 상품을 추천합니다. 주가 상승은 제한되지만 연 7% 이상의 높은 월배당을 지급합니다.", "relevant": ["미국배당+7%프리미엄다우존스", "미국30년국채프리미엄"]},
    {"query": "잠시 현금을 보관할 파킹형 통장 대용 ETF 추천해줘", "reference": "CD금리액티브나 미국달러SOFR금리 ETF를 추천합니다. 매일 이자가 복리로 쌓이는 초저위험 상품입니다.", "relevant": ["CD금리액티브(합성)", "미국달러SOFR금리", "KODEX 단기채권PLUS"]},
    {"query": "배당금 받을 필요 없고 알아서 재투자해 주는 미국 주식 ETF?", "reference": "분배금을 자동으로 재투자하여 복리 효과를 극대화하는 KODEX 미국S&P500TR 또는 KODEX 미국나스닥100TR을 추천합니다.", "relevant": ["KODEX 미국S&P500TR", "KODEX 미국나스닥100TR"]},
    {"query": "요즘 비만약 관련주가 뜬다던데 관련 ETF 찾아줘", "reference": "글로벌 비만치료제 기업인 노보노디스크, 일라이릴리 등에 집중 투자하는 '글로벌비만치료제' ETF가 있습니다.", "relevant": ["글로벌비만치료제"]},
    {"query": "환율 변동 신경 쓰기 싫어. 금에 투자하는 환헤지 상품은?", "reference": "환헤지(H)가 적용되어 달러 변동성 없이 금 가격만을 추종하는 KODEX 골드선물(H)을 고려해보세요.", "relevant": ["KODEX 골드선물(H)"]}
]

# ==========================================
# 4. 고도화된 Langchain Document 변환 함수
# (LLM이 문맥을 정확히 파악하도록 Document 생성 텍스트의 해상도를 극대화)
# ==========================================
def get_documents():
    from langchain_core.documents import Document
    documents = []
    for etf in etf_data:
        # 이 텍스트 덩어리가 RAG 검색의 핵심이 됩니다. 특징을 명확한 포맷으로 매핑합니다.
        text = (
            f"[{etf['name']}] "
            f"분류: {etf['category']} | 지역: {etf['region']} | 투자전략: {etf['investment_style']} | "
            f"배당주기: {etf['distribution_freq']} | 환노출여부: {etf['currency_hedge']} \n"
            f"설명: {etf['description']} \n"
            f"주요 키워드: {', '.join(etf['keywords'])} \n"
            f"지표 -> 수수료: {etf['expense_ratio']}% | 배당수익률: {etf['dividend_yield']}% | "
            f"최근 1년 수익률: {etf['return_1y']:.1f}% | 연환산 변동성: {etf['volatility']:.1f}% | 위험도: {etf['risk_level']}"
        )
        documents.append(Document(page_content=text, metadata=etf))
    return documents

if __name__ == "__main__":
    print(f"총 ETF 데이터 개수: {len(etf_data)}개")
    print(f"총 평가 질의 개수: {len(eval_queries)}개")
    print("\n[생성된 고해상도 샘플 Document 텍스트]")
    print(get_documents()[-1].page_content)