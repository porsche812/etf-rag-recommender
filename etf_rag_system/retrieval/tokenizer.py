# etf_rag_system/retrieval/tokenizer.py

# ==============================================================================
# [tokenizer.py 핵심 기능 요약]
# 1. 형태소 분석 (Kiwi): "삼성전자가" -> "삼성전자"처럼 의미 없는 조사(가, 는, 은)를 떼어내고 알맹이(명사, 숫자, 영어)만 추출합니다.
# 2. 금융 도메인 맞춤 단어장: "S&P500", "2차전지" 같은 고유명사가 쪼개져서 검색 망가지는 것을 막습니다.
# 3. 동의어 확장 (Query Expansion): 일반인이 쓰는 단어(예: "안전")를 금융 전문 용어("저위험", "보수적")로 뻥튀기하여 검색 성공률(Recall)을 높입니다.
# ==============================================================================

# 정규표현식. 단어에 붙은 숫자나 기호(%, 억)를 규칙으로 쪼갤 때 사용합니다.
import re
# 빠르고 정확한 한국어 형태소 분석기입니다. (Okt나 Mecab 대신 최근 파이썬 환경에서 설치가 쉽고 성능이 좋아 많이 쓰입니다)
from kiwipiepy import Kiwi

# 형태소 분석기 객체를 미리 생성해둡니다. (매번 함수 안에서 부르면 느리기 때문입니다)
kiwi = Kiwi()

# ==========================================
# [사용자 정의 사전]
# 왜 필요한가?: 형태소 분석기가 똑똑하긴 하지만 금융 전문 용어는 잘 모릅니다.
# "2차전지"를 "2", "차", "전지"로 쪼개거나 "S&P500"을 산산조각 내버리면 검색이 안 되기 때문에, "얘네는 절대 쪼개지 마!" 하고 강제하는 리스트입니다.
# ==========================================
FINANCE_TERMS = {"2차전지", "S&P500", "CSI300", "KOSPI200", "KODEX200", "나스닥100"}

# ==========================================
# [동의어 사전]
# 왜 필요한가?: RAG의 고질적인 문제인 '어휘 불일치(Vocabulary Mismatch)'를 해결하기 위함입니다.
# 사용자는 "이자수익 많이 주는 거"라고 검색하지만, 실제 문서에는 "고배당" 혹은 "인컴"이라고 적혀있을 수 있습니다.
# ==========================================
finance_synonyms = {
    'ETF': ['상장지수펀드', '인덱스펀드', '지수추종'],
    '배당': ['분배금', '배당금', '인컴', '이자수익'],
    '안정': ['안전', '보수적', '저위험', '원금보존'],
    '성장': ['그로스', '공격적', '고수익', '고성장'],
    '미국': ['해외', '글로벌', '나스닥', 'S&P'],
}


def kiwi_tokenize(text):
    """
    [Kiwi 기반 메인 토크나이저]
    문장을 받아서 검색에 쓸모 있는 '진짜 키워드'만 리스트로 뽑아줍니다.
    router.py의 BM25 검색기가 문서를 찾을 때 사용하는 핵심 함수입니다.
    """
    # t.form: 추출된 형태소(단어) 자체
    # t.tag: 그 단어의 품사
    # NNG(일반명사), NNP(고유명사), SL(알파벳/영어), SN(숫자)만 뽑아냅니다.
    # 동사나 형용사("~합니다", "~좋은")는 ETF 이름이나 본문 검색에 방해만 되므로 버립니다.
    # 대소문자 구분을 없애기 위해 .lower()로 전부 소문자로 통일합니다 (S&P -> s&p).
    return [t.form.lower() for t in kiwi.tokenize(text) if t.tag in ("NNG", "NNP", "SL", "SN")]


def korean_financial_tokenize(text):
    """
    [규칙 기반(Rule-based) 서브 토크나이저]
    Kiwi 같은 무거운 AI 형태소 분석기를 돌리기엔 부담스럽거나, 정밀한 커스텀 규칙이 필요할 때 썼던 방식입니다.
    띄어쓰기와 정규표현식을 이용해 "한국어 조사"를 수동으로 발라냅니다.
    """
    # 일단 띄어쓰기 기준으로 툭툭 자릅니다.
    words = text.split()
    tokens = []
    
    # 떼어내고 싶은 한국어 조사 리스트 (긴 것부터 짧은 것 순서로 배치해 오류 방지)
    particles = ['으로', '에서', '부터', '까지', '에게', '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '로']
    
    for word in words:
        # 1. 절대 쪼개면 안 되는 금융 고유명사인지 먼저 확인합니다. 맞으면 통과.
        if word in FINANCE_TERMS:
            tokens.append(word)
            continue
            
        # 2. 숫자+단위 처리 로직 (예: "10%이상", "100억부터")
        # 숫자로 시작해서 단위(%, 억, 만 등)가 오고 그 뒤에 뭔가 붙어있다면 쪼갭니다.
        m = re.match(r'^([\d,.]+)(%|억|만|원|개|배)(.*)$', word)
        if m:
            tokens.extend([m.group(1), m.group(2)]) # 숫자와 단위를 쪼개서 넣음
            if m.group(3): tokens.append(m.group(3)) # 뒤에 붙은 떨거지(조사 등)도 넣음
            continue
            
        # 3. 영어+한글 조합 처리 로직 (예: "KODEX는", "S&P500이")
        # ETF 이름은 보통 영어인데 뒤에 조사가 붙어버리면 검색이 안 됩니다. 이를 분리합니다.
        m = re.match(r'^([A-Za-z0-9&]+)([가-힣]+)$', word)
        if m:
            tokens.append(m.group(1)) # 앞의 영어부분 저장
            rest = m.group(2)         # 뒤의 한글부분
            separated = False
            
            # 한글 부분에서 조사 리스트에 해당하는 게 있으면 떼어버립니다.
            for p in sorted(particles, key=len, reverse=True):
                if rest == p or (rest.endswith(p) and len(rest) > len(p)):
                    if rest.endswith(p) and len(rest) > len(p): tokens.append(rest[:-len(p)])
                    tokens.append(p)
                    separated = True
                    break
            if not separated: tokens.append(rest)
            continue
            
        # 4. 순수 한글 단어에 조사가 붙은 경우 처리 (예: "주식이")
        separated = False
        for p in sorted(particles, key=len, reverse=True):
            if word.endswith(p) and len(word) > len(p):
                tokens.append(word[:-len(p)]) # 조사 앞부분(명사) 저장
                tokens.append(p)              # 조사 저장
                separated = True
                break
                
        # 아무 규칙에도 안 걸렸으면 그냥 원본 단어를 넣습니다.
        if not separated: tokens.append(word)
        
    return tokens


def synonym_expand(query):
    """
    [동의어 기반 쿼리 확장기 (Query Expansion)]
    검색의 '그물'을 넓게 던지는 기능입니다.
    사용자가 "미국 배당 ETF 찾아줘"라고 질문했다면,
    -> ["미국 배당 ETF 찾아줘", "해외 분배금 상장지수펀드 찾아줘", "글로벌 인컴 인덱스펀드 찾아줘" ...]
    이런 식으로 비슷한 의미의 문장들을 여러 개 복제해서 만들어냅니다.
    이렇게 만들어진 여러 문장으로 검색을 돌리면, 사용자가 어떤 단어를 쓰든 문서를 찾아낼 확률이 압도적으로 높아집니다.
    """
    expanded = [query] # 원본 질문을 가장 먼저 넣습니다.
    
    # 미리 정의해둔 동의어 사전을 싹 뒤집니다.
    for keyword, synonyms in finance_synonyms.items():
        # 사용자의 질문 안에 사전의 키워드(예: '배당')가 포함되어 있다면
        if keyword in query:
            # 해당 키워드를 동의어(예: '분배금', '인컴')로 바꿔치기한 새로운 문장들을 만들어 리스트에 추가합니다.
            for syn in synonyms:
                expanded.append(query.replace(keyword, syn))
                
    return expanded