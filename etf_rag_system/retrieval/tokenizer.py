# etf_rag_system/retrieval/tokenizer.py
import re
from kiwipiepy import Kiwi

kiwi = Kiwi()

FINANCE_TERMS = {"2차전지", "S&P500", "CSI300", "KOSPI200", "KODEX200", "나스닥100"}

finance_synonyms = {
    'ETF': ['상장지수펀드', '인덱스펀드', '지수추종'],
    '배당': ['분배금', '배당금', '인컴', '이자수익'],
    '안정': ['안전', '보수적', '저위험', '원금보존'],
    '성장': ['그로스', '공격적', '고수익', '고성장'],
    '미국': ['해외', '글로벌', '나스닥', 'S&P'],
}

def kiwi_tokenize(text):
    """Kiwi 형태소 분석 (Weekend 3)"""
    return [t.form.lower() for t in kiwi.tokenize(text) if t.tag in ("NNG", "NNP", "SL", "SN")]

def korean_financial_tokenize(text):
    """금융 도메인 특화 한국어 토큰 분리 (Weekend 2)"""
    words = text.split()
    tokens = []
    particles = ['으로', '에서', '부터', '까지', '에게', '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '로']
    for word in words:
        if word in FINANCE_TERMS:
            tokens.append(word)
            continue
        m = re.match(r'^([\d,.]+)(%|억|만|원|개|배)(.*)$', word)
        if m:
            tokens.extend([m.group(1), m.group(2)])
            if m.group(3): tokens.append(m.group(3))
            continue
        m = re.match(r'^([A-Za-z0-9&]+)([가-힣]+)$', word)
        if m:
            tokens.append(m.group(1))
            rest = m.group(2)
            separated = False
            for p in sorted(particles, key=len, reverse=True):
                if rest == p or (rest.endswith(p) and len(rest) > len(p)):
                    if rest.endswith(p) and len(rest) > len(p): tokens.append(rest[:-len(p)])
                    tokens.append(p)
                    separated = True
                    break
            if not separated: tokens.append(rest)
            continue
            
        separated = False
        for p in sorted(particles, key=len, reverse=True):
            if word.endswith(p) and len(word) > len(p):
                tokens.append(word[:-len(p)])
                tokens.append(p)
                separated = True
                break
        if not separated: tokens.append(word)
    return tokens

def synonym_expand(query):
    """도메인 특화 동의어 사전 확장 (Weekend 1)"""
    expanded = [query]
    for keyword, synonyms in finance_synonyms.items():
        if keyword in query:
            for syn in synonyms:
                expanded.append(query.replace(keyword, syn))
    return expanded