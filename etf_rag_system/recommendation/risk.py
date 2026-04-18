# etf_rag_system/recommendation/risk.py
import json
import re
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@dataclass
class InvestorProfile:
    name: str
    risk_tolerance: str
    investment_goal: str
    investment_horizon: int
    monthly_budget: int

def extract_profile(user_text):
    # LLM이 JSON 형식만 딱 내뱉도록 지시를 더 명확히 합니다.
    prompt = f"사용자 메시지에서 투자자 프로필을 JSON 형식으로만 추출하세요. 다른 설명은 하지 마세요.\n메시지: {user_text}\n반드시 지켜야 할 JSON 구조: {{\"name\": \"사용자\", \"risk_tolerance\": \"보수적|중립|공격적\", \"investment_goal\": \"안정수익|자산증식|배당수익\", \"investment_horizon\": 5, \"monthly_budget\": 100}}"
    
    # .bind(response_format={"type": "json_object"})를 사용해 JSON 출력을 강제합니다.
    resp = llm.bind(response_format={"type": "json_object"}).invoke([
        {"role": "system", "content": "너는 사용자의 문장에서 금융 프로필을 추출해 JSON으로 응답하는 전문가야."},
        {"role": "user", "content": prompt}
    ]).content
    
    try: 
        data = json.loads(resp)
    except Exception as e:
        # 만약 JSON 파싱에 실패하면 정규식으로 중괄호 내부만 긁어옵니다.
        print(f"JSON 파싱 실패, 재시도 중... 에러: {e}")
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                data = {}
        else:
            data = {}
            
    return InvestorProfile(
        name=data.get("name", "사용자"), 
        risk_tolerance=data.get("risk_tolerance", "중립"),
        investment_goal=data.get("investment_goal", "자산증식"),
        investment_horizon=int(data.get("investment_horizon", 5)),
        monthly_budget=int(data.get("monthly_budget", 100))
    )

def rule_based_filter(risk_tolerance, etf_data):
    allowed = {"보수적": ["낮음"], "중립": ["낮음", "중간"], "공격적": ["낮음", "중간", "높음"]}
    return [e for e in etf_data if e["risk_level"] in allowed.get(risk_tolerance, ["낮음", "중간", "높음"])]

def llm_recommend(profile, candidates, top_k=3):
    etf_list = "\n".join([f"- {e['name']}: {e['category']}, 수익률 {e['return_1y']}%, 위험 {e['risk_level']}" for e in candidates])
    prompt = f"투자자: {profile.name} / {profile.risk_tolerance} / {profile.investment_goal}\n후보: {etf_list}\n적합한 ETF {top_k}개를 JSON 배열로 반환. [{{\"name\":\"ETF명\",\"allocation\":비중(%),\"reason\":\"이유\"}}]"
    resp = llm.invoke([{"role": "system", "content": "ETF 투자 전문가."}, {"role": "user", "content": prompt}]).content
    try: return json.loads(resp)
    except:
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        return json.loads(match.group()) if match else []

def analyze_risk(profile, recommendations, etf_data):
    etf_dict = {e["name"]: e for e in etf_data}
    warnings, rec_etfs = [], []
    for r in recommendations:
        if r.get("name") in etf_dict: rec_etfs.append(etf_dict[r["name"]])
    if not rec_etfs: return warnings

    cat_counts = Counter(e["category"] for e in rec_etfs)
    top_cat, top_n = cat_counts.most_common(1)[0]
    if top_n / len(rec_etfs) * 100 > 60:
        warnings.append({"type": "집중투자", "severity": "warning", "message": f"카테고리 '{top_cat}'에 집중"})

    tol_order = {"보수적": 1, "중립": 2, "공격적": 3}
    risk_order = {"낮음": 1, "중간": 2, "높음": 3}
    p_risk = tol_order.get(profile.risk_tolerance, 2)
    for e in rec_etfs:
        if risk_order.get(e["risk_level"], 2) > p_risk:
            warnings.append({"type": "위험불일치", "severity": "danger", "message": f"{e['name']}({e['risk_level']}) > {profile.risk_tolerance}"})
    return warnings

def explain_risks(profile, warnings):
    if not warnings: return "특별한 리스크 경고가 없습니다."
    w_text = "\n".join([f"- [{w['type']}] {w['message']}" for w in warnings])
    prompt = f"투자자 {profile.name}({profile.risk_tolerance})에게 리스크를 쉽게 설명. 100자 이내.\n경고: {w_text}"
    return llm.invoke([{"role": "system", "content": "친절한 금융 어드바이저."}, {"role": "user", "content": prompt}]).content.strip()