# etf_rag_system/recommendation/engine.py
import numpy as np
from numpy.linalg import norm
from etf_rag_system.data.dataset import etf_data

risk_map = {"낮음": 1, "중간": 2, "높음": 3}
item_names = [e["name"] for e in etf_data]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

def etf_to_vector(etf):
    return np.array([risk_map[etf["risk_level"]]/3, (etf["return_1y"]+30)/80,
                     etf["dividend_yield"]/6, etf["expense_ratio"]/1, etf["volatility"]/40])

item_vectors = np.array([etf_to_vector(e) for e in etf_data])
n = len(item_vectors)
item_sim = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        item_sim[i, j] = cosine_sim(item_vectors[i], item_vectors[j])

def cbf_similar_items(target_name, top_k=5):
    idx = item_names.index(target_name)
    scores = [(item_names[j], item_sim[idx, j]) for j in range(n) if j != idx]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def cbf_diverse(target_name, top_k=5):
    idx = item_names.index(target_name)
    scored = [(j, item_sim[idx, j]) for j in range(n) if j != idx]
    scored.sort(key=lambda x: x[1], reverse=True)
    result, seen_cats = [], set()
    for j, score in scored:
        cat = etf_data[j]["category"]
        if cat in seen_cats: continue
        result.append((item_names[j], round(score, 3), cat))
        seen_cats.add(cat)
        if len(result) >= top_k: break
    return result

# 협업 필터링 (Mock Data)
np.random.seed(42)
user_types = ["보수적_배당", "중립_인덱스", "공격적_성장", "중립_균형", "배당매니아", "글로벌투자자"]
R = np.zeros((len(user_types), len(etf_data)))
# (시드 평점 행렬 생략 - 필요시 Weekend 3 코드 참고, 인터페이스 유지를 위해 함수만 정의)

def cf_recommend(user_idx, top_k=5):
    return [("(협업필터링 미적용 사용자)", 0)]

def cold_start_recommend(user_idx, favorite_etf=None, top_k=5):
    if favorite_etf is None:
        return [("(선호 ETF를 알려주세요)", 0, "CBF-불가")]
    return [(n, s, "CBF-폴백") for n, s in cbf_similar_items(favorite_etf, top_k)]