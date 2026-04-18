# etf_rag_system/evaluation/metrics.py
import math
import numpy as np
from collections import Counter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def precision_at_k(retrieved_ids, relevant_ids, k=5):
    relevant_retrieved = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return relevant_retrieved / k

def recall_at_k(retrieved_ids, relevant_ids, k=5):
    relevant_retrieved = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return relevant_retrieved / len(relevant_ids) if relevant_ids else 0

def hit_rate(eval_data, search_fn, k=5):
    hits = 0
    for item in eval_data:
        results = search_fn(item["query"], k=k)
        if any(r[0].metadata["name"] in item["relevant"] for r in results):
            hits += 1
    return hits / len(eval_data)

def average_precision(ranked_names, relevant_set, k=5):
    ranked = ranked_names[:k]
    hits, sum_prec = 0, 0
    for i, name in enumerate(ranked, 1):
        if name in relevant_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(k, len(relevant_set)) if relevant_set else 0

def map_at_k(eval_data, search_fn, k=5):
    aps = []
    for item in eval_data:
        results = search_fn(item["query"], k=k)
        ranked = [r[0].metadata["name"] for r in results]
        aps.append(average_precision(ranked, set(item["relevant"]), k))
    return sum(aps) / len(aps)

# --- 텍스트 유사도 ---
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def modified_precision(ref_tokens, cand_tokens, n):
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    cand_ngrams = Counter(get_ngrams(cand_tokens, n))
    clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
    total = sum(cand_ngrams.values())
    return clipped / total if total > 0 else 0.0

def brevity_penalty(ref_len, cand_len):
    if cand_len >= ref_len: return 1.0
    return math.exp(1 - ref_len / cand_len)

def compute_bleu(reference, candidate, max_n=4):
    ref_tokens, cand_tokens = reference.split(), candidate.split()
    bp = brevity_penalty(len(ref_tokens), len(cand_tokens))
    log_avg = 0.0
    for n in range(1, max_n + 1):
        p = modified_precision(ref_tokens, cand_tokens, n)
        if p == 0: return 0.0
        log_avg += (1.0 / max_n) * math.log(p)
    return bp * math.exp(log_avg)

def rouge_n(reference, candidate, n=1):
    ref_tokens, cand_tokens = reference.split(), candidate.split()
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    cand_ngrams = Counter(get_ngrams(cand_tokens, n))
    overlap = sum(min(ref_ngrams[ng], cand_ngrams.get(ng, 0)) for ng in ref_ngrams)
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
    precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}

def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(reference, candidate):
    ref_tokens, cand_tokens = reference.split(), candidate.split()
    lcs = lcs_length(ref_tokens, cand_tokens)
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "lcs": lcs}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def simple_bertscore(reference, candidate):
    ref_emb = np.array(embeddings.embed_query(reference))
    cand_emb = np.array(embeddings.embed_query(candidate))
    return cosine_sim(ref_emb, cand_emb)