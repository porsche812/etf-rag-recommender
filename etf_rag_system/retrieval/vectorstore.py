# etf_rag_system/retrieval/vectorstore.py
import os
import numpy as np
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from etf_rag_system.data.dataset import get_documents
from etf_rag_system.retrieval.tokenizer import kiwi_tokenize

documents = get_documents()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vectorstore 초기화
vectorstore = LangchainFAISS.from_documents(documents, embeddings)

# BM25 초기화
corpus = [kiwi_tokenize(doc.page_content) for doc in documents]
bm25 = BM25Okapi(corpus)

def get_vectorstore():
    return vectorstore

def get_bm25():
    return bm25, documents

def mmr_search(query, k=5, fetch_k=10):
    """MMR 검색 (Weekend 1)"""
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)