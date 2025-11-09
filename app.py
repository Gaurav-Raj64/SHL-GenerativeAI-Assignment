#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Service — SHL Recommender (TF-IDF, boosted + role-aware)
----------------------------------------------------------------
Endpoints:
- GET  /health        -> {"status":"ok"}
- POST /recommend     -> { "query": "...", "topk": 10 } => [{name,url,score}, ...]

Startup:
- Loads shl_catalog.csv (columns: name, url)
- Builds TF-IDF(1–3 grams, sublinear tf) on catalog texts (name + url slug)
- Applies:
    * "solution" filter (excludes pre-packaged job solutions)
    * skill-term boosts
    * role-aware penalty for generic soft skills on engineer queries

Run:
    pip install fastapi uvicorn pandas scikit-learn numpy
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_PATH = os.getenv("CATALOG_PATH", "shl_catalog.csv")
TOP_K_DEFAULT = 10

# ------------------------ Data & Model ------------------------
def load_catalog(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog not found at: {path}")
    df = pd.read_csv(path)
    if not set(["name","url"]).issubset(df.columns):
        raise ValueError("Catalog must have columns: name, url")
    return df.dropna(subset=["name","url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)

def slug_from_url(u: str) -> str:
    try:
        parts = [p for p in str(u).strip("/").split("/") if p]
        return parts[-1].replace("-", " ")
    except Exception:
        return ""

def build_catalog_text(df: pd.DataFrame):
    texts = []
    for _, r in df.iterrows():
        name = str(r["name"]).strip()
        slug = slug_from_url(r["url"])
        texts.append(f"{name} | {slug}" if slug and slug.lower() not in name.lower() else name)
    return texts

def prepare_vectorizer(corpus_texts):
    vect = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,3),
        sublinear_tf=True,
        min_df=1,
        max_df=0.98,
        norm="l2",
    )
    X = vect.fit_transform(corpus_texts)
    return vect, X

BOOST_TERMS = {
    "java": 0.20, "python": 0.20, "sql": 0.18, "database": 0.10,
    "excel": 0.10, "analytics": 0.10, "ml": 0.12, "machine learning": 0.12,
    "cloud": 0.10, "aws": 0.12, "azure": 0.12, "gcp": 0.12,
    "communication": 0.18, "collaboration": 0.18, "teamwork": 0.16,
    "business": 0.10, "stakeholder": 0.10, "leadership": 0.10,
    "spring": 0.12, "hibernate": 0.12, "javascript": 0.12, "react": 0.10, "api": 0.10,
}
ENGINEERING_CUES = ["developer", "engineer", "programmer", "software", "backend", "frontend"]
TECH_TERMS = ["java","python","sql","spring","hibernate","api","web services",
              "javascript","react","cloud","aws","azure","gcp","database","databases"]

def recommend(query: str, vect: TfidfVectorizer, X_catalog, catalog_df: pd.DataFrame, top_k: int = TOP_K_DEFAULT):
    q = vect.transform([query])
    sims = cosine_similarity(q, X_catalog)[0]
    q_lower = query.lower()

    def text_for_match(i):
        name = str(catalog_df.iloc[i]["name"])
        url  = str(catalog_df.iloc[i]["url"])
        slug = url.rstrip("/").split("/")[-1].replace("-", " ")
        return f"{name} {slug}".lower(), name, url

    order = np.argsort(-sims)[:300]
    candidates = []
    for i in order:
        text, name_i, url_i = text_for_match(i)
        if "solution" in name_i.lower() or "solution" in url_i.lower():
            continue
        score = sims[i]
        for term, w in BOOST_TERMS.items():
            if term in q_lower and term in text:
                score += w
        if any(cue in q_lower for cue in ENGINEERING_CUES):
            if ("communication" in text or "collaboration" in text or "teamwork" in text or "business" in text):
                if not any(t in text for t in TECH_TERMS):
                    score -= 0.06
        candidates.append((score, i))

    candidates.sort(key=lambda x: -x[0])
    top = candidates[:top_k]
    out = []
    for score, i in top:
        out.append({
            "name": str(catalog_df.iloc[i]["name"]),
            "url":  str(catalog_df.iloc[i]["url"]),
            "score": float(score)
        })
    return out

# ------------------------ FastAPI App ------------------------
class RecRequest(BaseModel):
    query: str
    topk: Optional[int] = TOP_K_DEFAULT

app = FastAPI(title="SHL Recommender API (TF-IDF boosted)")

@app.on_event("startup")
def _startup():
    global CATALOG_DF, VECTORIZER, X_CATALOG
    CATALOG_DF = load_catalog(CATALOG_PATH)
    texts = build_catalog_text(CATALOG_DF)
    VECTORIZER, X_CATALOG = prepare_vectorizer(texts)

@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": int(len(CATALOG_DF))}

@app.post("/recommend")
def post_recommend(req: RecRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must be a non-empty string.")
    topk = int(req.topk or TOP_K_DEFAULT)
    topk = max(1, min(10, topk))  # enforce PDF limit
    results = recommend(req.query.strip(), VECTORIZER, X_CATALOG, CATALOG_DF, top_k=topk)
    return {"query": req.query, "topk": topk, "results": results}
