#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — SHL Recommendation Engine (TF-IDF, boosted + role-aware)
-----------------------------------------------------------------
- No deep learning deps (uses scikit-learn only)
- Filters out "Solution" (pre-packaged job solutions)
- Boosts skill terms to improve Recall@10
- Adds a small penalty to generic soft-skill items when the query clearly targets engineers/developers
- Outputs eval + predictions in the PDF-required format

Usage:
  pip install pandas scikit-learn numpy
  python step3_recommender_tfidf_boosted_roleaware.py --build-and-eval
  python step3_recommender_tfidf_boosted_roleaware.py --predict-test
  python step3_recommender_tfidf_boosted_roleaware.py --recommend "Hiring Java developer who can collaborate with business teams" --topk 10
"""
import argparse, os, sys
from typing import List, Dict, Tuple
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_PATH = "shl_catalog.csv"
DATASET_PATH = "Gen_AI Dataset.xlsx"
TOP_K = 10

# --- Data loading ---
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not set(["name","url"]).issubset(df.columns):
        raise ValueError("Catalog must have columns: name, url")
    return df.dropna(subset=["name","url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)

def load_datasets(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    train_df = pd.read_excel(xls, sheet_name="Train-Set").dropna(subset=["Query","Assessment_url"]).reset_index(drop=True)
    test_df  = pd.read_excel(xls, sheet_name="Test-Set").dropna(subset=["Query"]).reset_index(drop=True)
    return train_df, test_df

# --- Catalog text building ---
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

# --- Vectorizer ---
def prepare_vectorizer(corpus_texts):
    vect = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,3),      # trigrams help short titles
        sublinear_tf=True,      # dampen raw tf
        min_df=1,
        max_df=0.98,
        norm="l2",
    )
    X = vect.fit_transform(corpus_texts)
    return vect, X

# --- Recommender (with filtering + boosting + role-aware penalty) ---
BOOST_TERMS = {
    # programming & data
    "java": 0.20, "python": 0.20, "sql": 0.18, "database": 0.10,
    "excel": 0.10, "analytics": 0.10, "ml": 0.12, "machine learning": 0.12,
    "cloud": 0.10, "aws": 0.12, "azure": 0.12, "gcp": 0.12,
    # communication / behavior
    "communication": 0.18, "collaboration": 0.18, "teamwork": 0.16,
    "business": 0.10, "stakeholder": 0.10, "leadership": 0.10,
    # web/backend extras
    "spring": 0.12, "hibernate": 0.12, "javascript": 0.12, "react": 0.10, "api": 0.10,
}

ENGINEERING_CUES = ["developer", "engineer", "programmer", "software", "backend", "frontend"]

TECH_TERMS = ["java","python","sql","spring","hibernate","api","web services","javascript","react","cloud","aws","azure","gcp","database","databases"]

def recommend(query, vect, X_catalog, catalog_df, top_k=TOP_K):
    # TF-IDF similarities
    q = vect.transform([query])
    sims = cosine_similarity(q, X_catalog)[0]

    q_lower = query.lower()

    def text_for_match(i):
        name = str(catalog_df.iloc[i]["name"])
        url  = str(catalog_df.iloc[i]["url"])
        slug = url.rstrip("/").split("/")[-1].replace("-", " ")
        return f"{name} {slug}".lower(), name, url

    # take wide candidate pool; then filter "solution", boost skills, and apply role-aware penalty
    order = np.argsort(-sims)[:300]
    candidates = []
    for i in order:
        text, name_i, url_i = text_for_match(i)

        # hard filter: ignore job "Solutions"
        if "solution" in name_i.lower() or "solution" in url_i.lower():
            continue

        score = sims[i]

        # skill-term boosts: if term appears in both query and item
        for term, w in BOOST_TERMS.items():
            if term in q_lower and term in text:
                score += w

        # role-aware penalty: if query is clearly engineering-focused,
        # and item is mostly soft skills without core tech cues, nudge down slightly
        if any(cue in q_lower for cue in ENGINEERING_CUES):
            if ("communication" in text or "collaboration" in text or "teamwork" in text or "business" in text):
                if not any(t in text for t in TECH_TERMS):
                    score -= 0.06  # small penalty

        candidates.append((score, i))

    # final sort by adjusted score
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

# --- Metrics / IO ---
def recall_at_k(truth_urls, pred_urls, k=TOP_K):
    if not truth_urls: return 0.0
    truth = set(u.strip().lower() for u in truth_urls)
    pred  = set(u.strip().lower() for u in pred_urls[:k])
    return len(truth & pred) / max(1, len(truth))

def evaluate_on_train(vect, X_catalog, catalog_df, train_df, top_k=TOP_K):
    truth_map = {}
    for q, url in train_df[["Query","Assessment_url"]].itertuples(index=False):
        truth_map.setdefault(str(q), []).append(str(url))
    scores = []
    for q, truth in truth_map.items():
        preds = recommend(q, vect, X_catalog, catalog_df, top_k=top_k)
        pred_urls = [p["url"] for p in preds]
        scores.append(recall_at_k(truth, pred_urls, k=top_k))
    return {"mean_recall_at_10": float(np.mean(scores) if scores else 0.0), "n_queries": len(scores)}

def write_eval_report(res, path="eval_train_recall_at_10_tfidf.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Mean Recall@10: {res.get('mean_recall_at_10',0.0):.4f}\n")
        f.write(f"Evaluated on N={res.get('n_queries',0)} train queries.\n")
    print(f"[done] Wrote eval report to {path}")

def predict_test_and_save(vect, X_catalog, catalog_df, test_df, out_csv="predictions_test_tfidf.csv", top_k=TOP_K):
    rows = []
    for q in test_df["Query"].tolist():
        preds = recommend(q, vect, X_catalog, catalog_df, top_k=top_k)
        for p in preds:
            rows.append({"Query": q, "Assessment_url": p["url"]})
    pd.DataFrame(rows, columns=["Query","Assessment_url"]).to_csv(out_csv, index=False)
    print(f"[done] Wrote test predictions to {out_csv}")

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="TF-IDF recommender for SHL assignment (boosted + role-aware)")
    ap.add_argument("--build-and-eval", action="store_true")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--recommend", type=str, default=None)
    ap.add_argument("--topk", type=int, default=TOP_K)
    ap.add_argument("--catalog", type=str, default=CATALOG_PATH)
    ap.add_argument("--data", type=str, default=DATASET_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.catalog): print(f"[error] Catalog not found: {args.catalog}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.data): print(f"[error] Dataset not found: {args.data}", file=sys.stderr); sys.exit(1)

    catalog_df = load_catalog(args.catalog)
    train_df, test_df = load_datasets(args.data)
    texts = build_catalog_text(catalog_df)
    vect, X_catalog = prepare_vectorizer(texts)

    if args.recommend:
        for r in recommend(args.recommend, vect, X_catalog, catalog_df, top_k=args.topk):
            print(f"- {r['name']} -> {r['url']} (score={r['score']:.3f})")

    if args.build_and_eval:
        res = evaluate_on_train(vect, X_catalog, catalog_df, train_df, top_k=args.topk)
        write_eval_report(res)

    if args.predict_test:
        predict_test_and_save(vect, X_catalog, catalog_df, test_df, top_k=args.topk)

if __name__ == "__main__":
    main()
