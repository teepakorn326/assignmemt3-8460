"""Shared helpers for all notebooks. Run 01_data_loading_and_sampling.ipynb first."""
import gzip
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PATH = DATA_DIR / "arxiv_cs_sample.jsonl.gz"
SPLIT_PATH = DATA_DIR / "split_ids.json"


def load_sample():
    """The cached 12k-paper CS sample, with the same light cleaning as notebook 01."""
    with gzip.open(SAMPLE_PATH, "rt", encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    for col in ("title", "abstract"):
        df[col] = df[col].str.split().str.join(" ")
    df["primary_category"] = df["categories"].str.split().str[0]
    df["all_categories"] = df["categories"].str.split()
    df["author_list"] = df["authors_parsed"].map(
        lambda names: [" ".join(p for p in (n[1], n[0]) if p) for n in names])
    df["text"] = df["title"] + ". " + df["abstract"]
    return df.drop(columns=["authors_parsed"]).drop_duplicates("id").reset_index(drop=True)


def load_split(df=None):
    """(train_df, test_df) using the fixed stratified split saved by notebook 01."""
    df = load_sample() if df is None else df
    split = json.loads(SPLIT_PATH.read_text())
    train = df[df["id"].isin(set(split["train"]))].reset_index(drop=True)
    test = df[df["id"].isin(set(split["test"]))].reset_index(drop=True)
    return train, test


ARTIFACTS_DIR = DATA_DIR.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_results(notebook, results):
    """Write a notebook's headline metrics to artifacts/<notebook>_results.json.

    Every notebook calls this once at the end so 06 and the report can read all
    results without re-running anyone else's notebook. Keys are free-form, e.g.
    {"model": "LogReg TF-IDF", "accuracy": 0.81, "macro_f1": 0.80}.
    """
    path = ARTIFACTS_DIR / f"{notebook}_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    return path


def load_results(notebook):
    path = ARTIFACTS_DIR / f"{notebook}_results.json"
    return json.loads(path.read_text()) if path.exists() else {}


_tfidf_cache = {}


def _tfidf_retrieve(query, k=5):
    """Fallback retriever: TF-IDF cosine over title+abstract. Built once per process."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    if "df" not in _tfidf_cache:
        df = load_sample()
        vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.8, sublinear_tf=True)
        _tfidf_cache.update(df=df, vec=vec, X=vec.fit_transform(df["text"]))
    c = _tfidf_cache
    scores = linear_kernel(c["vec"].transform([query]), c["X"]).ravel()
    top = scores.argsort()[::-1][:k]
    out = c["df"].iloc[top][["id", "title", "abstract", "primary_category", "author_list"]].copy()
    out["score"] = scores[top]
    return out.reset_index(drop=True)


def retrieve(query, k=5):
    """Return the k papers most similar to `query` as a DataFrame with a `score` column.

    Contract (keep stable, D's notebooks 05/06 depend on it):
      columns: id, title, abstract, primary_category, author_list, score; sorted by score desc.

    Member C implements the real retriever in `retrieval.py` exposing the same
    `retrieve(query, k)`; it is used automatically when that file exists. Until
    then a TF-IDF baseline is used, so nobody is blocked.
    """
    try:
        from retrieval import retrieve as _impl  # owned by member C
    except ImportError:
        _impl = _tfidf_retrieve
    return _impl(query, k)
