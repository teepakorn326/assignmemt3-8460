"""Paper retrieval for the RAG pipeline (owned by member C, built in notebook 04).

`utils.retrieve(query, k)` imports `retrieve` from here automatically, so notebooks 05 and 06
get the sentence-embedding retriever as soon as this file exists.

Index = all 12,000 papers in the cached sample. Embeddings (all-MiniLM-L6-v2, 384-d, L2-normalised)
are cached in artifacts/minilm_embeddings.npy with the matching paper ids in
artifacts/minilm_ids.json, so later notebooks never re-encode the corpus.

    from retrieval import retrieve, recommend, embed
    retrieve("graph neural networks for molecules", k=5)   # DataFrame: id, title, abstract,
                                                           #   primary_category, author_list, score
    recommend("2404.00934", k=5)                           # similar papers to a paper in the sample
"""
import json
import warnings

import numpy as np

from utils import ARTIFACTS_DIR, load_sample

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_PATH = ARTIFACTS_DIR / "minilm_embeddings.npy"
IDS_PATH = ARTIFACTS_DIR / "minilm_ids.json"
COLUMNS = ["id", "title", "abstract", "primary_category", "author_list"]

_state = {}


def _model():
    if "model" not in _state:
        from sentence_transformers import SentenceTransformer
        _state["model"] = SentenceTransformer(MODEL_NAME)
    return _state["model"]


def embed(texts, batch_size=64, show_progress_bar=False):
    """L2-normalised MiniLM embeddings for a list of strings -> (n, 384) float32 array."""
    return _model().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=show_progress_bar).astype(np.float32)


def build_index(df=None, force=False):
    """Encode every paper once and cache the matrix. Returns (df, embeddings)."""
    df = load_sample() if df is None else df
    if EMB_PATH.exists() and IDS_PATH.exists() and not force:
        emb = np.load(EMB_PATH)
        ids = json.loads(IDS_PATH.read_text())
        if ids == df["id"].tolist():
            return df, emb
        warnings.warn("Cached embeddings do not match the sample; rebuilding.")
    emb = embed(df["text"], show_progress_bar=True)
    np.save(EMB_PATH, emb)
    IDS_PATH.write_text(json.dumps(df["id"].tolist()))
    return df, emb


def _index():
    if "df" not in _state:
        df, emb = build_index()
        _state.update(df=df, emb=emb, id_to_row={pid: i for i, pid in enumerate(df["id"])})
    return _state["df"], _state["emb"], _state["id_to_row"]


def _top(scores, k, exclude_row=None):
    if exclude_row is not None:
        scores = scores.copy()
        scores[exclude_row] = -np.inf
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return top[np.argsort(-scores[top])]


def retrieve(query, k=5):
    """The k papers most similar to a free-text query (cosine over MiniLM embeddings)."""
    df, emb, _ = _index()
    scores = emb @ embed([query])[0]
    top = _top(scores, k)
    out = df.iloc[top][COLUMNS].copy()
    out["score"] = scores[top]
    return out.reset_index(drop=True)


def recommend(paper_id, k=5):
    """Papers most similar to a paper already in the sample (the paper itself is excluded)."""
    df, emb, id_to_row = _index()
    row = id_to_row[paper_id]
    scores = emb @ emb[row]
    top = _top(scores, k, exclude_row=row)
    out = df.iloc[top][COLUMNS].copy()
    out["score"] = scores[top]
    return out.reset_index(drop=True)
