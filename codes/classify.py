"""Member B: research-domain classifier and TF-IDF keywords.

Used by 03_classification_keywords.ipynb, and by 06_end_to_end_demo.ipynb:

    from classify import predict, predict_proba, top_keywords
    predict(["Title. Abstract ..."])          -> ['cs.CL']
    predict_proba(["Title. Abstract ..."])    -> DataFrame, one column per category
    top_keywords(["Title. Abstract ..."], k=5) -> [['keyword', ...]]

The model is the TF-IDF + Logistic Regression pipeline trained in NB03 and saved to
artifacts/classifier.pkl (Week 4 TF-IDF, Week 5 classification).
"""
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import ARTIFACTS_DIR

CLASSIFIER_PATH = ARTIFACTS_DIR / "classifier.pkl"

# arXiv's official names for the six primary categories in our sample
CATEGORY_NAMES = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language (natural language processing)",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.IR": "Information Retrieval",
    "cs.LG": "Machine Learning",
}
CATEGORIES = sorted(CATEGORY_NAMES)

# Short descriptions condensed from https://arxiv.org/category_taxonomy. With the bare names,
# zero-shot and few-shot both sent most papers to cs.LG ("Machine Learning" fits nearly every abstract).
CATEGORY_DESCRIPTIONS = {
    "cs.AI": "artificial intelligence: agents, reasoning, planning and knowledge representation",
    "cs.CL": "natural language processing: text, language models, translation and speech",
    "cs.CR": "computer security, privacy and cryptography",
    "cs.CV": "computer vision: images, video, 3D scenes and visual recognition",
    "cs.IR": "information retrieval: search engines, ranking and recommender systems",
    "cs.LG": "machine learning methods: training algorithms, optimisation and learning theory",
}

_model = None


def build_tfidf_logreg(seed=42):
    """TF-IDF (unigrams + bigrams) followed by Logistic Regression, as one sklearn Pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.8,
                                  sublinear_tf=True, ngram_range=(1, 2))),
        ("logreg", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def save_classifier(model, path=CLASSIFIER_PATH):
    joblib.dump(model, path)
    return path


def load_classifier(path=CLASSIFIER_PATH):
    """Load the saved pipeline once and keep it in memory."""
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def _as_list(texts):
    # a bare string would otherwise be split into characters by list()
    return [texts] if isinstance(texts, str) else list(texts)


def predict(texts):
    """Predicted primary category for each text (title + abstract works best)."""
    return list(load_classifier().predict(_as_list(texts)))


def predict_proba(texts):
    """Probability of every category for each text, as a DataFrame."""
    model = load_classifier()
    return pd.DataFrame(model.predict_proba(_as_list(texts)), columns=model.classes_)


def top_keywords(texts, k=5):
    """The k terms with the highest TF-IDF weight in each text, using the classifier's vectoriser."""
    vectoriser = load_classifier().named_steps["tfidf"]
    terms = vectoriser.get_feature_names_out()
    matrix = vectoriser.transform(_as_list(texts))
    keywords = []
    for i in range(matrix.shape[0]):
        row = matrix[i].toarray().ravel()
        best = np.argsort(row)[::-1][:k]
        keywords.append([terms[j] for j in best if row[j] > 0])
    return keywords


# ---------------------------------------------------------------------------
# Few-shot LLM classifier (Week 9 prompting), through the team's llm.generate
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ("You are an arXiv moderator who assigns each computer-science paper to exactly "
                 "one primary category. Answer with the category code only.")


def build_few_shot_prompt(text, examples, describe=False):
    """Category list + one short worked example per category + the paper to classify.

    `examples` is a list of (text, category) pairs taken from the TRAINING set.
    describe=True lists CATEGORY_DESCRIPTIONS instead of the official names.
    """
    labels = CATEGORY_DESCRIPTIONS if describe else CATEGORY_NAMES
    lines = ["Categories:"]
    for code in CATEGORIES:
        lines.append(f"- {code}: {labels[code]}")
    lines.append("")
    for example_text, code in examples:
        lines.append("Paper: " + shorten(example_text, 400))
        lines.append("Category: " + code)
        lines.append("")
    lines.append("Paper: " + shorten(text, 1200))
    lines.append("Category:")
    return "\n".join(lines)


def shorten(text, max_chars):
    """Cut long abstracts at a word boundary so prompts stay short."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def parse_category(reply):
    """Find the first valid category code in the LLM's reply (e.g. 'cs.CL'); None if there is none."""
    match = re.search(r"cs\.(AI|CL|CR|CV|IR|LG)\b", reply, flags=re.IGNORECASE)
    if match:
        return "cs." + match.group(1).upper()
    # fall back to the category names ("Machine Learning" -> cs.LG)
    for code, name in CATEGORY_NAMES.items():
        if name.split(" (")[0].lower() in reply.lower():
            return code
    return None


def llm_classify(text, examples, describe=False, max_new_tokens=10):
    from llm import generate
    reply = generate(build_few_shot_prompt(text, examples, describe), system=SYSTEM_PROMPT,
                     max_new_tokens=max_new_tokens, temperature=0.0)
    return parse_category(reply)
