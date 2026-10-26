# AGENTS.md — instructions for AI coding assistants working in this repo

You are helping one member of a 4-person student team build **COMP8460 Assignment 3, Use Case 3: Research Paper Analysis and Recommendation System** on the arXiv metadata dataset. Read this whole file before touching anything. The human will tell you which member they are (A, B, C or D); if they don't, ask.

## Hard constraints

- **Deadline: code + presentation Thu 29 Oct 2026. Report by Thu 5 Nov.** Prefer a simple method that runs and produces a number over an ambitious one that might not finish.
- **Only edit files owned by this member** (table below). Never edit another member's notebook or module. `codes/utils.py` is frozen; if it has a bug, tell the human to raise it with A instead of patching it.
- **Lift code from the unit's prac notebooks** when a prac covers the technique (the human has them locally; they are not in this repo). Match library choices the pracs use (NLTK, spaCy, scikit-learn, Hugging Face `transformers`, `sentence-transformers`). Do not introduce new frameworks.
- **Academic integrity:** code must be the team's own work with citations for external snippets. Add a one-line comment with a URL when you adapt anything from outside the pracs. Do not fabricate results, metrics or references.
- **Reproducibility:** seed 42 everywhere, use the fixed split, cache LLM outputs, save metrics with `save_results`.
- The spec says explanations should be **2–4 lines each**. Keep notebook markdown short. Don't pad.

## Repository layout

```
codes/
  01_data_loading_and_sampling.ipynb   done (A). Produces data/ below. Do not rerun unless asked.
  02_preprocessing_extraction.ipynb    A
  03_classification_keywords.ipynb     B
  04_retrieval_recommendation.ipynb    C
  05_llm_summarisation_gaps.ipynb      D
  06_end_to_end_demo.ipynb             D  (the live demo in the presentation)
  utils.py        frozen. load_sample(), load_split(), retrieve(), save_results(), load_results()
  llm.py          D owns. generate(prompt, system=None, max_new_tokens=300, temperature=0.0)
  extraction.py   A creates    classify.py  B creates    retrieval.py  C creates
  data/arxiv_cs_sample.jsonl.gz   12,000 CS papers, 2,000 per category, cached. The ONLY data input.
  data/split_ids.json             fixed stratified 80/20 split (9,600 / 2,400), seed 42
  artifacts/      *_results.json (committed), llm_cache.jsonl (committed), *.pkl models (git-ignored)
  requirements.txt
report/sections/0X_*.md   one markdown file per member, written as work progresses
TEAM_PLAN.md              roles, interfaces, timeline, week-by-week coverage table
```

Run notebooks from inside `codes/` so `from utils import ...` works.

## Shared interfaces (use these; never reimplement them)

```python
from utils import load_sample, load_split, retrieve, save_results, load_results
from llm import generate

df = load_sample()                 # DataFrame: id, title, abstract, categories, primary_category,
                                   #   all_categories, author_list, text (= title + ". " + abstract), ...
train_df, test_df = load_split(df) # ALWAYS evaluate on test_df only

retrieve("graph neural networks", k=5)
# -> DataFrame [id, title, abstract, primary_category, author_list, score], sorted by score desc.
# Works today (TF-IDF fallback). When C adds codes/retrieval.py exposing retrieve(query, k),
# it is used automatically. D's notebooks call this; the signature and columns must not change.

generate("Summarise this abstract in 2 sentences: ...", system="You are a careful research assistant.")
# -> str. Cached in artifacts/llm_cache.jsonl, so reruns are instant and identical.
# Backend via env: LLM_BACKEND=hf (default, Qwen/Qwen2.5-1.5B-Instruct), openai (needs OPENAI_API_KEY),
# or echo (returns the prompt; lets you build a notebook with no model installed).

save_results("03", {"model": "LogReg TF-IDF", "accuracy": 0.81, "macro_f1": 0.80})
# -> writes artifacts/03_results.json. Every notebook calls this once at the end.
```

Labels for classification = `primary_category` (6 classes). Relevance for retrieval evaluation = same `primary_category` as the query paper.

## Notebook conventions

- Keep the existing section headers in each notebook; fill the `# TODO` cells under them. Add cells, don't restructure.
- First cell stays as is (loads `df`, `train_df`, `test_df`).
- Each technique section = 2–4 lines of markdown justification (why this technique, which week it comes from) + code + one metric or table.
- Use subsets where compute is tight (e.g. 500 test papers for zero-shot, 3k train for fine-tuning) and say so in markdown.
- Save any fitted model to `artifacts/<name>.pkl` (joblib) or `artifacts/models/` so NB06 can load it.
- Last cell: `save_results(...)` with every headline number from the notebook.
- Clear outputs before committing (`jupyter nbconvert --clear-output --inplace <nb>`), except the final run on Wed 28.

## Per-member task list

Each row maps to a lecture/prac week; the code must visibly cover **every** week 1–12. "lite" = run 1 epoch / small subset, report the number, move on.

### Member A — NB02 `02_preprocessing_extraction.ipynb`, `codes/extraction.py`
1. W1: corpus stats per category (doc count, mean tokens, vocab size, type-token ratio), top-20 word frequency plot, one word cloud.
2. W2: `preprocess(text) -> list[str]`: lowercase, tokenise, remove stop-words/punctuation, lemmatise (spaCy `en_core_web_sm`). Compare stem vs lemma on 5 examples. Put the function in `extraction.py`.
3. W3: POS-pattern key phrases (`ADJ* NOUN+`) and spaCy `noun_chunks`; spaCy NER (PERSON, ORG, DATE) plus regex for method/dataset acronyms (`\b[A-Z][A-Za-z]*[A-Z]\w*\b`, e.g. BERT, ImageNet) and years.
4. Advanced (W9 prompting): few-shot `generate()` prompt with 3 worked examples returning JSON `{methods, datasets, tasks}`.
5. Eval: 20 hand-labelled test abstracts (the human labels; you build the labelling template and scorer). Precision/recall/F1 per extractor: regex vs spaCy vs LLM. `save_results("02", ...)`.

### Member B — NB03 `03_classification_keywords.ipynb`, `codes/classify.py`
1. W4/W5: TF-IDF top keywords per category; NB on BoW, NB on TF-IDF, LogReg on TF-IDF. Accuracy, macro-F1, confusion matrix (seaborn heatmap).
2. W7: zero-shot classification pipeline (`facebook/bart-large-mnli`) with the 6 category names as labels, on 500 test papers.
3. W6 (lite): the simplified LSTM from prac 6 with a HF tokenizer, 1–2 epochs, on titles+abstracts truncated to 256 tokens.
4. W9 (lite): `distilbert-base-uncased` fine-tune with HF `Trainer`, 1 epoch, 3,000 train papers, evaluate on full test. Needs GPU (Colab) — if unavailable, use frozen DistilBERT [CLS] embeddings + LogReg instead and say so.
5. Advanced: few-shot LLM classifier via `generate()` on 200 test papers.
6. One comparison table of all models. Save the best classical model as `artifacts/classifier.pkl` with a `predict(texts)` wrapper in `classify.py`. `save_results("03", ...)`.

### Member C — NB04 `04_retrieval_recommendation.ipynb`, `codes/retrieval.py`
1. W4: three document representations — TF-IDF, averaged pretrained word vectors (spaCy `en_core_web_md` or GloVe as in prac 4 §4), and `all-MiniLM-L6-v2` sentence embeddings. PCA 2-D plot of 600 papers coloured by category.
2. W4/W8: cosine-similarity retrieval for each representation. Note in markdown that MiniLM is an encoder model (W8).
3. Clustering: KMeans k=6 on MiniLM embeddings; purity and adjusted Rand index vs `primary_category`; top TF-IDF terms per cluster.
4. Eval: for 200 random test papers as queries, P@5 and nDCG@5 with same-category relevance, for all three retrievers. One table.
5. W12 RAG: `codes/retrieval.py` with `retrieve(query, k=5)` returning the exact columns listed above, backed by MiniLM embeddings cached to `artifacts/minilm_embeddings.npy` (sklearn `NearestNeighbors`, cosine). This is the retriever D's RAG uses. `save_results("04", ...)`.

### Member D — NB05 `05_llm_summarisation_gaps.ipynb`, NB06 `06_end_to_end_demo.ipynb`, `codes/llm.py`
1. Basic: extractive baseline — top-3 TF-IDF-scored sentences of the abstract.
2. W9 supp: BART (`facebook/bart-large-cnn`) and T5 summaries on 10 test abstracts.
3. W9: prompt ladder with `generate()` — basic → role → instruction → few-shot — for 2-sentence summaries; zero-shot CoT ("think step by step") and few-shot CoT for research-gap generation.
4. W12 RAG: for a query, `retrieve(query, 5)` → prompt with the 5 abstracts → synthesis + 3 recommended papers with one-line reasons.
5. W10: groundedness check — split each LLM summary into claims, ask the LLM whether each claim is supported by the source abstract, report % grounded over 10 samples; a safety system prompt; one test where an abstract contains an injected instruction ("ignore previous instructions and ...") and show the guardrail holds.
6. Eval: ROUGE-1/2/L of extractive vs BART vs LLM against the title + first sentence; 1–5 rubric on 10 samples. `save_results("05", ...)`.
7. NB06: `analyse(query) -> dict` = preprocess (A's `extraction.py` if present) → classify (B's `classifier.pkl` if present, else skip) → `retrieve` → LLM summary, gaps, recommendations → groundedness score. Build it on stand-ins first (`LLM_BACKEND=echo`), swap real parts in as they land. 3 demo queries, one summary table from `load_results()` of all notebooks. Stretch (W11/W12): wrap `retrieve`/`summarise` as smolagents tools; voice query via wav2vec.

## Definition of done for a notebook

- Runs top to bottom on a fresh kernel in under ~20 min on CPU (except the marked GPU step).
- Every section has a 2–4 line justification and at least one number or table.
- `save_results` called; model pickles saved where NB06 needs them.
- Outputs cleared; committed on the member's branch; PR to `main`.

## Git

Branches: `a-extraction`, `b-classify`, `c-retrieval`, `d-llm`. Small commits with clear messages (the commit history is each member's contribution evidence). Never commit `dataset/` or the raw 5.6 GB arXiv file. Never force-push `main`.

## Do not

- Do not change `utils.py`, `data/`, or another member's files.
- Do not change the `retrieve()` return columns or the `generate()` signature.
- Do not add dependencies beyond `requirements.txt` without telling the human why.
- Do not invent metrics, citations, or dataset facts. If something can't be computed in time, write "not run — out of time" in the markdown rather than a made-up number.
