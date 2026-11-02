# 4. Retrieval, recommendation and topic discovery (Member C)

> **OUTLINE for Member C to write up in their own words (target ~700 words).** Every number below is from NB04 (`artifacts/04_results.json`). Delete this box when done.

## 4.1 System architecture (figure)
- One diagram: query/abstract → `extraction` (NB02) → `classify` (NB03) → `retrieve` / `recommend` (NB04, MiniLM index) → RAG prompt + LLM (NB05, Qwen2.5-1.5B) → groundedness judge → output. Slide 4 already has the stages; redraw it as a figure.
- Mark which boxes are basic (Weeks 1–6) and which are advanced (Weeks 7–12).

## 4.2 Techniques and why (2–4 lines each)
- **Three representations (W4):** TF-IDF 1–2 grams (sparse, lexical); averaged pretrained word vectors (spaCy `en_core_web_md`); `all-MiniLM-L6-v2` sentence embeddings (an encoder transformer, W8).
- **Cosine-similarity retrieval** for each representation; MiniLM wins and becomes `retrieval.retrieve`, which the RAG step uses (W11–12).
- **KMeans, k = 6** in each space, to see if unsupervised topics recover the arXiv domains.
- **Evaluation design:** index = 9,600 training papers, queries = 200 test papers (no self-matches); relevant = same primary category; P@5 and nDCG@5.

## 4.3 Results (tables)

| Retriever | P@5 | nDCG@5 |
|---|---|---|
| MiniLM sentence embeddings | **0.704** | **0.708** |
| TF-IDF (1–2 grams) | 0.619 | 0.636 |
| Averaged word vectors | 0.420 | 0.420 |
| Random baseline | 0.167 | 0.167 |

The three-way comparison uses a 9,600-paper training index. On the **deployed 12,000-paper MiniLM index**, with each query paper excluded, P@5 is **0.717** and nDCG@5 is **0.720** on the same 200 test queries.

| Space | Purity | ARI |
|---|---|---|
| MiniLM | **0.727** | **0.484** |
| TF-IDF (SVD-100) | 0.572 | 0.263 |
| Averaged word vectors | 0.296 | 0.046 |

- P@5 by query domain (MiniLM): cs.CR 0.865, cs.IR 0.843, cs.CV 0.812, cs.CL 0.754, cs.AI 0.526, cs.LG 0.440.
- Clean MiniLM clusters: cs.IR 92%, cs.CR 89%, cs.CV 84%; cs.LG (67%), cs.AI (63%), cs.CL (56%) are mixed.
- Latency depends on hardware and cache state; the current CPU run measured 88.9 ms per query including query encoding.

## 4.4 Discussion points
- Why averaged word vectors are worst: word order is lost and frequent generic words dominate the average.
- Why MiniLM beats TF-IDF: it can match paraphrases, whereas TF-IDF needs shared words (find one real query in NB04 that shows this before claiming it).
- cs.LG and cs.AI are worst for both retrieval and classification (NB03 confusion pairs): the labels overlap, not just the models.
- PCA plots explain little variance (TF-IDF 0.9%, MiniLM 12.9%), so they illustrate the clusters but don't prove them.

## 4.5 Limitations and improvements
- Same-category relevance is a proxy; a cross-listed paper counts as a miss.
- MiniLM is a general-domain model (try SPECTER/SciBERT embeddings); 200 queries; no citation graph in arXiv metadata (Semantic Scholar API as future work); hybrid TF-IDF + MiniLM retrieval or a reranker.
