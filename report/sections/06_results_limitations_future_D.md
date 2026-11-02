# 6. End-to-end results, comparative analysis, limitations and future work (Member D)

> **OUTLINE for Member D to write up in their own words (target ~800 words).** Numbers are from NB06 (`artifacts/06_results.json`, `06_summary_table.csv`) and the other notebooks' results files. Delete this box when done.

## 6.1 End-to-end demo (NB06)
`analyse(query)`: extraction (02) → classifier (03) → MiniLM retrieval, or `recommend(paper_id)` for a pasted paper so it can't retrieve itself (04) → one RAG call (05) → groundedness judge on the synthesis.

| Query | Predicted domain | Retrieved | Synthesis grounded | LLM time* |
|---|---|---|---|---|
| defending LLMs against prompt injection and jailbreak attacks | cs.CR | 4 cs.CR, 1 cs.LG | 3/3 | 30.1 s |
| GNNs to recommend items to users | cs.LG | 5 cs.IR | 4/5 | 78.6 s |
| pasted abstract: SHACL review (test paper) | cs.AI | 3 cs.AI, 1 cs.IR, 1 cs.CL | **0/4** | 23.1 s |

\*Uncached Qwen2.5-1.5B on an Apple-silicon laptop (MPS). Everything before the LLM takes under 0.5 s once models are loaded. All three replies kept the SYNTHESIS / GAP / RECOMMENDATIONS format.

- Discussion points:
  - Query 2 is predicted cs.LG, but all five retrieved papers are cs.IR: the classifier and the retriever disagree on an overlapping topic.
  - Query 3 shows the main failure mode: the synthesis claims "all four papers discuss LLMs", which the judge rejects for every sentence. It is a generic summary, not a grounded one.
  - Across the 3 demos, 7 of 12 synthesis sentences are supported, against 16/17 on the five NB05 questions. Five questions is a small sample.

## 6.2 Comparative analysis: what we kept and why (one table)

| Component | Compared | Kept | Evidence |
|---|---|---|---|
| Extraction | regex, spaCy NER, zero-/few-shot LLM, hybrid | hybrid | F1 0.682 vs 0.630 few-shot (paired CI of the gain +0.02 to +0.09) |
| Classification | NB, LogReg, LSTM, DistilBERT, NLI, LLM, ensembles | TF-IDF 1–2 gram LogReg | macro-F1 0.792 |
| Retrieval | TF-IDF, avg. word vectors, MiniLM | MiniLM | nDCG@5 0.708 |
| Summary | extractive, BART, T5, LLM ladder | LLM few-shot | same rubric (3.70) at half the length; 51/51 grounded |
| Gap finding | direct, zero-shot CoT, few-shot CoT | direct prompt inside RAG | CoT did not improve specificity (0.174–0.382 vs 0.368) |
| Grounding | RAG vs no RAG | RAG | 94% vs 25% supported sentences |

Pattern: classic methods win on labelled tasks (classification, extraction precision). The LLM helps where the output is free text, but only when it is grounded by retrieval.

## 6.3 Limitations
- Proxy labels: relevance = same primary category; single-label classification although 55% of papers are cross-listed.
- Small evaluation sets: 20 extraction abstracts (AI-drafted labels, confirmed by A), 20 summaries, 10 CoT papers, 5 RAG questions, 9 injection attempts.
- A 1.5B model that also acts as its own judge (validated only on clear-cut planted claims).
- No validation split in NB03; the hybrid extractor was designed on its evaluation set.
- No citation graph in the arXiv metadata; no affiliations, so institution NER is not possible.
- The guard sentence did not reduce prompt injection (2/9 vs 3/9).

## 6.4 Future work
- Human relevance labels and human summary ratings (template in `artifacts/05_human_check.csv`).
- A larger LLM or a separate judge model; a reranker or hybrid TF-IDF + MiniLM retrieval.
- Multi-label classification; SciBERT/SPECTER embeddings; citations from the Semantic Scholar API.
- Stronger injection defences: input sanitisation, or delimiting abstracts and checking for the injected string.
