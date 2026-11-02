# 6. End-to-end results, comparative analysis, limitations and future work (Member D)

> **OUTLINE for Member D to write up in their own words (target ~800 words).** Numbers are from NB06 (`artifacts/06_results.json`, `06_summary_table.csv`) and the other notebooks' results files. Delete this box when done.

## 6.1 End-to-end demo (NB06)
`analyse(query)`: key phrases for short topics or the evaluated hybrid extractor for abstracts (02) → classifier (03) → MiniLM retrieval, or `recommend(paper_id)` for a known paper so it cannot retrieve itself (04) → a few-shot single-paper summary for abstracts and RAG synthesis (05) → grounding and citation checks. If generated synthesis fails, cited sentences copied from the retrieved abstracts are shown as an explicit fallback.

| Query | Predicted domain | Retrieved | Generated support (judge and count check) | Final exact-source support |
|---|---|---|---|---|
| defending LLMs against prompt injection and jailbreak attacks | cs.CR | 4 cs.CR, 1 cs.LG | 2/3; no citations | 3/3 fallback |
| GNNs to recommend items to users | cs.LG | 5 cs.IR | 5/5; 3/5 cited | 3/3 fallback |
| pasted abstract: SHACL review (test paper) | cs.AI | 3 cs.AI, 1 cs.IR, 1 cs.CL | **0/4; no citations** | 3/3 fallback |

All three generated replies kept the SYNTHESIS / GAP / RECOMMENDATIONS headings, but none passed both support and exact cited-source checks. The judge sees only the excerpts supplied to the generator. The final fallback copies sentences verbatim from numbered retrieved abstracts and verifies them by exact text matching. The gap and recommendation reasons remain generated and unverified.

- Discussion points:
  - Query 2 is predicted cs.LG, but all five retrieved papers are cs.IR: the classifier and the retriever disagree on an overlapping topic.
  - Query 3 shows the main failure mode: the synthesis claims "all four papers discuss LLMs" despite five retrieved sources. The explicit count check catches this error; none of its four synthesis claims passes the combined support check.
  - Across the 3 demos, 7 of 12 **generated** synthesis sentences are supported. The fallback supplies 9/9 exact source sentences across the three final replies; this is an extractive safety response, not evidence that the LLM became more accurate. Five NB05 RAG questions is a small sample.

## 6.2 Comparative analysis: what we kept and why (one table)

| Component | Compared | Kept | Evidence |
|---|---|---|---|
| Extraction | regex, spaCy NER, zero-/few-shot LLM, hybrid | hybrid | F1 0.671 vs 0.628 few-shot (paired CI of the gain +0.005 to +0.083; exploratory) |
| Classification | NB, LogReg, LSTM, DistilBERT, NLI, LLM, ensembles | TF-IDF 1–2 gram LogReg | macro-F1 0.792; the larger DistilBERT ensemble reaches 0.805 but is not used in the demo |
| Retrieval | TF-IDF, avg. word vectors, MiniLM | MiniLM | nDCG@5 0.708 on the training index; 0.720 on the deployed 12k index with self excluded |
| Summary | extractive, BART, T5, LLM ladder | LLM few-shot | same rubric (3.70) at half the length; 51/51 grounded |
| Gap finding | direct, zero-shot CoT, few-shot CoT | direct prompt inside RAG | CoT did not improve specificity (0.174–0.382 vs 0.368) |
| Grounding | RAG vs no RAG | RAG | 76% vs 40% judge-supported sentences against supplied excerpts; only 5/17 RAG claims cited |

Pattern: LogReg is a simple, competitive classifier; the larger DistilBERT ensemble has the highest measured classification score. The LLM helps with free-text output, but its generated synthesis needs grounding and citation checks.

## 6.3 Limitations
- Proxy labels: relevance = same primary category; single-label classification although 55% of papers are cross-listed.
- The paper-trained classifier scores macro-F1 0.698 on titles alone; its domain label for a short typed query is only a rough guide.
- Small evaluation sets: 20 extraction abstracts (AI-drafted labels, independent human confirmation pending), 20 summaries, 10 CoT papers, 5 RAG questions, 9 injection attempts.
- A 1.5B model that also acts as its own judge (validated only on clear-cut planted claims).
- NB03 now has five-fold cross-validation for LogReg and validation-based epoch choices, but its label descriptions were chosen after inspecting test scores; the hybrid extractor was designed on its evaluation set.
- No citation graph in the arXiv metadata; no affiliations, so institution NER is not possible.
- The guard sentence did not reduce prompt injection (2/9 vs 3/9). All three demo syntheses needed extractive fallback; the generated gaps and recommendation reasons are not independently grounded.

## 6.4 Future work
- Human relevance labels and human summary ratings (template in `artifacts/05_human_check.csv`).
- A larger LLM or a separate judge model; a reranker or hybrid TF-IDF + MiniLM retrieval.
- Multi-label classification; SciBERT/SPECTER embeddings; citations from the Semantic Scholar API.
- Stronger injection defences: input sanitisation, or delimiting abstracts and checking for the injected string.
