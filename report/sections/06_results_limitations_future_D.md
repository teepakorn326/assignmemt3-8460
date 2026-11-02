# 6. End-to-end results, comparative analysis, limitations and future work (Member D)

> **OUTLINE for Member D to write up in their own words (target ~800 words).** Numbers are from NB06 (`artifacts/06_results.json`, `06_summary_table.csv`) and the other notebooks' results files. Delete this box when done.

## 6.1 End-to-end demo (NB06)
`analyse(query)`: key phrases for short topics or the evaluated hybrid extractor for abstracts (02) → classifier (03) → MiniLM retrieval, or `recommend(paper_id)` for a known paper so it cannot retrieve itself (04) → a few-shot single-paper summary for abstracts and RAG synthesis (05) → sanitised abstracts → per-claim citation filter. Each synthesis sentence is kept only if it cites a retrieved paper and the judge finds it supported by that paper; if none survives, cited sentences copied from the retrieved abstracts are shown as an explicit fallback.

| Query | Predicted domain | Retrieved | Generated: judge-supported; cited | Kept by per-claim filter | Final |
|---|---|---|---|---|---|
| defending LLMs against prompt injection and jailbreak attacks | cs.CR | 4 cs.CR, 1 cs.LG | 1/3; 1/3 cited | 0/3 | 3/3 exact-source fallback |
| GNNs to recommend items to users | cs.LG | 5 cs.IR | 3/3; none cited | 0/3 | 3/3 exact-source fallback |
| pasted abstract: SHACL review (test paper) | cs.AI | 3 cs.AI, 1 cs.IR, 1 cs.CL | 1/2; none cited | 0/2 | 3/3 exact-source fallback |

All three replies kept the SYNTHESIS / GAP / RECOMMENDATIONS headings, but no generated sentence survived the per-claim filter: 7 of 8 were dropped for having no citation, and the one cited sentence was not supported by the papers it cites. The final fallback copies sentences verbatim from numbered retrieved abstracts. The gap and recommendation reasons remain generated and unverified.

- Discussion points:
  - Why fallback: the earlier gate also required verbatim copying, which rejected any paraphrase. With that removed, the remaining cause is the 1.5B model not following the citation instruction on these queries (it did on 10/21 NB05.1 sentences).
  - Model-size check (NB06.1): with Qwen2.5-3B as the RAG generator and everything else unchanged, all three demo answers pass the per-claim filter (12/13 sentences kept, 0/3 fallback). NB06 keeps 1.5B so that every evaluated number uses one model.
  - Query 2 is predicted cs.LG, but all five retrieved papers are cs.IR: the classifier and the retriever disagree on an overlapping topic. Its three uncited sentences were all judge-supported, so a citation would likely have kept them.
  - The fallback is an extractive safety response, not evidence that the LLM became more accurate. Five NB05 RAG questions and three demos are small samples.

## 6.2 Comparative analysis: what we kept and why (one table)

| Component | Compared | Kept | Evidence |
|---|---|---|---|
| Extraction | regex, spaCy NER, zero-/few-shot LLM, hybrid | hybrid | F1 0.634 vs 0.599 few-shot; paired CI of the gain −0.005 to +0.075 includes zero, so it is kept for its higher recall (0.680 vs 0.587), not a proven F1 gain |
| Classification | NB, LogReg, LSTM, DistilBERT, NLI, LLM, ensembles | TF-IDF 1–2 gram LogReg | macro-F1 0.792; the larger DistilBERT ensemble reaches 0.805 but is not used in the demo |
| Retrieval | TF-IDF, avg. word vectors, MiniLM | MiniLM | nDCG@5 0.708 on the training index; 0.720 on the deployed 12k index with self excluded |
| Summary | extractive, BART, T5, LLM ladder | LLM few-shot | same rubric (3.70) at half the length; 51/51 grounded |
| Gap finding | direct, zero-shot CoT, few-shot CoT | direct prompt inside RAG | CoT did not improve specificity (0.174–0.382 vs 0.368) |
| Grounding | no RAG, RAG v1 (NB05), RAG v2 (NB05.1) | RAG v2 + per-claim filter | judge-supported 40% → 76% → 86%; cited sentences 0/20 → 5/17 → 10/21 |
| RAG generator | Qwen2.5 1.5B vs 3B (NB06.1) | 1.5B in evaluated notebooks; 3B recommended for deployment | 3B: 26/29 cited, 0/8 fallback vs 11/29, 5/8; judge support similar (24/29 vs 23/29) |
| Injection defence | none, guard sentence (NB05), sanitizer + guard (NB05.1) | sanitizer + guard | 0/9 vs 3/9 on known attacks; 4/9 in every setting on held-out attacks |

Pattern: LogReg is a simple, competitive classifier; the larger DistilBERT ensemble has the highest measured classification score. The LLM helps with free-text output, but its generated synthesis needs grounding and citation checks.

## 6.3 Limitations
- Proxy labels: relevance = same primary category; single-label classification although 55% of papers are cross-listed.
- The paper-trained classifier scores macro-F1 0.698 on titles alone; its domain label for a short typed query is only a rough guide.
- Small evaluation sets: 20 extraction abstracts (AI-drafted labels reviewed by one team member, no inter-annotator agreement), 20 summaries, 10 CoT papers, 5 RAG questions, 9 injection attempts.
- A 1.5B model that also acts as its own judge (validated only on clear-cut planted claims).
- NB03 now has five-fold cross-validation for LogReg and validation-based epoch choices, but its label descriptions were chosen after inspecting test scores; the hybrid extractor was designed on its evaluation set.
- No citation graph in the arXiv metadata; no affiliations, so institution NER is not possible.
- The guard sentence did not reduce prompt injection (3/9 either way); the regex sanitizer stops the known attacks (0/9) but none of the held-out ones. All three demo syntheses still needed extractive fallback because the model rarely cited; the generated gaps and recommendation reasons are not independently grounded.

## 6.4 Future work
- Human relevance labels and human summary ratings (template in `artifacts/05_human_check.csv`).
- A larger LLM or a separate judge model; a reranker or hybrid TF-IDF + MiniLM retrieval.
- Multi-label classification; SciBERT/SPECTER embeddings; citations from the Semantic Scholar API.
- Stronger injection defences: a learned injection classifier instead of fixed patterns, or delimiting abstracts and checking outputs for the injected string.
- Post-hoc citation attribution: attach the best-matching retrieved paper to an uncited sentence and judge it against that paper, so supported but uncited sentences are not dropped.
