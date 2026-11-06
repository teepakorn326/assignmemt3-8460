# 6. End-to-end results, comparison, limitations and future work (Member D)

## 6.1 End-to-end demo (NB06)
One function, `analyse(query)`, joins the components without retraining anything (Figure 1). It extracts key phrases (and, for a pasted abstract, methods and datasets with the hybrid extractor), predicts the domain with the TF-IDF classifier, retrieves five papers with MiniLM (excluding the query paper itself), sanitises their abstracts, generates a RAG answer with the v2 prompt and passes the synthesis through the per-claim filter. If no sentence survives, the system shows the first sentence of papers [1]–[3], quoted exactly, instead of an unchecked answer.

| Query | Domain | Retrieved | Generated: supported; cited | Kept | Shown |
|---|---|---|---|---|---|
| Defending LLMs against prompt injection and jailbreaks | cs.CR | 4 cs.CR, 1 cs.LG | 1/3; 1/3 | 0/3 | 3 exact quotes |
| GNNs to recommend items to users | cs.LG | 5 cs.IR | 3/3; 0/3 | 0/3 | 3 exact quotes |
| Pasted abstract (SHACL review) | cs.AI | 3 cs.AI, 1 cs.IR, 1 cs.CL | 1/2; 0/2 | 0/2 | 3 exact quotes |

*Table 9. Demo queries with Qwen2.5-1.5B.*

The extraction, classification and retrieval steps took 2.5 s on the first query, including model loading. All three answers kept the required format, but no generated sentence passed the filter: seven of eight had no citation, and the one cited sentence was not supported by the papers it cited. The second query shows that the filter enforces traceability rather than only truth, since its three uncited sentences were all judge-supported. It also shows two components disagreeing: the classifier says cs.LG while all five retrieved papers are cs.IR. With the 3B generator, all three demo answers passed the filter. The fallback is a safety response, not evidence that the model became more accurate, and the gap and recommendation reasons are still unverified. For a user, the 1.5B system currently gives the domain, key phrases and five papers to read, but quoted sentences instead of a synthesis; the 3B generator makes the cited answer usable.

## 6.2 What we compared and kept

| Component | Compared | Kept | Evidence |
|---|---|---|---|
| Extraction | regex, spaCy NER, zero-/few-shot LLM, hybrid | hybrid | F1 0.634 vs 0.599 and recall 0.68 vs 0.59; the gain's CI includes zero |
| Classification | NB, LogReg, LSTM, DistilBERT, NLI, LLM, ensembles | TF-IDF 1–2 gram LogReg | macro-F1 0.792; the ensemble's 0.805 needs a GPU and a second model |
| Retrieval | TF-IDF, averaged word vectors, MiniLM | MiniLM | nDCG@5 0.708; 0.720 on the deployed 12k index |
| Summary | extractive, BART, T5, LLM prompts | few-shot LLM | same rubric at half the length; 51/51 supported |
| Research gap | direct, zero-shot CoT, few-shot CoT | direct prompt in RAG | CoT did not improve specificity |
| Grounding | no RAG, RAG v1, RAG v2 + filter | RAG v2 + filter | supported 40% → 76% → 86% |
| Injection defence | none, guard, sanitiser + guard | sanitiser + guard | 0/9 known attacks; 4/9 held-out |

*Table 10. Final choices.*

Across the system, simple methods held up where the task is decided by vocabulary (classification) and pre-trained embeddings won where meaning matters (retrieval). The LLM was most useful for free text, but only once it was grounded in retrieved papers and its claims were checked. For a researcher, one query returns the field, the key methods, five papers to read next and a cited summary and gap, in place of a keyword list. Retrieval makes that answer checkable: support doubled with the evidence in the prompt, and the filter drops any sentence that does not cite a retrieved paper or is not supported by it.

## 6.3 Limitations
- Relevance is approximated by the same primary category, and each paper gets one label although 55% are cross-listed.
- Evaluation sets are small: 20 extraction abstracts (AI-drafted labels reviewed by one member), 20 summaries, 10 CoT papers, 5 RAG questions, 9 injection attempts per setting.
- The judge is the same 1.5B model as the generator and was validated only on clear-cut errors; there are no human ratings yet.
- The classifier scores 0.698 on titles alone, so its label for a short query is a rough guide.
- arXiv metadata has no citation graph and no affiliations, so mutual-citation analysis and institution NER were not possible.

## 6.4 Future work
- Use Qwen2.5-3B, or a larger model, as the RAG generator, with a separate judge model.
- Add human relevance labels and human summary ratings (template in `artifacts/05_human_check.csv`).
- Replace the pattern sanitiser with a learned injection detector.
- Attach the best-matching paper to uncited sentences instead of dropping them, and test multi-label classification and scientific embeddings (SciBERT, SPECTER).
