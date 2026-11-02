# 5. LLM summarisation, research gaps and RAG (Member D)

> **OUTLINE for Member D to write up in their own words (target ~900 words).** Every number is from NB05 (`artifacts/05_results.json`, 20 test papers unless stated). Delete this box when done.

## 5.1 Setup
- Foundation LLM: Qwen2.5-1.5B-Instruct, run locally through `llm.generate`, temperature 0, every call cached in `artifacts/llm_cache.jsonl` (reproducible). Shared prompts, parser and judge are in `rag.py`.
- Baselines: extractive (top-3 TF-IDF sentences, basic), BART-large-CNN and T5-base (Week 9 supplementary).

## 5.2 Prompt engineering ladder (W9)
- basic → role → instruction → few-shot. Mean length: basic 117.6, role 130.6, instruction 55.8, few-shot 50.5 words (abstract 163.2). None was cut off at 300 new tokens.
- Point: instructions and examples control length and format; the few-shot prompt is kept for single-paper summaries.

## 5.3 Summary quality

| System | ROUGE-L | Rubric 1–5 | Grounded claims |
|---|---|---|---|
| Extractive | 0.308 | 3.72 | 59/59 |
| BART | **0.371** | 3.55 | 49/50 |
| T5 | 0.288 | 3.43 | 54/55 |
| LLM basic | 0.171 | **3.73** | – |
| LLM few-shot | 0.259 | 3.70 | 51/51 |

- ROUGE reference = title + first abstract sentence, which rewards copying the lead (state this plainly).
- The rubric is judged by the same Qwen model (self-grading bias); the top three are within 0.03, so it can't separate them.
- **Judge validation:** 40/40 copied sentences → SUPPORTED; 23/23 planted claims → UNSUPPORTED. It is reliable on clear-cut cases; subtle errors were not tested.

## 5.4 Research gaps: CoT ablation (10 papers)

| Prompt | Reasoned first | Specificity | Judge 1–5 |
|---|---|---|---|
| Direct (no CoT) | n/a | **0.368** | **3.3** |
| Zero-shot CoT | 0/10 | 0.382 | 3.3 |
| Few-shot CoT | 10/10 | 0.174 | 3.0 |

- Specificity = share of the gap's content words that appear in the abstract (generic gaps score low).
- Finding: CoT did **not** improve gaps with a 1.5B model. Zero-shot CoT was ignored; few-shot CoT copied the example's format and gave generic gaps.

## 5.5 RAG ablation (5 research questions)

| Mode | Format OK | Supported synthesis sentences | Cited synthesis sentences | Valid [n] markers |
|---|---|---|---|---|
| RAG | 5/5 | 16/17 (94%) | 5/17 | 9 |
| No RAG | 4/5 | 5/20 (25%) | 0/20 | 0 |

- Finding: retrieval gives the judge more support (16/17 vs 5/20), but most generated synthesis sentences still lack a citation. The end-to-end demo checks both support and citation coverage before displaying a synthesis as verified.

## 5.6 Safety (W10)
- Prompt injection, 3 attacks × 3 abstracts, same few-shot prompt, only the guard sentence toggled: succeeded 2/9 without the guard, 3/9 with it. The guard sentence adds no measurable protection; the few-shot template itself resists most attacks.
- Correct the slide: the earlier "blocked" example changed the prompt template and the guard at the same time.

## 5.7 Limitations
- 1.5B model; 20 / 10 / 5 samples; the judge is the generator; the ROUGE reference comes from the abstract; no human ratings yet (template `artifacts/05_human_check.csv`, 10 summaries); the injection check is an exact-string match (one partial hijack was missed).
