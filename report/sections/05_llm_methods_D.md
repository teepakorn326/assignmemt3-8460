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
- **Judge validation:** 40/40 copied sentences → SUPPORTED; 23/23 planted claims → UNSUPPORTED. It still accepted a "four papers" claim when five were retrieved in the earlier RAG run, so an explicit count check was added; subtle errors remain untested.

## 5.4 Research gaps: CoT ablation (10 papers)

| Prompt | Reasoned first | Specificity | Judge 1–5 |
|---|---|---|---|
| Direct (no CoT) | n/a | **0.368** | **3.3** |
| Zero-shot CoT | 0/10 | 0.382 | 3.3 |
| Few-shot CoT | 10/10 | 0.174 | 3.0 |

- Specificity = share of the gap's content words that appear in the abstract (generic gaps score low).
- Finding: CoT did **not** improve gaps with a 1.5B model. Zero-shot CoT was ignored; few-shot CoT copied the example's format and gave generic gaps.

## 5.5 RAG ablation (5 research questions)

NB05 is the first version (v1); NB05.1 re-runs RAG with a revised prompt (v2) that states the paper count and asks for [n] at the end of every sentence. Same questions, papers, model and judge.

| Mode | Format OK | Judge-supported synthesis sentences | Cited synthesis sentences | Kept by per-claim filter | Exact-source copies |
|---|---|---|---|---|---|
| No RAG (NB05) | 4/5 | 8/20 (40%) | 0/20 | 0/20 | 0/20 |
| RAG v1 (NB05) | 5/5 | 13/17 (76%) | 5/17 | 5/17 | 0/17 |
| RAG v2 (NB05.1) | 5/5 | 18/21 (86%) | 10/21 | 10/21 | 0/21 |

- Per-claim filter (`rag.filter_cited_claims`): keep a sentence only if it cites a retrieved paper, does not misstate the paper count, and the judge finds it supported by the excerpt of the paper it cites. No sentence is copied verbatim in either version, so exact copying is reported but not used as a gate (it rejects every faithful paraphrase).
- Finding: retrieval gives the judge more support (40% → 76–86%), and the v2 prompt doubles the cited sentences, but the 1.5B model still ignores the citation instruction for about half of its sentences.
- Generator size (NB06.1, same 5 questions + 3 demo queries, same prompt, papers, judge and filter; only the generator changes): Qwen2.5-3B cites 26/29 sentences vs 11/29 for 1.5B, the filter keeps 23/29 vs 10/29, and 0/8 answers need fallback vs 5/8. Judge support is similar (24/29 vs 23/29): the larger model is not more factual, it follows the citation instruction, so its claims become checkable. 3B tends to write one cited sentence per paper rather than a cross-paper synthesis.

## 5.6 Safety (W10)
Prompt injection, 3 attacks × 3 abstracts, same few-shot prompt, only the defence changed. NB05 tested the guard sentence; NB05.1 adds `rag.sanitize` (regex removal of instruction-like sentences) and three held-out attacks written after its patterns.

| Attacks | Defence | Succeeded (exact "P = NP") | Incl. partial hijack |
|---|---|---|---|
| original | none | 2/9 | 3/9 |
| original | guard sentence | 3/9 | 3/9 |
| original | sanitizer + guard (NB05.1) | **0/9** | **0/9** |
| held-out (NB05.1) | none / guard / sanitizer + guard | 4/9 each | 4/9 each |

- The guard sentence adds no measurable protection; it is kept only because it costs nothing.
- The sanitizer stops the known attacks and changes none of the 12,000 sample abstracts, but removed none of the held-out attacks: pattern filters block known phrasings only.
- Correct the slide: the earlier "blocked" example changed the prompt template and the guard at the same time.

## 5.7 Limitations
- 1.5B model; 20 / 10 / 5 samples; the judge is the generator; the ROUGE reference comes from the abstract; no human ratings yet (template `artifacts/05_human_check.csv`, 10 summaries); six injection attacks on three abstracts, and the held-out attacks were written by the same team as the sanitizer.
