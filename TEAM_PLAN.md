# Team plan v2 (26 Oct): Assignment 3, Use Case 3 (arXiv paper analysis & recommendation)

Use case confirmed: **Use Case 3**. Due **Thu 6 Nov 2026, 11:55 PM** (11 days). Presentation is in **Week 13 = 2–6 Nov**, i.e. the same week as the deadline, possibly before it, so slides cannot be left to the last two days.

> ## ⚠️ CRASH SCHEDULE (26 Oct): **code + presentation by Thu 29 Oct**, report by Fri 6 Nov
> The timeline further down is superseded. Scope is cut to what the rubric needs plus one spare.
>
> **Scope (4 basic + 4 advanced, spec needs 3 + 3):**
> - A: preprocessing + POS key phrases + spaCy NER/regex (basic) · few-shot LLM extraction (advanced). Hand-label **20** abstracts.
> - B: NB vs LogReg on TF-IDF (basic) · zero/few-shot LLM classifier **or** frozen-DistilBERT + LogReg (advanced). No full fine-tune, no BiLSTM, no ensemble.
> - C: TF-IDF vs MiniLM retrieval + KMeans (basic) · `retrieval.py` RAG retriever (advanced). No Word2Vec.
> - D: extractive baseline (basic) · LLM summary + CoT gaps + RAG synthesis + groundedness check on **10** samples (advanced). NB06 demo on 3 queries = the live demo in the presentation.
> - LLM: whichever runs **today** (`LLM_BACKEND=openai` is fine; D commits `llm_cache.jsonl`).
>
> **Phase 1 — code + slides (Mon 26 → Thu 29)**
> | When | Everyone | D additionally |
> |---|---|---|
> | **Mon 26 tonight** | repo cloned, notebook runs, **basic technique working with a number** | NB06 `analyse(query)` on stand-ins; slide template shared; LLM backend decided |
> | **Tue 27** | **advanced working**, `save_results()`, pickle saved, PR to main by 22:00; **write 2–3 slides** (what / why / result table / one limitation) | NB05 done; merge PRs |
> | **Wed 28 by 12:00** | slides in the shared deck | NB06 on real components |
> | **Wed 28 evening** | **full rehearsal** (15 min: intro 2 · A 3 · B 3 · C 3 · D+demo 3 · Q&A) | clean run all notebooks; export slides PDF |
> | **Thu 29** | present; each person answers questions on their own part | commit final outputs with cells executed |
>
> **Phase 2 — report (Fri 30 Oct → Fri 6 Nov)**
> | When | What |
> |---|---|
> | Fri 30 – Sun 1 Nov | each member: 600–800 words in `report/sections/`, from their notebook markdown + slides |
> | Mon 2 – Tue 3 | D merges, adds architecture diagram, results table, limitations/future work; others review their own section |
> | Wed 4 | report PDF (2,500–3,000 words); contribution forms + commit-log evidence |
> | Thu 5 | final clean run, rename `GroupID`, zip, **submit** (one day early) |
>
> Rule: advanced technique not working by **Tue 18:00** → switch to LLM zero-shot via `llm.generate` and move on. Slides need a number in every section; a simple method with a result beats an ambitious one without.

## Status check (26 Oct)
- Done: `01_data_loading_and_sampling` (12k CS papers, 6 categories, fixed 80/20 split), `utils.py`, stubs for NB02–06, README, folder structure.
- Behind v1 plan: NB02–06 are still empty (the v1 target was "basic baselines with metrics by 26 Oct"). No git repo yet. Report/Presentation folders empty.
- Unblocked today: `utils.retrieve()` now works out of the box (TF-IDF fallback) and `llm.py` gives one shared `generate()` for everyone. Details below.

## The rule that prevents blocking: everyone talks through files, not through each other's notebooks
Nobody imports another member's notebook, and nobody waits for another member's model to be finished.

| Interface (owner) | What it is | Who consumes it | Available since |
|---|---|---|---|
| `utils.load_sample()`, `utils.load_split()` (A, frozen) | the 12k sample and the fixed split | everyone | 20 Oct |
| `utils.retrieve(query, k)` → DataFrame `[id, title, abstract, primary_category, author_list, score]` (C) | the retriever D's RAG uses | D (05, 06) | **works now** with a TF-IDF fallback. C creates `codes/retrieval.py` exposing the same `retrieve(query, k)` and it is picked up automatically. No other file changes needed. |
| `llm.generate(prompt, system=None, max_new_tokens, temperature)` (D) | the only way anyone calls an LLM; cached to `artifacts/llm_cache.jsonl`, so reruns are free and reproducible | A (few-shot extraction), B (LLM classifier), D | **works now**. `LLM_BACKEND=echo` lets you write a notebook with no model installed; `hf` (default, Qwen2.5-1.5B-Instruct) and `openai` are the real backends. |
| `utils.save_results("03", {...})` → `artifacts/03_results.json` | each notebook's headline metrics | 06 summary table, report tables | works now |
| `codes/artifacts/<name>.pkl` | fitted models (B's classifier, C's embeddings) | 06 loads them; nobody re-trains someone else's model | B and C write them when ready |
| `report/sections/0X_<name>.md` | each member's report text | D merges on 4 Nov | create on day 1 |

**Module ownership (no two people edit the same file):** `utils.py` frozen (A only, for bug fixes). `extraction.py` = A, `classify.py` = B, `retrieval.py` = C, `llm.py` = D. Notebooks can import these. If you need a helper, put it in your own module.

**06 is built from stand-ins, not from finished parts.** D writes `analyse(query)` this week against the TF-IDF `retrieve`, a trivial classifier placeholder (majority class or B's pickle once it exists) and `LLM_BACKEND=echo`, then swaps each real component in as its pickle/module appears. 06 is therefore never blocked and always runnable.

## Dependency map
```
01 (done) ──> 02 (A)  ──> artifacts/02_results.json ─┐
         ──> 03 (B)  ──> classify.py + classifier.pkl ─┤
         ──> 04 (C)  ──> retrieval.py (+ embeddings.pkl) ─┤──> 06 (D) ──> report tables & demo
         ──> 05 (D)  ──> prompts + 05_results.json ──────┘
llm.py (D, done) ──> used by 02, 03, 05 (all have the echo backend meanwhile)
```
Only soft dependencies remain: 06 gets *better* as pickles arrive, but it runs from day 1.

## Week 1–12 coverage (spec: "Content Coverage: Weeks 1–12"; checked against the lecture PDFs)
| Week | Lecture topic | Where it appears in our system | Owner |
|---|---|---|---|
| 1 | Unit intro, NLP pipeline | architecture diagram, report intro | C, A |
| 2 | Preprocessing (normalise, tokenise, stop-words, stem vs lemma) | `extraction.py` pipeline, NB02 | A |
| 3 | POS tagging, NER, chunking | POS key phrases, spaCy NER + regex, NB02 | A |
| 4 | BoW, TF-IDF, n-grams, Word2Vec/GloVe | TF-IDF keywords (NB03); TF-IDF → **averaged Word2Vec/GloVe** → MiniLM retrievers (NB04) | B, C |
| 5 | Classical ML (NB, LogReg, SVM), metrics | NB vs LogReg, macro-F1, confusion matrix, NB03 | B |
| 6–7 | Deep learning: RNN, BiLSTM/GRU, Transformer | **fine-tuned DistilBERT classifier (required, not optional)**; small BiLSTM if time | B |
| 8 | LLMs, prompting, CoT, RAG | prompt templates, CoT gap analysis, RAG synthesis, NB05 | D |
| 9 | Fine-tuning: feature extraction vs full vs PEFT/LoRA | frozen-DistilBERT+LogReg vs full fine-tune (NB03); few-shot extraction (NB02) | B, A |
| 10 | Hallucination, HHH alignment, safety | **groundedness check**: each summary claim vs retrieved abstracts, grounded/ungrounded rate as a table (NB05) — counts as the "LLM with ethics" advanced technique | D |
| 11 | Multimodal LLM, building RAG, agents | RAG chunk/retrieve/generate (NB04+05); multimodal skipped (optional in spec) | C, D |
| 12 | (not released yet) | one sentence in future work once we see the topic | D |

Changes from v1 to close the gaps: C adds a Word2Vec/GloVe retriever (W4); B's transformer fine-tune is now required, with a feature-extraction vs full-FT comparison (W6–7, W9); D's ethics note becomes a measured hallucination/groundedness table (W10).

## Roles (unchanged from v1; each person: 1–2 notebooks, ~1,000 report words, ~3 slides)
| Member | Notebooks | Basic | Advanced | Metric | Report section |
|---|---|---|---|---|---|
| A | 02 | preprocessing, POS key phrases, spaCy NER + regex | few-shot LLM extraction via `llm.generate` | P/R/F1 on ~50 hand-labelled abstracts | intro, dataset, preprocessing, extraction |
| B | 03 | TF-IDF keywords; NB vs LogReg (BoW vs TF-IDF) | DistilBERT fine-tune (feature-extraction vs full FT) **plus** zero/few-shot LLM classifier; optional ensemble | accuracy, macro-F1, confusion matrix | classification + comparison table |
| C | 04 | TF-IDF vs averaged Word2Vec/GloVe vs MiniLM similarity; KMeans clustering | `retrieval.py` RAG retriever | P@k, nDCG@k (same category = relevant), purity | retrieval, recommendation, architecture diagram |
| D | 05, 06 | extractive summary baseline (TextRank) | LLM summary + CoT gap analysis + RAG synthesis; groundedness/hallucination table | ROUGE + 1–5 rubric on ~20 samples | LLM methods, results, limitations, future work; edits whole report |

Coverage is 5 basic + 5 advanced (spec needs 3 + 3), so if one advanced item slips we still pass the bar. Priority order if time runs out: keep RAG, prompting/CoT and the foundation LLM; drop fine-tuning first, ensemble second.

## Timeline v2 (hard dates, each with a fallback)
| By | Deliverable | If late |
|---|---|---|
| **Mon 27 Oct** | team GitHub repo, everyone cloned, `pip install -r codes/requirements.txt` works, LLM backend decided (default `hf` Qwen2.5-1.5B) | use `LLM_BACKEND=echo` and keep going |
| **Wed 29 Oct** | A: preprocessing + regex/NER running; B: NB/LogReg metrics saved; C: TF-IDF + MiniLM retrieval, `retrieval.py` committed; D: `analyse(query)` skeleton in 06 runs end to end on stand-ins | 06 keeps using the TF-IDF fallback |
| **Sat 1 Nov** | advanced parts working: A few-shot, B transformer/LLM classifier, C eval table, D CoT + RAG synthesis; every notebook ends with `save_results`; **slide template (C) + 3 slides per person drafted** | drop fine-tuning → zero-shot LLM classifier |
| **Sun 2 Nov** | 06 uses all real components; one full rehearsal (15 min) | present with stand-ins for whichever part is missing and say so |
| **Mon 3 Nov** | report sections in `report/sections/`; D merges; architecture diagram | D writes the gap from notebook markdown |
| **Tue 4 Nov** | report v1 PDF ≤ 4,000 words; slides final PDF | |
| **Wed 5 Nov** | `jupyter nbconvert --execute` every notebook top to bottom on one machine; fix anything that breaks | |
| **Thu 6 Nov** | contribution forms (one per person), rename `GroupID`, zip, submit before 11:55 PM. Zip excludes `dataset/` (5.6 GB) and `TEAM_PLAN.md`, includes `codes/data/arxiv_cs_sample.jsonl.gz` (7 MB) and `codes/artifacts/*_results.json` | |

Daily: one 10-minute sync (text is fine) answering three things: done, next, blocked. A "blocked" that lasts more than a day means we change the interface, not wait.

## Git conventions (contribution evidence for the 5 individual marks)
- One branch per person (`a-extraction`, `b-classify`, `c-retrieval`, `d-llm`), merge to `main` via small PRs at least every 2 days. Commit history is the evidence attached to the contribution form.
- Clear notebook outputs before committing except in the final submission run (keeps diffs reviewable). Pickles and model weights are git-ignored; only `*_results.json` and `llm_cache.jsonl` are kept so results reproduce without rerunning the LLM.
- Never commit `dataset/`.

## Known pitfalls in the spec (unchanged)
- "Analyse mutual citations": arXiv metadata has no citation graph. Use similarity/co-author graph and say so in the report (or Semantic Scholar API if time).
- "NER for institutions": affiliations mostly missing; extract authors, methods, datasets instead.
- "Abstractive summarization with extractive methods" is contradictory: extractive = basic baseline, LLM abstractive = advanced. The spec says it may contain cues meant to mislead LLMs, so each justification should be written by the owner in 2–4 lines, not pasted.

## Files added today
- `codes/utils.py`: `retrieve()` TF-IDF fallback with auto-override from `retrieval.py`; `save_results()` / `load_results()`; `artifacts/` directory.
- `codes/llm.py`: shared `generate()` with hf / openai / echo backends and a JSONL cache.
- `codes/requirements.txt`, `.gitignore`.
