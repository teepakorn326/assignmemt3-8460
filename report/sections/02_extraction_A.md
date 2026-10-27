# 2. Information extraction (Member A)

<!-- drafted with an AI assistant from NB02 code and saved results; owner reviews, rewords and fills ▢ after the final run -->

Goal: turn each abstract into structured metadata (key phrases, methods, datasets, tasks) that the demo shows and that the LLM components can use. Since arXiv has no gold entities, we hand-labelled 20 held-out test abstracts (guideline in NB02) and compare five extractors on them.

## 2.1 Basic techniques (Week 2–3)
- **POS key phrases.** Token runs matching `ADJ* NOUN+` (e.g. "masked autoencoder") plus spaCy `noun_chunks`. No training data is needed and multi-word terms survive, which single-word TF-IDF loses. On 600 test papers this gives 25.3 key phrases and 27.3 noun chunks per paper.
- **spaCy NER.** The pretrained news model has no METHOD or DATASET label; on 600 papers it labels names mostly as ORG (2,020 of 4,142 entities), so we use ORG/PRODUCT/WORK_OF_ART as its "artefact" guess.
- **Rule-based extraction (regex).** Mixed-case acronyms (`HiRM`, `GPT-3`), acronym definitions ("Dynamic Task Coordination (DTC)"), "X dataset/benchmark", years and percentages, with a stop-list of generic acronyms (NLP, CNN, LLM). Only 54% of regex names are also spaCy entities, so the two basic extractors are complementary.

## 2.2 Advanced techniques (Weeks 9–10)
- **Few-shot LLM extraction.** An instruction, three invented worked examples (so no test leakage) and the abstract; the model (Qwen2.5-1.5B-Instruct, temperature 0, cached for reproducibility) returns JSON `{methods, datasets, tasks}`. A **zero-shot** prompt with the same instruction is the ablation.
- **Hybrid ensemble.** Few-shot LLM output plus any regex name the LLM missed.
- **Reliability check (W10).** Valid-JSON rate, and the share of returned names that occur in the abstract (names that do not were invented or paraphrased).

## 2.3 Results

| Extractor | Precision | Recall | F1 |
|---|---|---|---|
| Regex rules | 0.78 | 0.36 | 0.49 |
| spaCy NER (ORG/PRODUCT) | ▢ | ▢ | ▢ |
| Zero-shot LLM | ▢ | ▢ | ▢ |
| Few-shot LLM | ▢ | ▢ | ▢ |
| Hybrid (few-shot + regex) | ▢ | ▢ | ▢ |

*Table 3. Micro P/R/F1 over methods + datasets, 20 abstracts, 90 gold names, relaxed matching.*

| Prompt | Valid JSON | Names found in abstract | Tasks F1 |
|---|---|---|---|
| Zero-shot | ▢ | ▢ | ▢ |
| Few-shot | ▢ | ▢ | ▢ |

*Table 4. LLM reliability.*

## 2.4 Discussion
Rules are precise but miss every lower-case method ("symbolic execution", "itemset mining"): ▢ of ▢ lower-case gold names are missed by regex vs ▢ by the few-shot LLM. The worked examples mainly fix the output format and the method/dataset boundary (▢ → ▢ valid JSON). The LLM's errors are generic terms ("LLMs") and over-long spans, and ▢% of its names do not occur in the abstract, so its output should be filtered by the grounding check before display. The hybrid combines the precise acronyms of the rules with the LLM's coverage and is the extractor we recommend for the system.

**Limitations.** 20 abstracts and one annotator (labels drafted with AI help, then checked), so small differences are within noise; substring matching is lenient; a small 1.5B model was used for cost; author/institution NER is limited by the missing affiliations. **Improvements:** 50+ abstracts with two annotators and Cohen's κ; keep only LLM spans that appear in the abstract; a scientific NER model (e.g. SciBERT trained on SciERC) as a stronger basic baseline.
