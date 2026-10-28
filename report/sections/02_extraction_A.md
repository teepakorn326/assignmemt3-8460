# 2. Extracting information (Member A)
Objective: Extract method names and dataset names from each abstract for display in the demo. Since arXiv does not provide ground-truth labels, the team manually annotated 20 abstracts from the test set (containing 90 names) and compared five different extraction methods on this data.

## 2.1 Basic techniques (weeks 2–3)
**Regex**: identifies acronyms (e.g., HiRM, GPT-3), their definitions, and patterns like “X dataset.”
**spaCy NER**: uses a model trained on news data; since it lacks specific labels for METHOD or DATASET, the ORG or PRODUCT labels are used as substitutes.
**POS key phrases (ADJ* NOUN+)**: extracts multi-word key phrases—such as “masked autoencoder”—at a rate of approximately 25 phrases per document, requiring no training data.

## 2.2 Advanced Techniques (Weeks 9–10)
**LLM few-shot**: Qwen2.5-1.5B (temperature 0) with 3 manually crafted examples, returning JSON output; compared against zero-shot (no examples).
**Hybrid**: the output from the few-shot LLM combined with additional names identified via regex that the LLM missed.
**Reliability check**: the proportion of returned names that actually appear in the abstract.

## 2.3 Result

| How to extract | Precision | Recall | F1 |
|---|---|---|---|
| Regex rules | 0.78 | 0.36 | 0.49 |
| spaCy NER (ORG/PRODUCT) | 0.64 | 0.30 | 0.41 |
| LLM zero-shot | **0.88** | 0.24 | 0.38 |
| LLM few-shot | 0.71 | 0.57 | 0.63 |
| Hybrid (few-shot + regex) | 0.70 | **0.67** | **0.68** |


*Table 2. Micro P/R/F1 across methods and datasets; 20 abstracts, 90 target labels, relaxed matching.*

| Prompt | F1 methods | F1 datasets | F1 tasks | Valid JSON | Name included in the abstract |
|---|---|---|---|---|---|
| Zero-shot | 0,44 | 0,10 | 0,44 | 100% | 96% |
| Few-shot | 0,61 | 0,70 | 0,30 | 100% | 89% |

*Table 4. LLM scores by type and reliability.*

## 2.4 Discussion
The regex is accurate but misses all 32 lowercase instances (“symbolic execution”) because it only searches for uppercase letters. The zero-shot approach is highly conservative, failing to identify almost any datasets (recall 0.05). Providing three examples does not help with formatting—as both prompts already yield 100% valid JSON—but it does clarify what the model should extract: dataset recall rises to 0.74, and the F1 score improves from 0.38 to 0.63. The trade-off is that 11% of the names identified via the few-shot approach do not appear verbatim in the abstracts, typically because the extracted phrases are too long (e.g., “causal tracing studies” instead of “causal tracing”). The hybrid method achieves the highest F1 score (0.68) with virtually no loss in precision, making it the preferred choice for the system.
**Limitations**: the study is based on only 20 abstracts annotated by a single person; it employs loose matching criteria; and it utilizes a small, 1.5B-parameter model.
**Future improvements**: expand the dataset to 50+ papers with dual-annotator verification; enforce strict verbatim matching against the source text; and experiment with larger models or SciBERT.