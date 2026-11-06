# 2. Extracting information (Member A)
Objective: Extract method names and dataset names from each abstract, so a user can see what a paper uses without reading it. Since arXiv does not provide ground-truth labels, we built a set of 20 labelled test abstracts (75 method/dataset names). The labels were drafted with an AI assistant (not the Qwen model being evaluated), checked for verbatim abstract spans, and reviewed by one team member, who removed 9 generic or background-only names (e.g. "prompting", "LLM", "CIFAR"). Five extraction methods were compared on this data.

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
| Regex rules | 0.66 | 0.36 | 0.47 |
| spaCy NER (ORG/PRODUCT) | 0.52 | 0.29 | 0.38 |
| LLM zero-shot | **0.88** | 0.29 | 0.44 |
| LLM few-shot | 0.61 | 0.59 | 0.60 |
| Hybrid (few-shot + regex) | 0.59 | **0.68** | **0.63** |


*Table 2. Micro P/R/F1 across methods and datasets; 20 abstracts, 75 reviewed target labels, relaxed matching.*

| Prompt | F1 methods | F1 datasets | F1 tasks | Valid JSON | Name included in the abstract |
|---|---|---|---|---|---|
| Zero-shot | 0.52 | 0.11 | 0.44 | 100% | 96% |
| Few-shot | 0.59 | 0.62 | 0.30 | 100% | 89% |

*Table 4. LLM scores by type and reliability.*

## 2.4 Discussion
The regex misses 25 lower-case gold names (including “symbolic execution”) because it searches for capital letters. The zero-shot approach is highly conservative, failing to identify almost any datasets (recall 0.06). Providing three examples does not help with formatting—as both prompts already yield 100% valid JSON—but it clarifies what the model should extract: dataset recall rises to 0.67, and the F1 score improves from 0.44 to 0.60. The trade-off is that 11% of the names identified via the few-shot approach do not appear verbatim in the abstracts, typically because the extracted phrases are too long. The hybrid method has the highest measured F1 (0.63), with lower precision than few-shot alone, and is the extractor used by the demo; however, the paired bootstrap interval of its gain over few-shot (−0.005 to +0.075) includes zero, so the gain is not established.
**Limitations**: the study is based on only 20 abstracts with AI-drafted labels reviewed by a single team member (no inter-annotator agreement); the same 20 abstracts were used while designing the hybrid and the acronym stop-list, so there is no development set; it employs loose matching criteria (exact-match F1 is lower, see NB02); and it utilizes a small, 1.5B-parameter model.
**Future improvements**: expand the dataset to 50+ papers with dual-annotator verification; enforce strict verbatim matching against the source text; and experiment with larger models or SciBERT.
