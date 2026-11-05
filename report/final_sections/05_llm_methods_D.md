# 5. LLM summarisation, research gaps and RAG (Member D)

## 5.1 Setup
All generation uses one foundation model, Qwen2.5-1.5B-Instruct, run locally through `llm.generate` at temperature 0. Every call is cached under a hash of the model, prompt and settings, so each notebook reproduces the same text. The prompts, the answer parser and the groundedness judge live in `rag.py`, so the demo (NB06) runs exactly what NB05 evaluated. Baselines are an extractive summary (the three highest-scoring TF-IDF sentences) and two pre-trained summarisers, BART-large-CNN and T5-base.

## 5.2 Single-paper summaries
We built four prompts in steps: basic, role, instruction and few-shot. Instructions and two worked examples controlled the output: mean length fell from 117.6 words (basic) to 50.5 (few-shot), against 163.2 for the abstract, and no summary was cut off.

| System | ROUGE-L | Rubric 1–5 | Supported sentences |
|---|---|---|---|
| Extractive | 0.308 | 3.72 | 59/59 |
| BART | **0.371** | 3.55 | 49/50 |
| T5 | 0.288 | 3.43 | 54/55 |
| LLM basic | 0.171 | **3.73** | – |
| LLM few-shot | 0.259 | 3.70 | 51/51 |

*Table 5. Summaries of 20 test papers.*

arXiv has no reference summaries, so ROUGE uses the title plus the first sentence of the abstract, which rewards copying the opening; this is why BART leads. The LLM judge scores coverage, faithfulness and readability with the abstract in view, but its top three systems are within 0.03, so it cannot separate them. Because the judge is the same model, we tested it first: it accepted 40/40 sentences copied from abstracts and rejected 23/23 planted false claims (another paper's claim, or a changed number). We keep the few-shot prompt for its length and format control, not for a proven quality gain.

## 5.3 Research gaps: chain of thought
Identifying a gap looks like a reasoning task, so we compared a direct prompt with zero-shot and few-shot chain of thought (CoT) on 10 papers.

| Prompt | Reasoned first | Specificity | Judge 1–5 |
|---|---|---|---|
| Direct (no CoT) | n/a | **0.37** | **3.3** |
| Zero-shot CoT | 0/10 | 0.38 | 3.3 |
| Few-shot CoT | 10/10 | 0.17 | 3.0 |

*Table 6. Specificity = share of the gap's content words found in the abstract.*

CoT did not help the 1.5B model. Told to think step by step, it never reasoned before answering; given a worked example, it reasoned every time but copied the example's gap pattern, so its gaps became generic. The system therefore uses the direct prompt inside RAG.

## 5.4 Retrieval-augmented synthesis
For a research question, the five MiniLM-retrieved abstracts are numbered in the prompt and the model writes a synthesis, one gap and three recommendations. The revised prompt (v2, NB05.1) also states the paper count and asks for a citation at the end of every sentence.

| Mode (5 questions) | Judge-supported | Cited sentences | Kept by filter |
|---|---|---|---|
| No retrieval | 8/20 (40%) | 0/20 | 0/20 |
| RAG v1 | 13/17 (76%) | 5/17 | 5/17 |
| RAG v2 | **18/21 (86%)** | 10/21 | 10/21 |

*Table 7. Synthesis sentences judged against the retrieved abstracts.*

Retrieval is the clearest advanced-technique gain: support doubled when the evidence was in the prompt. The per-claim filter (`rag.filter_cited_claims`) keeps a sentence only if it cites a retrieved paper, does not misstate the paper count, and the judge finds it supported by the paper it cites. Even with v2, the 1.5B model ignored the citation instruction for about half of its sentences. In NB06.1 only the generator changed: Qwen2.5-3B cited 26/29 sentences against 11/29, the filter kept 23/29 against 10/29, and no answer needed the fallback (0/8 against 5/8). Judge support was similar (24/29 vs 23/29), so the larger model is not clearly more factual; it makes its claims checkable.

## 5.5 Safety: prompt injection (Week 10)
We hid three instructions in three abstracts (nine attempts per setting) and changed only the defence.

| Defence | Known attacks | Held-out attacks |
|---|---|---|
| None | 2/9 | 4/9 |
| Guard sentence in the system prompt | 3/9 | 4/9 |
| Sanitiser + guard | **0/9** | 4/9 |

*Table 8. Attacks that made the model write the injected text.*

The guard sentence gave no measurable protection. The sanitiser, which deletes instruction-like sentences before prompting, stopped the known attacks and changed none of the 12,000 sample abstracts, but it was written from those attacks and missed every held-out one. Pattern filters only block phrasings they know, so injection remains an open risk.
