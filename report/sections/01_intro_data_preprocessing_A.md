# 1. Introduction, data and preprocessing (Member A)

## 1.1 Problem
Researchers face thousands of new arXiv papers a week. Our system takes a free-text query or a paper and returns its domain, its key methods and datasets, similar papers, a summary and possible research gaps. This section covers the data and the shared preprocessing that every later component (classification, retrieval, LLM) consumes.

## 1.2 Dataset
We use the arXiv metadata snapshot (Kaggle, ~3.2M records, 5.6 GB). NB01 streams it once and draws a class-balanced sample of **12,000 Computer Science papers, 2,000 per primary category** (cs.AI, cs.CL, cs.CR, cs.CV, cs.IR, cs.LG). Balancing removes the dominance of cs.LG/cs.CV so macro-F1 is meaningful, and 12k papers keeps every notebook runnable on a CPU. A single stratified 80/20 split by paper ID (9,600 / 2,400, seed 42) is shared by all notebooks, so no component is tuned on another's test data. The input text is `title + ". " + abstract`; the metadata has no affiliations, so author/institution extraction is limited to the `author_list` field.

| Category | Mean words | Vocabulary | Type–token ratio |
|---|---|---|---|
| cs.AI | 178.8 | 15,258 | 0.043 |
| cs.CL | 168.6 | 13,824 | 0.041 |
| cs.CR | 181.8 | 15,245 | 0.042 |
| cs.CV | 187.2 | 14,714 | 0.039 |
| cs.IR | 184.8 | 13,308 | 0.036 |
| cs.LG | 182.6 | 14,251 | 0.039 |

*Table 1. Week 1 corpus statistics (2,000 papers each).* Lengths are similar across domains (169–187 words), so length cannot separate them, but vocabularies differ, which the per-domain word clouds in NB02 make visible (Figure ▢). This motivated lexical features (TF-IDF) as the classification and retrieval baselines.

## 1.3 Preprocessing pipeline (Week 2)
`extraction.preprocess(text)` is the one function shared by the team: (1) remove LaTeX maths (`$...$`, ~7% of abstracts), LaTeX commands and URLs; (2) tokenise with spaCy `en_core_web_sm`; (3) lowercase and drop stop-words, punctuation and non-alphabetic tokens; (4) lemmatise.

| Step (1,000 training abstracts) | Vocabulary size |
|---|---|
| Raw lowercase words | 11,417 |
| Cleaned, no stop-words/punctuation/numbers | 10,818 |
| + Porter stemming | 6,483 |
| + spaCy lemmatisation | 8,466 |

*Table 2. Vocabulary reduction.* Stemming shrinks the vocabulary more (−40% vs −22%), but produces non-words (`analys`, `propos`). We chose **lemmatisation** because tokens are shown to users (keyword tables, key phrases) and inserted into LLM prompts, where real words matter more than a smaller vocabulary. Limitation: spaCy's small model mis-tags some technical terms (verbs tagged as nouns), which later creates noisy key phrases.
