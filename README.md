# COMP8460 Assignment 3: Research Paper Analysis and Recommendation System

Use Case 3. Dataset: arXiv metadata snapshot (Kaggle, `arxiv-metadata-oai-snapshot.json`).

## Structure

```
GroupID_Assignment3/
|-- codes/                          notebooks (+ data/ cached sample)
|-- report/GroupID_Report.pdf       max 4000 words
|-- Presentation/Presentation.pdf
|-- individual contribution form/GroupID_StudentID_Contribution.pdf
```

Rename `GroupID` (folder and file names) to your real group ID before zipping.

## Running

1. Create a Python 3.12 or 3.13 environment and install `codes/requirements.txt`. The Hugging Face package versions are pinned because an incompatible `datasets`/`huggingface-hub` pair prevented retrieval from importing.
2. Install the spaCy models: `python -m spacy download en_core_web_sm` and `python -m spacy download en_core_web_md`.
3. Run notebooks from inside `codes/`, in order 02, 03, 04, 05, `05_1_rag_citations_and_guardrails.ipynb` (revised RAG prompt and injection defences; notebook 05 is kept as the first version), 06, then `06_1_rag_model_comparison.ipynb` (RAG generator size comparison; downloads Qwen2.5-3B-Instruct on the first uncached run). The 12,000-paper sample and fixed split are included under `codes/data/`, so the raw snapshot is not needed for these notebooks. Pretrained Hugging Face model weights are downloaded on the first uncached run.
4. To regenerate the sample and split with notebook 01, put the raw snapshot at `../dataset/arxiv-metadata-oai-snapshot.json` (relative to this folder), or set `ARXIV_PATH`.

Do not put the 5.6 GB raw file in the submission zip.
# assignmemt3-8460
