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

1. Put the raw snapshot at `../dataset/arxiv-metadata-oai-snapshot.json` (relative to this folder), or set `ARXIV_PATH`.
2. Run `codes/01_data_loading_and_sampling.ipynb` from inside `codes/`. It writes `codes/data/arxiv_cs_sample.jsonl.gz` (12,000 papers, seed 42). Later notebooks read only this file.

Do not put the 5.6 GB raw file in the submission zip.
# assignmemt3-8460
