# 3. Research-domain classification and keywords (Member B)
Objective: assign each paper one of the six primary categories and extract its keywords, so the demo can tell a user which field a query or paper belongs to. Labels are the arXiv `primary_category`. Every model is scored on the shared 2,400-paper test split (400 per class) with accuracy and macro-F1.

## 3.1 Basic techniques (weeks 4–6)
**TF-IDF keywords**: a unigram + bigram TF-IDF vectoriser fitted on the training abstracts after A's `clean_text`. A domain's keywords are the terms with the highest mean weight over its papers; a paper's keywords are its own top-weighted terms. The domain lists look right (cs.CR: security, attacks, privacy; cs.IR: recommendation, user, retrieval), but cs.AI and cs.LG share generic terms such as "model" and "learning", an early sign of the confusion reported below.
**Naive Bayes and Logistic Regression**: MultinomialNB on BoW counts and on TF-IDF, then LogReg on TF-IDF unigrams and on unigrams + bigrams, changing one thing at each step.
**LSTM**: the Week 6 LSTM with the DistilBERT tokenizer, 256-token inputs, 2 epochs. Outputs are mean-pooled over non-padding tokens. With the practical's last hidden state the model scored 0.21, close to chance, because most of what it had read was padding.

## 3.2 Advanced techniques (weeks 7–9)
**Zero-shot NLI** (week 7): the DeBERTa-v3 NLI model from the practical, hypothesis "This paper is about {label}.", on 500 test papers. Two label sets: the official arXiv names, and one-line descriptions condensed from the arXiv taxonomy.
**DistilBERT** (week 9): both adaptation types from the lecture, on the same class-balanced 3,000 training papers: feature extraction (frozen [CLS] vector + LogReg) and full fine-tuning with the Hugging Face `Trainer` (1 epoch, learning rate 5e-5, 55 s on an RTX 4060). A TF-IDF + LogReg trained on the same 3,000 papers controls for data size.
**Few-shot LLM** (week 9 prompting): Qwen2.5-1.5B-Instruct through the team's `llm.generate`, one training example per category, temperature 0, 200 test papers, the same two label sets. Replies are parsed for a category code; none were unparseable.
**Ensemble**: soft voting over class probabilities, for LogReg + NB and for LogReg + fine-tuned DistilBERT.

## 3.3 Results

| Model | Train papers | Test papers | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Naive Bayes + BoW | 9,600 | 2,400 | 0.773 | 0.773 |
| Naive Bayes + TF-IDF | 9,600 | 2,400 | 0.770 | 0.769 |
| LogReg + TF-IDF | 9,600 | 2,400 | 0.787 | 0.788 |
| **LogReg + TF-IDF (1–2 grams)** | 9,600 | 2,400 | **0.792** | **0.792** |
| Ensemble: LogReg + NB | 9,600 | 2,400 | 0.792 | 0.791 |
| Ensemble: LogReg + DistilBERT | 9,600 + 3,000 | 2,400 | 0.788 | 0.784 |
| LSTM, 2 epochs | 9,600 | 2,400 | 0.669 | 0.671 |
| LogReg + TF-IDF (1–2 grams) | 3,000 | 2,400 | 0.777 | 0.777 |
| DistilBERT, full fine-tune, 1 epoch | 3,000 | 2,400 | 0.758 | 0.753 |
| DistilBERT, frozen + LogReg | 3,000 | 2,400 | 0.717 | 0.717 |
| Zero-shot NLI, category names | 0 | 500 | 0.366 | 0.356 |
| Zero-shot NLI, category descriptions | 0 | 500 | 0.616 | 0.628 |
| LogReg (1–2 grams), same 500 | 9,600 | 500 | 0.814 | 0.813 |
| Few-shot LLM, category names | 6 in prompt | 200 | 0.415 | 0.388 |
| Few-shot LLM, category descriptions | 6 in prompt | 200 | 0.485 | 0.469 |
| LogReg (1–2 grams), same 200 | 9,600 | 200 | 0.825 | 0.827 |

*Table 3. Research-domain classification on the shared test split. Rows on a subset are paired with LogReg scored on the same papers.*

## 3.4 Discussion
TF-IDF bigrams with Logistic Regression is the best model and the one the system uses (`classify.predict`, saved as `classifier.pkl`). LogReg beats Naive Bayes by about two points, and bigrams add half a point more. Its train–test gap (0.94 vs 0.79) shows some over-fitting, but nothing else generalised better.

The DistilBERT rows answer whether a pre-trained transformer beats word counts when both see the same data. It does not, at least after one epoch: on the same 3,000 papers LogReg reaches 0.777, the full fine-tune 0.758 and the frozen encoder 0.717. Full fine-tuning beats feature extraction, as the Week 9 lecture leads us to expect, but this task is decided mostly by topic vocabulary ("attack", "image", "recommendation"), and that is exactly what TF-IDF measures. Averaging LogReg with DistilBERT gave 0.788, just under LogReg alone: simple averaging gives the weaker model equal weight.

For zero-shot, the wording of the labels mattered more than the model. With the bare names, 298 of 500 zero-shot answers were cs.LG, since "Machine Learning" describes nearly every abstract, and cs.CV was chosen only 10 times. One-line descriptions raised zero-shot from 0.366 to 0.616 and few-shot from 0.415 to 0.485. The 1.5B LLM gained less and still labelled 70 of 200 papers cs.LG. Both stay well below LogReg on the same papers (0.814 and 0.825). Their advantage is that they need no labelled data, which would matter for a new category with no training papers.

Part of the remaining error comes from the labels rather than the models. The most frequent confusions are cs.AI ↔ cs.LG (56 and 44 papers) and cs.AI ↔ cs.CL (47 and 37). Cross-listed papers (1,332 of the 2,400) score 0.763 against 0.828 for single-category ones, and if any of a paper's listed categories counts as correct, LogReg reaches 0.875. One misclassified example, "Evaluating the Hidden Costs of Personalization in Large Language Models", is listed only under cs.AI, but the prediction cs.CL is defensible.

**Limitations**: zero-shot and few-shot ran on subsets of 500 and 200 papers; DistilBERT had one epoch on 3,000 papers and the LSTM two epochs; only one small LLM was tried; the label descriptions were written after the names failed on the same subset, so 0.616 is slightly optimistic; each paper gets a single label; and GPU runs are not bit-for-bit repeatable (over three runs the LSTM ranged 0.661–0.669 and the fine-tune 0.758–0.760), so differences under 0.01 mean little.
**Future improvements**: a multi-label classifier over `all_categories`; a domain-adapted encoder such as SciBERT, fine-tuned for several epochs on the full training set; a held-out validation split for choosing label wording; and a larger LLM for the few-shot classifier.
