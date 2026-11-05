# 3. Research-domain classification and keywords (Member B)
Objective: assign each paper one of the six primary categories and extract its keywords, so the demo can tell a user which field a query or paper belongs to. Labels are the arXiv `primary_category`, and every model is scored on the shared 2,400-paper test split (400 per class) with accuracy and macro-F1.

## 3.1 Techniques
**TF-IDF keywords** (week 4): a unigram + bigram vectoriser fitted on the training abstracts after A's `clean_text`. A domain's keywords are its highest mean-weight terms (cs.CR: security, attacks, privacy; cs.IR: recommendation, user, retrieval), while cs.AI and cs.LG share generic terms such as "model" and "learning", an early sign of later confusion.
**Naive Bayes and Logistic Regression** (week 5): MultinomialNB on BoW and TF-IDF, then LogReg on TF-IDF unigrams and unigrams + bigrams, changing one thing at each step.
**LSTM** (week 6): DistilBERT tokenizer, 256 tokens, mean-pooled over non-padding tokens; the practical's last hidden state scored 0.21 because most of what it read was padding.
**Zero-shot NLI** (week 7): DeBERTa-v3 NLI with "This paper is about {label}." on 500 test papers, using either the official category names or one-line descriptions.
**DistilBERT** (week 9): feature extraction (frozen [CLS] + LogReg) and full fine-tuning with `Trainer` on 3,000 papers, with a LogReg on the same papers as a control; a follow-up chose epochs on a validation split and trained on all 9,600 papers with three seeds.
**Few-shot LLM and ensembles**: Qwen2.5-1.5B with one training example per category (200 test papers), and soft voting of LogReg with NB or with fine-tuned DistilBERT.

## 3.2 Results

| Model | Train papers | Test papers | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Naive Bayes + BoW | 9,600 | 2,400 | 0.773 | 0.773 |
| Naive Bayes + TF-IDF | 9,600 | 2,400 | 0.770 | 0.769 |
| LogReg + TF-IDF | 9,600 | 2,400 | 0.787 | 0.788 |
| **LogReg + TF-IDF (1–2 grams)** | 9,600 | 2,400 | **0.792** | **0.792** |
| DistilBERT fine-tune, 2 epochs, 3 seeds | 9,600 | 2,400 | 0.798 | 0.798 |
| Ensemble: LogReg + DistilBERT, 3 seeds | 9,600 | 2,400 | 0.806 | 0.805 |
| LSTM, 5 epochs, 3 seeds | 9,600 | 2,400 | 0.711 | 0.712 |
| LogReg + TF-IDF (1–2 grams) | 3,000 | 2,400 | 0.777 | 0.777 |
| DistilBERT fine-tune, 1 epoch | 3,000 | 2,400 | 0.758 | 0.753 |
| DistilBERT frozen + LogReg | 3,000 | 2,400 | 0.717 | 0.717 |
| Zero-shot NLI, names / descriptions | 0 | 500 | 0.366 / 0.616 | 0.356 / 0.628 |
| LogReg, same 500 papers | 9,600 | 500 | 0.814 | 0.813 |
| Few-shot LLM, names / descriptions | 6 in prompt | 200 | 0.415 / 0.485 | 0.388 / 0.469 |
| LogReg, same 200 papers | 9,600 | 200 | 0.825 | 0.827 |
| LogReg, titles only | 9,600 | 2,400 | 0.690 | 0.698 |

*Table 3. Domain classification on the shared test split; subset rows are paired with LogReg on the same papers. Bold: the model the system uses.*

## 3.3 Discussion
The system uses TF-IDF bigrams with Logistic Regression (`classify.predict`, saved as `classifier.pkl`). LogReg beats Naive Bayes by about two points and bigrams add half a point; five-fold cross-validation on the training papers picked the same setting (0.799), so the choice does not rest on test scores.

A pre-trained transformer did not beat word counts. With 3,000 papers, the fine-tune (0.753) and frozen encoder (0.717) trailed LogReg on the same data (0.777). Trained on all 9,600 papers, DistilBERT reached 0.798 over three seeds, level with LogReg (no seed differed clearly in a paired test). Averaging the two gave the highest score, 0.805, a clear gain on two of three seeds (McNemar p = 0.004, 0.020, 0.198), so the models make different mistakes. We keep LogReg: about one point does not justify a GPU and a second model in the demo. The task is decided mostly by topic vocabulary, which TF-IDF measures directly. The LSTM improved with more epochs (0.712) but stays well below, since it learns its embeddings from only 9,600 abstracts.

For zero-shot, label wording mattered more than the model: with bare names, 298 of 500 answers were cs.LG because "Machine Learning" describes almost every abstract, while descriptions raised accuracy from 0.366 to 0.616. The 1.5B LLM gained less and stays far below LogReg on the same papers; its advantage is needing no labelled data.

Part of the error is in the labels. The main confusions are cs.AI ↔ cs.LG and cs.AI ↔ cs.CL; cross-listed papers (1,332 of 2,400) score 0.763 against 0.828, and counting any listed category as correct gives 0.875. On titles alone, macro-F1 falls to 0.698, so the label for a short typed query is only a rough guide.

**Limitations**: zero-shot and few-shot ran on 500 and 200 papers; only one small LLM was tried; the label descriptions were chosen after seeing test scores; each paper gets one label; and differences below 0.01 are within seed and GPU variation.
**Future improvements**: multi-label classification over `all_categories`, a domain-adapted encoder such as SciBERT, and labelled real queries for the demo's short input.
