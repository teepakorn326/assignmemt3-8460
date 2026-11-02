# 3. Research-domain classification and keywords (Member B)
Objective: assign each paper one of the six primary categories and extract its keywords, so the demo can tell a user which field a query or paper belongs to. Labels are the arXiv `primary_category`. Every model is scored on the shared 2,400-paper test split (400 per class) with accuracy and macro-F1.

## 3.1 Basic techniques (weeks 4–6)
**TF-IDF keywords**: a unigram + bigram TF-IDF vectoriser fitted on the training abstracts after A's `clean_text`. A domain's keywords are the terms with the highest mean weight over its papers; a paper's keywords are its own top-weighted terms. The domain lists look right (cs.CR: security, attacks, privacy; cs.IR: recommendation, user, retrieval), but cs.AI and cs.LG share generic terms such as "model" and "learning", an early sign of the confusion reported below.
**Naive Bayes and Logistic Regression**: MultinomialNB on BoW counts and on TF-IDF, then LogReg on TF-IDF unigrams and on unigrams + bigrams, changing one thing at each step. Five-fold cross-validation on the training papers then confirmed the setting (section 3.4).
**LSTM**: the Week 6 LSTM with the DistilBERT tokenizer, 256-token inputs, 2 epochs. Outputs are mean-pooled over non-padding tokens. With the practical's last hidden state the model scored 0.21, close to chance, because most of what it had read was padding. A follow-up run chose the number of epochs on a validation split (section 3.4).

## 3.2 Advanced techniques (weeks 7–9)
**Zero-shot NLI** (week 7): the DeBERTa-v3 NLI model from the practical, hypothesis "This paper is about {label}.", on 500 test papers. Two label sets: the official arXiv names, and one-line descriptions condensed from the arXiv taxonomy.
**DistilBERT** (week 9): both adaptation types from the lecture, on the same class-balanced 3,000 training papers: feature extraction (frozen [CLS] vector + LogReg) and full fine-tuning with the Hugging Face `Trainer` (1 epoch, learning rate 5e-5, 55 s on an RTX 4060). A TF-IDF + LogReg trained on the same 3,000 papers controls for data size. A follow-up run tested more epochs and all 9,600 papers, choosing the epoch on 960 validation papers and retraining with three seeds.
**Few-shot LLM** (week 9 prompting): Qwen2.5-1.5B-Instruct through the team's `llm.generate`, one training example per category, temperature 0, 200 test papers, the same two label sets. Replies are parsed for a category code; none were unparseable.
**Ensemble**: soft voting over class probabilities, for LogReg + NB and for LogReg + fine-tuned DistilBERT, the latter with both the 3,000-paper and the 9,600-paper DistilBERT.

## 3.3 Results

| Model | Train papers | Test papers | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Naive Bayes + BoW | 9,600 | 2,400 | 0.773 | 0.773 |
| Naive Bayes + TF-IDF | 9,600 | 2,400 | 0.770 | 0.769 |
| LogReg + TF-IDF | 9,600 | 2,400 | 0.787 | 0.788 |
| **LogReg + TF-IDF (1–2 grams)** | 9,600 | 2,400 | **0.792** | **0.792** |
| Ensemble: LogReg + NB | 9,600 | 2,400 | 0.792 | 0.791 |
| Ensemble: LogReg + DistilBERT, mean of 3 seeds | 9,600 | 2,400 | 0.806 | 0.805 |
| DistilBERT, full fine-tune, 2 epochs, mean of 3 seeds | 9,600 | 2,400 | 0.798 | 0.798 |
| Ensemble: LogReg + DistilBERT | 9,600 + 3,000 | 2,400 | 0.788 | 0.784 |
| LSTM, 5 epochs, mean of 3 seeds | 9,600 | 2,400 | 0.711 | 0.712 |
| LSTM, 2 epochs | 9,600 | 2,400 | 0.669 | 0.671 |
| LogReg + TF-IDF (1–2 grams) | 3,000 | 2,400 | 0.777 | 0.777 |
| DistilBERT, full fine-tune, 2 epochs, mean of 3 seeds | 3,000 | 2,400 | 0.776 | 0.774 |
| DistilBERT, full fine-tune, 1 epoch | 3,000 | 2,400 | 0.758 | 0.753 |
| DistilBERT, frozen + LogReg | 3,000 | 2,400 | 0.717 | 0.717 |
| Zero-shot NLI, category names | 0 | 500 | 0.366 | 0.356 |
| Zero-shot NLI, category descriptions | 0 | 500 | 0.616 | 0.628 |
| LogReg (1–2 grams), same 500 | 9,600 | 500 | 0.814 | 0.813 |
| Few-shot LLM, category names | 6 in prompt | 200 | 0.415 | 0.388 |
| Few-shot LLM, category descriptions | 6 in prompt | 200 | 0.485 | 0.469 |
| LogReg (1–2 grams), same 200 | 9,600 | 200 | 0.825 | 0.827 |
| LogReg (1–2 grams), titles only | 9,600 | 2,400 | 0.690 | 0.698 |

*Table 3. Research-domain classification on the shared test split. Bold marks the model the system uses. Rows on a subset are paired with LogReg scored on the same papers.*

## 3.4 Discussion
The system uses TF-IDF bigrams with Logistic Regression (`classify.predict`, saved as `classifier.pkl`). LogReg beats Naive Bayes by about two points, and bigrams add half a point more. We first picked the bigram setting on test scores, so we repeated the choice with five-fold cross-validation on the training papers, across unigrams to trigrams and four values of C. It picked the same setting (0.799). The train–test gap (0.94 vs 0.79) shows some over-fitting.

The DistilBERT rows ask whether a pre-trained transformer beats word counts. The planned run, one epoch on 3,000 papers, did not: LogReg reached a macro-F1 of 0.777 on those papers, the full fine-tune 0.753 and the frozen encoder 0.717. Full fine-tuning beat feature extraction, as the Week 9 lecture leads us to expect. One epoch was a scope limit rather than a tuned value, so we tested more training and more data. Validation chose 2 of up to 4 epochs in both cases. On the same 3,000 papers the fine-tune reaches 0.774, level with LogReg. On all 9,600 it reaches 0.798 over three seeds (0.794–0.802) against LogReg's 0.792, and no seed differs clearly from LogReg in a paired test. The transformer catches word counts but does not pass them, probably because this task is decided mostly by topic vocabulary ("attack", "image", "recommendation"), which is what TF-IDF measures.

The ensemble rows suggest the two models make different mistakes. Averaging LogReg with the 3,000-paper DistilBERT gave 0.784, under LogReg alone, because the weaker model had an equal vote. With the 9,600-paper DistilBERT it gave 0.805 over three seeds (0.799–0.810), the highest score in Table 3. The gain over LogReg is clear on two seeds of three (McNemar p = 0.004, 0.020 and 0.198). We still keep LogReg: about 1.3 points does not justify a GPU for training and a second model in the demo.

The LSTM row stops at two epochs because that was the limit in our team plan, not a tuned value, and its loss was still falling. We therefore tested longer training. A validation split of 10% of the training papers favoured epochs 4 to 6, and retraining on all 9,600 papers for 5 epochs gave a mean macro-F1 of 0.712 over three seeds (0.697–0.722). Training on to epoch 20 took train accuracy to 0.98 with test macro-F1 still at 0.704, so the extra epochs only over-fit. More epochs helped, but the LSTM stays 0.08 below LogReg and behind the fine-tuned DistilBERT, which saw a third of the data. It has about 4 million weights to learn from 9,600 abstracts, most of them in embeddings that start from nothing, while TF-IDF measures topic words directly.

For zero-shot, the wording of the labels mattered more than the model. With the bare names, 298 of 500 zero-shot answers were cs.LG, since "Machine Learning" describes nearly every abstract, and cs.CV was chosen only 10 times. One-line descriptions raised zero-shot accuracy from 0.366 to 0.616 and few-shot accuracy from 0.415 to 0.485. The 1.5B LLM gained less and still labelled 70 of 200 papers cs.LG. Both stay well below LogReg's accuracy on the same papers (0.814 and 0.825). Their advantage is that they need no labelled data, which would matter for a new category with no training papers.

Part of the remaining error comes from the labels rather than the models. The most frequent confusions are cs.AI ↔ cs.LG (56 and 44 papers) and cs.AI ↔ cs.CL (47 and 37). Cross-listed papers (1,332 of the 2,400) score 0.763 against 0.828 for single-category ones, and if any of a paper's listed categories counts as correct, LogReg reaches 0.875. One misclassified example, "Evaluating the Hidden Costs of Personalization in Large Language Models", is listed only under cs.AI, but the prediction cs.CL is defensible.

The classifier was built and scored on title + abstract, but the demo gives it a short typed query. We have no labelled queries, so test-paper titles (median 9 words) stood in for them. LogReg falls from 0.792 to 0.698, and real queries may do worse, so the domain label on short input is only a rough guide.

**Limitations**: zero-shot and few-shot ran on subsets of 500 and 200 papers; only one small LLM was tried; the label descriptions (written after the names failed on the same subsets) were chosen on test scores, which makes those rows slightly optimistic; DistilBERT's epoch comes from a single validation split of 960 papers, and every comparison uses one test set; titles only stand in for real queries; each paper gets a single label; and GPU runs are not bit-for-bit repeatable (over three runs with one seed the LSTM ranged 0.661–0.669 and the fine-tune 0.758–0.760), so differences under 0.01 mean little. The choice of seed matters more: at 2 epochs the LSTM's accuracy ranged 0.597–0.694 over three seeds, so its 0.669 is not a stable figure.
**Future improvements**: a multi-label classifier over `all_categories`; a domain-adapted encoder such as SciBERT; a labelled set of real queries, and a classifier trained on short text for the demo; a held-out validation split for choosing label wording; and a larger LLM for the few-shot classifier.
