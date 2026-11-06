# 4. Retrieval, recommendation and topic discovery (Member C)

## 4.1 System architecture
Figure 1 (Section 1.1) shows how the components fit together. A query or abstract passes through extraction (NB02), domain classification (NB03) and MiniLM retrieval (NB04); the retrieved abstracts then ground the LLM's synthesis (NB05), and every generated sentence is checked against the paper it cites before it is shown (NB06). Blue stages use basic techniques (weeks 1–6) and orange stages advanced ones (weeks 7–12).

## 4.2 Techniques
We compared three ways of representing a paper (week 4): sparse TF-IDF over unigrams and bigrams, the average of pretrained spaCy `en_core_web_md` word vectors, and `all-MiniLM-L6-v2` sentence embeddings, a small pretrained encoder transformer (week 8) used without fine-tuning. Papers are ranked by cosine similarity. To evaluate without self-matches, the 9,600 training papers form the index and 200 test papers are the queries; a result counts as relevant if it shares the query's primary category. We report P@5 and nDCG@5, which also rewards placing relevant papers higher. KMeans with k = 6 in each space tests whether the six domains emerge without labels.

## 4.3 Results

| Representation | P@5 | nDCG@5 | Purity | ARI |
|---|---|---|---|---|
| **MiniLM sentence embeddings** | **0.704** | **0.708** | **0.727** | **0.484** |
| TF-IDF (1–2 grams; SVD-100 for KMeans) | 0.619 | 0.636 | 0.572 | 0.263 |
| Averaged word vectors | 0.420 | 0.420 | 0.296 | 0.046 |
| Random baseline | 0.167 | 0.167 | – | – |

*Table 4. Retrieval on 200 test queries and KMeans clustering of the 9,600 training papers.*

MiniLM is best on both tests, so it became the system's retriever (`retrieval.retrieve` and `retrieval.recommend`). On the deployed index of all 12,000 papers, with each query's own paper excluded, it scores P@5 0.717 and nDCG@5 0.720, and a query takes under 0.1 s (88.9 ms on CPU, including encoding).

## 4.4 Discussion
MiniLM wins because it matches meaning rather than shared words, whereas TF-IDF needs the query and paper to use the same terms. Averaged word vectors do worst: averaging about 150 words loses word order and lets frequent generic words pull every abstract towards the same vector, so their clusters are close to random (ARI 0.046). By query domain, MiniLM is strongest on cs.CR (0.865), cs.IR (0.843) and cs.CV (0.812) and weakest on cs.AI (0.526) and cs.LG (0.440). Its clusters show the same pattern: cs.IR (92%), cs.CR (89%) and cs.CV (84%) are clean, while cs.LG, cs.AI and cs.CL are mixed. These are the categories that classification confuses too, so the overlap lies in the labels, not only the models. The PCA plots explain little variance (0.9% for TF-IDF, 12.9% for MiniLM), so they illustrate the clusters rather than prove them.

**Limitations**: same-category relevance is a proxy that counts related cross-listed papers as misses; there are only 200 queries; MiniLM is a general-domain model; and arXiv metadata has no citation graph for citation-based recommendation.
**Future improvements**: scientific embeddings such as SPECTER, hybrid TF-IDF + MiniLM retrieval or a reranker, and citations from the Semantic Scholar API.
