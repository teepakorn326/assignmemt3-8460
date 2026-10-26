"""Member A: preprocessing and information extraction for arXiv abstracts.

Used by 02_preprocessing_extraction.ipynb, and by 06_end_to_end_demo.ipynb
(D calls `preprocess` and `extract` on the user's query or on retrieved papers).

    from extraction import clean_text, preprocess, extract, llm_extract
    preprocess("We fine-tune BERT models ...")   -> ['fine', 'tune', 'bert', 'model', ...]
    extract("We fine-tune BERT on ImageNet ...")  -> {'key_phrases': [...], 'entities': [...], ...}

Techniques: W2 cleaning / tokenisation / stop-words / lemmatisation, W3 POS tags,
noun chunks and NER (spaCy), W3 rule-based extraction (regex), W9 few-shot prompting.
"""
import json
import re

import spacy

_nlp = None


def get_nlp():
    """Load spaCy's small English model once and reuse it (loading takes a few seconds)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# W2: cleaning and preprocessing
# ---------------------------------------------------------------------------

def clean_text(text):
    """Remove arXiv-specific noise: LaTeX maths, LaTeX commands, URLs and extra spaces.

    Adapted from clean_text() in the Week 2 practical, with two LaTeX rules added
    because about 7% of the abstracts contain $...$ maths.
    """
    text = re.sub(r"\$[^$]*\$", " ", text)                 # inline maths: $\alpha$, $O(n^2)$
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)   # \emph{word} -> word
    text = re.sub(r"\\[a-zA-Z]+", " ", text)                # leftover commands: \alpha
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\s+", " ", text).strip()               # extra whitespace
    return text


def preprocess(text, lemmatise=True):
    """Clean -> tokenise -> lowercase -> drop stop-words, punctuation and numbers -> lemmatise.

    Returns a list of tokens. With lemmatise=False the lowercase surface form is kept,
    so the two versions can be compared.
    """
    doc = get_nlp()(clean_text(text))
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if not token.is_alpha:          # drops numbers and tokens like "3D", "x2"
            continue
        if lemmatise:
            tokens.append(token.lemma_.lower())
        else:
            tokens.append(token.text.lower())
    return tokens


# ---------------------------------------------------------------------------
# W3: POS-based key phrases, noun chunks, NER
# ---------------------------------------------------------------------------

def pos_key_phrases(doc):
    """Key-phrase candidates matching the POS pattern ADJ* NOUN+ (e.g. 'masked autoencoder').

    Walk through the tokens and keep building a phrase while we see adjectives or nouns.
    A phrase is kept only if it ends in a noun and has at least two words.
    """
    phrases = []
    current = []
    for token in doc:
        if _is_hyphen_inside_word(token) and current:
            current.append(token)          # keep "ground-truth" together
        elif token.pos_ in ("ADJ", "NOUN", "PROPN"):
            # an adjective after a noun starts a new phrase (ADJ must come first)
            if token.pos_ == "ADJ" and current and current[-1].pos_ in ("NOUN", "PROPN"):
                _close_phrase(doc, current, phrases)
                current = []
            current.append(token)
        else:
            _close_phrase(doc, current, phrases)
            current = []
    _close_phrase(doc, current, phrases)
    return phrases


def _is_hyphen_inside_word(token):
    """True for the '-' in 'ground-truth' (no spaces on either side)."""
    return token.text == "-" and not token.whitespace_ and token.i > 0 and not token.nbor(-1).whitespace_


def _close_phrase(doc, tokens, phrases):
    # trim trailing adjectives / hyphens so the phrase ends in a noun
    while tokens and (tokens[-1].pos_ == "ADJ" or tokens[-1].text == "-"):
        tokens = tokens[:-1]
    words = [t for t in tokens if t.text != "-"]
    if len(words) >= 2:
        span = doc[tokens[0].i: tokens[-1].i + 1]      # original text, so hyphens keep their spacing
        phrases.append(span.text.lower())


def noun_chunks(doc):
    """spaCy's built-in base noun phrases, without leading determiners/pronouns."""
    chunks = []
    for chunk in doc.noun_chunks:
        start = chunk.start
        while start < chunk.end and doc[start].pos_ in ("DET", "PRON"):
            start += 1
        span = doc[start:chunk.end]
        if len(span) >= 2:
            chunks.append(span.text.lower())
    return chunks


def spacy_entities(doc, labels=("ORG", "PRODUCT", "PERSON", "GPE", "WORK_OF_ART", "DATE")):
    """Named entities from spaCy's pretrained NER, as (text, label) pairs."""
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in labels]


# ---------------------------------------------------------------------------
# W3 (rule-based IE): regular expressions for method and dataset names
# ---------------------------------------------------------------------------

# Acronyms so common in CS abstracts that they are never the paper's own method/dataset.
GENERIC_ACRONYMS = {
    "AI", "ML", "DL", "NLP", "CV", "IR", "RL", "LLM", "LLMs", "LM", "LMs", "VLM", "VLMs",
    "MLLM", "MLLMs", "GPU", "GPUs", "CPU", "API", "APIs", "SOTA", "ID", "IDs", "RGB", "IoT",
    "QA", "OOD", "DNN", "DNNs", "CNN", "CNNs", "GNN", "GNNs", "RNN", "RNNs", "GAN", "GANs",
    "AUC", "US", "UK", "EU", "PhD", "TB", "GB", "MB", "FPS", "SQL", "PDF", "URL",
}

ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9.]+)?\b")
DEFINITION_PATTERN = re.compile(r"((?:[A-Z][a-z]+[\s-]){1,6}[A-Z][a-z]+)\s\(([A-Z][A-Za-z0-9-]+)\)")
DATASET_PATTERN = re.compile(
    r"\b([A-Z][\w-]*(?:\s[A-Z][\w-]*)*)\s(?:dataset|datasets|benchmark|benchmarks|corpus|corpora)\b")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s?%")


def regex_extract(text):
    """Rule-based extraction of named methods/datasets, acronym definitions, years and percentages."""
    text = clean_text(text)

    acronyms = []
    for match in ACRONYM_PATTERN.findall(text):
        if "-" in match:
            base, suffix = match.split("-", 1)
            if suffix.isalpha() and suffix.islower():   # "ReAct-style", "LLM-based" -> drop the suffix
                match = base
        else:
            base = match
        if match not in GENERIC_ACRONYMS and base not in GENERIC_ACRONYMS and match not in acronyms:
            acronyms.append(match)

    definitions = []                      # "Dynamic Task Coordination (DTC)" -> ("DTC", "Dynamic Task Coordination")
    for long_form, short_form in DEFINITION_PATTERN.findall(text):
        definitions.append((short_form, long_form.strip()))

    datasets = []                         # "the GLUE benchmark", "MS-COCO dataset"
    for name in DATASET_PATTERN.findall(text):
        # a single ordinary capitalised word ("Synthetic datasets ...") is just a sentence start
        looks_like_a_name = (" " in name or any(ch.isdigit() for ch in name)
                             or any(ch.isupper() for ch in name[1:]))
        if looks_like_a_name and name not in datasets:
            datasets.append(name)

    return {
        "acronyms": acronyms,
        "definitions": definitions,
        "datasets": datasets,
        "years": sorted(set(YEAR_PATTERN.findall(text))),
        "percentages": PERCENT_PATTERN.findall(text),
    }


def extract(text):
    """Everything the basic (non-LLM) extractors find, in one dict. Used by NB06."""
    doc = get_nlp()(clean_text(text))
    rules = regex_extract(text)
    return {
        "key_phrases": pos_key_phrases(doc),
        "noun_chunks": noun_chunks(doc),
        "entities": spacy_entities(doc),
        "acronyms": rules["acronyms"],
        "definitions": rules["definitions"],
        "datasets": rules["datasets"],
        "years": rules["years"],
        "percentages": rules["percentages"],
    }


# Predictions in the {methods, datasets} format used for the hand-labelled evaluation

def regex_artefacts(text):
    """Regex extractor's guess of method + dataset names."""
    rules = regex_extract(text)
    names = list(rules["acronyms"]) + list(rules["datasets"])
    for short_form, long_form in rules["definitions"]:
        names.append(short_form)
    return _unique(names)


def spacy_artefacts(text):
    """spaCy NER's guess of method + dataset names (it labels most of them ORG or PRODUCT)."""
    doc = get_nlp()(clean_text(text))
    return _unique([ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART")])


def _unique(items):
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


# ---------------------------------------------------------------------------
# W9: few-shot LLM extraction (advanced)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "abstract": "We introduce RoBERTa-Large-Med, a variant of RoBERTa further pre-trained on "
                    "PubMed abstracts, and evaluate it on the BC5CDR and NCBI-Disease named entity "
                    "recognition benchmarks, where it outperforms BioBERT.",
        "answer": {"methods": ["RoBERTa-Large-Med", "RoBERTa", "BioBERT"],
                   "datasets": ["PubMed", "BC5CDR", "NCBI-Disease"],
                   "tasks": ["named entity recognition"]},
    },
    {
        "abstract": "We propose a lightweight vision transformer, TinyViT, trained with knowledge "
                    "distillation from a large teacher. On ImageNet-1k it reaches 84.8% top-1 "
                    "accuracy and transfers well to COCO object detection.",
        "answer": {"methods": ["TinyViT", "knowledge distillation"],
                   "datasets": ["ImageNet-1k", "COCO"],
                   "tasks": ["image classification", "object detection"]},
    },
    {
        "abstract": "Phishing websites remain a major threat. We study how well classical "
                    "detectors generalise over time using a new longitudinal collection of URLs, "
                    "and find that gradient-boosted trees degrade sharply after six months.",
        "answer": {"methods": ["gradient-boosted trees"],
                   "datasets": [],
                   "tasks": ["phishing detection"]},
    },
]

SYSTEM_PROMPT = ("You are an information-extraction assistant for computer-science papers. "
                 "Only list names that appear in the abstract. Answer with JSON only.")


def build_few_shot_prompt(abstract):
    """Instruction + three worked examples + the new abstract (few-shot prompting, Week 9)."""
    parts = ["Extract the named methods/models, datasets/benchmarks and research tasks "
             "mentioned in the abstract. Return JSON with keys \"methods\", \"datasets\", \"tasks\". "
             "Use an empty list when nothing fits.\n"]
    for example in FEW_SHOT_EXAMPLES:
        parts.append("Abstract: " + example["abstract"])
        parts.append("JSON: " + json.dumps(example["answer"]) + "\n")
    parts.append("Abstract: " + clean_text(abstract))
    parts.append("JSON:")
    return "\n".join(parts)


def parse_json_answer(text):
    """Pull the first {...} block out of the LLM's reply; return empty lists if it is not valid JSON."""
    empty = {"methods": [], "datasets": [], "tasks": []}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return empty
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return empty
    result = {}
    for key in empty:
        value = data.get(key, [])
        result[key] = [str(v) for v in value] if isinstance(value, list) else []
    return result


def llm_extract(abstract, max_new_tokens=200):
    """Few-shot LLM extraction through the team's shared llm.generate (cached, temperature 0)."""
    from llm import generate
    reply = generate(build_few_shot_prompt(abstract), system=SYSTEM_PROMPT,
                     max_new_tokens=max_new_tokens, temperature=0.0)
    return parse_json_answer(reply)


# ---------------------------------------------------------------------------
# Evaluation against hand labels
# ---------------------------------------------------------------------------

def normalise(name):
    """Lowercase and keep only letters/digits, so 'ImageNet-1k' == 'imagenet 1k'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_counts(predicted, gold):
    """Count true positives with a relaxed match: equal after normalising, or one contains the other.

    Returns (true_positives, number_predicted, number_gold).
    """
    pred = _unique([normalise(p) for p in predicted if normalise(p)])
    gold = _unique([normalise(g) for g in gold if normalise(g)])
    matched_gold = set()
    true_positives = 0
    for p in pred:
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if p == g or (len(p) > 2 and len(g) > 2 and (p in g or g in p)):
                matched_gold.add(i)
                true_positives += 1
                break
    return true_positives, len(pred), len(gold)


def precision_recall_f1(tp, n_pred, n_gold):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)
