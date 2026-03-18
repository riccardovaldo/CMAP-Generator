# Technical Audit Report: CMAP-Generator

**Role:** Lead Data Scientist & Academic Reviewer  
**Audit Type:** Deep-Dive Methodological Implementation Review  
**Repository:** `riccardovaldo/CMAP-Generator`  
**Files Analyzed:** `main_notebook.ipynb` (27 cells), `classifier.ipynb` (80 cells), `evaluation.ipynb` (44 cells), `graph.py`, `baseline.py`, `import_modules.py`, and all six dataset files.

---

## 1. Project Objective & Hypothesis

### Core Scientific Question

The project asks: **Can an NLP pipeline that combines Open Information Extraction (OpenIE) with rule-based filtering, unsupervised concept clustering, and a supervised classifier automatically generate concept maps from free-form Wikipedia paragraphs that are informatively comparable to human-made concept maps and abstractive summaries?**

### Target Variable and Goal

The pipeline is **dual-purpose**:

1. **Unsupervised / rule-based sub-task (main_notebook.ipynb):** There is no single numeric target variable. The goal is to produce a directed graph (concept map) in which nodes are named entities/concepts and edges are labelled predicate phrases, derived from subject–predicate–object (SPO) triples extracted by Stanford OpenIE. The "quality" of the output is measured downstream by human judgement and by ROUGE-2 overlap against a reference abstractive summarizer (DistilBART).

2. **Supervised sub-task (classifier.ipynb):** The target variable for the saved model (`models/model_file.joblib`) is `is_valuable` (binary, 0/1) defined on the `arg1` column of `data/second_try_annotations.csv`. A value of `1` means the first argument of an SPO triple is a meaningful, non-vague noun phrase that should be retained in the concept map; `0` means it should be pruned.

### Hypothesis

The implicit hypothesis is that combining (a) rule-based linguistic filters, (b) TF-IDF cosine-similarity concept merging, (c) keyword-relevance scoring (TextRank + RAKE), and (d) an `arg1` Random Forest classifier can produce concept maps whose text-serialisation overlaps with a state-of-the-art abstractive summary more than a naive confidence-threshold baseline does.

---

## 2. The Data Pipeline (Step-by-Step Logic)

### 2.1 Data Acquisition & Loading

**Source:** Pre-extracted Wikipedia paragraphs are stored in two forms:

- `data/new_sample_solved.csv` — 19 rows, columns `title` and `text`. Loaded in **main_notebook Cell 16** via `pd.read_csv('../data/new_sample_solved.csv')`. These texts have already been pre-processed with NeuralCoref coreference resolution (performed offline, not shown in the notebooks) so that pronouns have been replaced with their antecedents before OpenIE is applied.
- `data/eval_extract.pickle` — a Python dictionary `{title: DataFrame}`. Loaded in **main_notebook Cell 17** via `pickle.load(open('eval_extract.pickle', 'rb'))`. Each per-title DataFrame has five columns: `sentence`, `arg1`, `rel`, `arg2`, `confidence`. These DataFrames represent the raw output of the Stanford OpenIE standalone server (external tool, not invoked in-notebook), which was applied to the coreference-resolved texts. The OpenIE confidence score is a probability in [0, 1] measuring the extractor's self-assessed certainty that the triple is a valid extraction.

**Annotation datasets (classifier.ipynb only):**

- `data/full_annot.xlsx` — 4,170 rows (first annotation pass). Loaded in **classifier Cell 3** via `pd.read_excel('../data/full_annot.xlsx')`. Columns include `confidence`, `sentence`, `arg1`, `rel`, `arg2`, `negated`, `passive`, `is_valuable`.
- `data/second_try_annotations.csv` — 605 rows (second annotation pass, actually 150 manually selected triples × ~4 annotators). Loaded in **classifier Cell 28** via `pd.read_csv('../data/second_try_annotations.csv')`. Adds `is_valuable_rel` and `is_valuable_arg2` as separate binary targets.

**Evaluation dataset:**

- `data/sample250.csv` — 254 rows with columns `title`, `Model_summary`, `Original_text`, `Bart_summary`. Loaded in **evaluation Cell 28** via `pd.read_csv('../data/sample250.csv')`.
- `data/sample25_with_summaries.csv` — 19 rows with columns `title`, `Original_text`, `Model_summary`, `Baseline`, `Handmade_Map`, `Bart_summary`. Loaded in **evaluation Cell 40** via `pd.read_csv('../data/sample25_with_summaries.csv')`.

### 2.2 Preprocessing & Cleaning

All preprocessing occurs inside `cmap_pipeline()` (**main_notebook Cell 15**), which applies the following seven sequential stages to each per-title OpenIE DataFrame:

#### Stage 1 — `relation_preselection(rel_df)` (main_notebook Cell 13)

Six Boolean filters are applied in sequence using NLTK POS tags (`nltk.pos_tag(word_tokenize(arg))`):

1. **`contains_noun(arg1)`** — Retains only rows where `arg1` contains at least one token with a POS tag in `{'NN', 'NNS', 'NNP', 'NNPS'}`. Rationale: subjects that contain no noun are typically pronouns or adverbs and produce meaningless map nodes.

2. **`just_pronouns_(arg2)`** — Drops rows where `arg2` contains ≤ 2 tokens AND all tokens are tagged as `PRP` or `PRP$`. Rationale: objects consisting of only one or two pronouns carry no informational content (e.g., "it", "they").

3. **`infinitive_rel(rel)`** — Drops rows where `rel` is tagged `TO` and has ≤ 2 tokens (i.e., is of the form "to \<verb\>"). Rationale: infinitive relations extracted by OpenIE typically indicate an incomplete or dependent clause (e.g., "to be" as a predicate is meaningless out of context).

4. **`not_a_verb(rel)`** — Drops rows where `rel` contains no token tagged with any of `{'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}`. Rationale: a predicate with no verb is not a grammatical relation.

5. **`end_in_preposition(arg1)`** and **`end_in_preposition(arg2)`** — Uses the regex pattern `r'\b\w+(?:\s+\w+)*\s+(?:in|on|at|for|of|...)$'` to drop rows where either argument ends with one of 25 specified English prepositions. Rationale: truncated phrases ending in a preposition signal that OpenIE cut off the sentence mid-complement.

6. **`lower_casing(arg1)` and `lower_casing(arg2)`** — If the first token of `arg1` (or `arg2`) is NOT tagged as a noun or numeral, its first character is lower-cased. `rel` is unconditionally lower-cased via `lambda x: x.lower()`. Rationale: normalises capitalisation introduced by sentence-initial positions in OpenIE output.

#### Stage 2 — `filter_rows(df)` (main_notebook Cell 7)

Two sub-steps:

1. **Exact-duplicate removal:** Iterates over the DataFrame, constructs the string `arg1 + ' ' + rel + ' ' + arg2` for each row, adds it to a `seen` set, and drops any row whose phrase string is already in `seen`.
2. **Substring pruning:** For every pair `(l1, l2)` in `seen`, if `l1` is a strict substring of `l2`, `l1` is removed. The final DataFrame is filtered to keep only rows whose concatenated phrase is still in `seen`. Rationale: a shorter phrase is considered redundant if a longer, more informative phrase already covers its content.

#### Stage 3 — `merge_concepts(df, title)` (main_notebook Cell 3)

1. Constructs `nodes = list(df['arg1'].unique()) + [title]`.
2. Fits a `TfidfVectorizer(ngram_range=(1,2))` (defined at module level in **Cell 3**) on `nodes` and computes the cosine similarity matrix via `sklearn.metrics.pairwise.cosine_similarity`.
3. For each node `i`, creates a set of all nodes `j` with `similarity_matrix[i][j] > 0.4`. This produces an initial cluster for each node.
4. Merges overlapping clusters: for each pair of clusters `(set_list[i], set_list[j])`, if their intersection is non-empty, replaces `set_list[i]` with the union and sets `set_list[j]` to the empty set. This is a single-pass greedy merge (not guaranteed to be transitive across three or more overlapping clusters).
5. For each merged cluster, selects the "representative" node: `title` if it is present in the cluster; otherwise `min(cluster, key=lambda x: len(x))` (shortest string by character length).
6. Builds a substitution dictionary `d` and applies it via `df.arg1 = df.arg1.map(d)`. Rationale: concepts that are surface variations of each other (e.g., "the Soviet Union" vs. "Soviet Union") are collapsed to a single node, reducing graph fragmentation.

#### Stage 4 — `concat_concepts(df)` (main_notebook Cell 5)

Handles OpenIE's tendency to decompose conjunction phrases. For example, "France and Britain are rivals" is extracted as two triples: `(France, are, rivals)` and `(Britain, are, rivals)`.

1. **`concat_objects(relation)`** — Groups by `(arg1, rel)`. For the group's list of `arg2` values, finds the longest common prefix word-by-word. Recombines as `"<common prefix> <obj1>, <obj2> and <obj3>"`. Takes the maximum `confidence` in the group.
2. **`concat_subjects(relation)`** — Groups by `(arg2, rel)`. Joins all `arg1` values with `' and '`. Takes the maximum `confidence`.
3. Applied in sequence: first `groupby(['arg1','rel']).apply(concat_objects)`, then `groupby(['arg2','rel']).apply(concat_subjects)`.

#### Stage 5 — `remove_similar_phrases(df)` (main_notebook Cell 7)

1. Constructs the concatenated phrase `arg1 + ' ' + rel + ' ' + arg2`.
2. Fits the same `TfidfVectorizer(ngram_range=(1,2))` on the phrase column and computes the pairwise cosine similarity matrix.
3. For every pair `(i, j)` with `sim[i][j] > 0.3`, adds the index of the **shorter** phrase (by word count) to a removal set `rem_ind`.
4. Drops all rows in `rem_ind`. Rationale: phrases with > 30% TF-IDF cosine overlap are considered semantically redundant; the longer phrase is considered more informative.

#### Stage 6 — Confidence Percentile Filtering (main_notebook Cell 15, line `df = df[df['confidence'] >= df["confidence"].quantile(min_conf_level)]`)

Retains only rows where `confidence` (OpenIE self-assessed extraction confidence) is at or above the value of the 50th percentile (`min_conf_level=0.5`, set in **Cell 22** via `cmap_pipeline(..., 0.5, ...)`). This is a per-title dynamic threshold — the median of each title's own confidence distribution — rather than a global fixed cutoff.

#### Stage 7 — Random Forest Classifier Filtering (main_notebook Cell 15)

The pre-trained model from `models/model_file.joblib` is loaded via `joblib.load('model_file.joblib')` (**Cell 15**, line 1). The `for_class(df, "arg1")` function (**Cell 1**) is applied to the filtered DataFrame, producing a feature matrix `input`. `model.predict(input)` is called and the DataFrame is filtered to `df.loc[res == 1]`, retaining only triples for which the classifier predicts that `arg1` is a valid, meaningful noun phrase.

#### Stage 8 — Keyword Scoring and Final Length Filters (main_notebook Cells 10–11, 15)

`assign_keywords_score(df_text, df, title)` (**Cell 11**) does the following:

1. Calls `extract_keywords_TR(df_text)` using the `summa` library's `keywords.keywords(text, scores=True)` (TextRank algorithm) to produce a `{keyword: score}` dictionary per title.
2. Calls `extract_keywords_RAKE(df_text)` using `multi_rake.Rake().apply(text)[:30]` (Rapid Automatic Keyword Extraction) to produce a second `{keyword: score}` dictionary per title.
3. For each relation phrase, `compute_keywords_score(phrase, keywords_dic)` counts how many keywords appear (case-insensitive substring match) and sums their scores.
4. Columns `TR_count`, `TR_score`, `RAKE_count`, `RAKE_score` are added to the DataFrame.

Three final rule-based filters are then applied in **Cell 15**:

- `df['arg2'].apply(lambda x: len(x.split(' ')) > 1)` — Drops rows where `arg2` is a single word. Rationale: single-word objects produce trivially uninformative edges.
- `df['rel'].apply(lambda x: len(x.split(' ')) < 6)` — Drops rows where `rel` is 6 or more space-separated tokens. Rationale: very long predicates are typically nested clauses that produce visually unreadable edges.
- `(df['TR_score'] != 0) | (df['RAKE_score'] != 0)` — Drops rows where neither TextRank nor RAKE assigned any keyword score. Rationale: relations containing no topic-relevant keywords are considered off-topic for the paragraph.

### 2.3 Feature Engineering

All feature engineering for the classifier is implemented in the `for_class(df, x)` function (**classifier.ipynb Cell 37** for training; replicated as a simplified version in **main_notebook Cell 1** for inference, which omits the label columns, undersampling, and auxiliary DataFrame return).

For a given component `x` ∈ `{'arg1', 'rel', 'arg2'}`, the following features are computed:

| Feature Name | Type | Computation | Domain Rationale |
|---|---|---|---|
| **Embedding** (384 dims) | Dense float vector | `SentenceTransformer('all-MiniLM-L6-v2').encode(x)` — a 384-dimensional L2-normalised sentence embedding from the `all-MiniLM-L6-v2` model | Captures semantic content and contextual meaning of the phrase |
| **`length`** | Scalar int | `len(word_tokenize(x))` — NLTK word token count | Token length as a proxy for phrase completeness and complexity |
| **`Noun`** | Scalar int | Count of tokens tagged `NN`, `NNS`, `NNP`, `NNPS` via `nltk.pos_tag` | A valid `arg1` should be noun-heavy |
| **`Pronoun`** | Scalar int | Count of tokens tagged `PRP`, `PRP$` | High pronoun count in `arg1` indicates vague, unresolved references |
| **`Adjective`** | Scalar int | Count of tokens tagged `JJ`, `JJR`, `JJS` | Adjectival density |
| **`Verb`** | Scalar int | Count of tokens tagged `VB`, `VBD`, `VBG`, `VBN`, `VBP`, `VBZ` | A verb in `arg1` suggests a clause, not a noun phrase |
| **`rel_nouns`** | Scalar float | `Noun / length` | Relative noun density, normalised for phrase length |
| **`rel_pronouns`** | Scalar float | `Pronoun / length` | Relative pronoun density |
| **`rel_adjectives`** | Scalar float | `Adjective / length` | Relative adjective density |
| **`rel_verbs`** | Scalar float | `Verb / length` | Relative verb density |
| **`rel_adverbs`** | Scalar float | Count of `RB`, `RBR`, `RBS` tokens / `length` | Relative adverb density |
| **`rel_determiners`** | Scalar float | Count of `DT`, `PDT`, `WDT` tokens / `length` | Relative determiner density |
| **`rel_numerals`** | Scalar float | Count of `CD` tokens / `length` | Relative numeral density |

The full feature matrix is assembled via `np.hstack([embeddings, len, Noun, Pronoun, Adjective, Verb, rel_nouns, rel_pronouns, rel_adjectives, rel_verbs, rel_adverbs, rel_determiners, rel_numerals])`, resulting in a matrix of shape `(n_samples, 384 + 12) = (n_samples, 396)` (**classifier Cell 37**, last line of `for_class`).

**Note on class imbalance handling (classifier only):** For the `rel` classifier only, `for_class` applies majority-class undersampling (**classifier Cell 37**, lines `db_majority_undersampled = resample(db_majority, replace=False, n_samples=len(db_minority))`). The `arg1` and `arg2` classifiers do not apply any resampling.

---

## 3. Modeling Strategy ("The How")

### 3.1 Baseline Model (`baseline.py`)

**Algorithm:** Confidence-threshold filter only (no machine learning).

**Implementation:** `baseline_cmap(df, title, min_conf_level=0.75)` in **baseline.py, lines 6–14**:

```python
df_filtered = df[df["confidence"] >= df["confidence"].quantile(min_conf_level)]
G = GraphViz(df_filtered, title)
G.graph.write_png(f'example_graph_baseline_{title}_{min_conf_level}.png')
```

Retains all OpenIE triples with confidence at or above the 75th percentile of each title's distribution. **No linguistic filtering, no concept merging, no classifier.** Used as the comparison baseline in the user study and the ROUGE-2 sample-of-25 evaluation.

### 3.2 Attempt 1: Full-sentence Random Forest Classifier (`classifier.ipynb`, Cells 0–25)

**Instantiation:** `RandomForestClassifier(random_state=42)` — **classifier Cell 23**. Default scikit-learn hyperparameters: 100 trees, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, Gini impurity criterion.

**Training data:** `data/full_annot.xlsx`, 4,170 rows. Features: concatenated 384-dim embeddings from all three components `[embeddings_arg1 || embeddings_rel || embeddings_arg2]` plus `confidence` (shape: `(4170, 1)`) and `len_arg1`, `len_rel`, `len_arg2` (each shape `(4170, 1)`), assembled via `np.hstack` in **Cell 19**, giving a `(4170, 1156)` matrix (`3 × 384 + 1 + 3 = 1156`). However, a subsequent overwrite in **Cell 20** replaces this matrix with `final.drop(['is_valuable','sentence','arg1','rel','arg2','negated','passive'], axis=1)`, which retains only the four scalar columns (`confidence`, `len_arg1`, `len_rel`, `len_arg2`) — silently discarding all embeddings. This means the model was actually trained on only 4 scalar features, not the 1,156-dimensional feature matrix assembled in Cell 19.

**Validation:** 5-fold cross-validation (`cross_val_score(rf_classifier, X, y, cv=5)`) in **Cell 23**. Results not stored in a variable. The authors conclude the scores were "disappointingly low" (exact values not printed to a retained output cell).

**Mathematical logic:** Random Forest builds an ensemble of `n_estimators=100` decision trees, each trained on a bootstrap sample of the data. Each split selects the best feature from a random subset of size `max_features='sqrt'` (default) using the Gini impurity criterion: $Gini(t) = 1 - \sum_{k=0}^{1} p_k^2$. Final prediction is by majority vote across all trees.

**Outcome:** Abandoned after poor cross-validation performance. This attempt is NOT the saved model.

### 3.3 Attempt 2: Component-wise Random Forest Classifiers (`classifier.ipynb`, Cells 26–72)

Three separate Random Forest classifiers are trained — one for `arg1`, one for `rel`, one for `arg2` — on the 605-row `second_try_annotations.csv` dataset with 396-dimensional features (384 embedding dims + 12 POS/length features).

#### 3.3.1 Arg1 Classifier (THE SAVED MODEL)

**Instantiation:** `rf_arg1 = RandomForestClassifier(random_state=42)` — **classifier Cell 45**. Default hyperparameters.

**Training:** `rf_arg1.fit(db_arg1, y_arg1)` — **classifier Cell 46**. Dataset: 605 samples from `second_try_annotations.csv`, target `is_valuable`, no resampling applied.

**Serialisation:** `dump(rf_arg1, '../models/model_file.joblib')` — **classifier Cell 47**. This is the default-hyperparameter model (not the tuned one) that is saved.

**Cross-validation:** `cross_val_score(rf_arg1, db_arg1, y_arg1, cv=10)` — **classifier Cell 45**. The narrative states accuracy exceeds 81% but the exact per-fold values are not shown in the retained output.

**Hyperparameter tuning:** `RandomizedSearchCV` with `n_iter=100, cv=5, random_state=50` — **classifier Cell 49**. Search space:
- `n_estimators`: integers in `[100, 200, ..., 900]`
- `max_depth`: `None` or integers in `[5, 10, 15, 20, 25]`
- `min_samples_split`: integers in `[2, 3, ..., 9]`
- `min_samples_leaf`: integers in `[1, 2, ..., 9]`
- `min_impurity_decrease`: floats in `[0.00, 0.01, ..., 0.09]`

The tuned model (`rf_arg1_best`) is evaluated in **Cell 50** but **is not saved**. The default-hyperparameter `rf_arg1` from Cell 46 is the one persisted to disk.

**F1 score:** `cross_val_score(rf_arg1, db_arg1, y_arg1, cv=10, scoring=make_scorer(f1_score))` — **classifier Cell 51**. Exact value not retained in the output cell.

#### 3.3.2 Rel Classifier (NOT saved)

**Instantiation:** `rf_rel = RandomForestClassifier(random_state=42)` — **classifier Cell 54**.

**Class imbalance handling:** Inside `for_class(final, "rel")`, majority-class undersampling is applied: `resample(db_majority, replace=False, n_samples=len(db_minority))` — **classifier Cell 37**. The auxiliary DataFrame `db_aux` is returned for label alignment.

**Custom loss function:** Defined in **classifier Cell 56**:
```python
def loss(y_true, y_pred):
    vect = y_true - y_pred
    loss = 0
    for i in vect:
        if i > 0: loss += i          # false negative: weight 1
        if i < 0: loss += (-5 * i)   # false positive: weight 5
    return loss
```
False positives (predicting a relation is valid when it is not) are penalised 5× more than false negatives. This is passed to `RandomizedSearchCV` as `scoring=make_scorer(loss, greater_is_better=False)` — **classifier Cell 57**.

**Training and evaluation:** 10-fold cross-validation in **Cell 54**; confusion matrix on an 80/20 train-test split (`train_test_split(db_rel, y_rel, test_size=0.2, random_state=50)`) in **Cell 59**. The narrative reports that the model consistently predicts the positive class and struggles with the negative class.

#### 3.3.3 Arg2 Classifier (NOT saved)

**Instantiation:** `rf_arg2 = RandomForestClassifier(random_state=42)` — **classifier Cell 65**.

**Training:** `rf_arg2.fit(db_arg2, y_arg2)` — **classifier Cell 66**. No resampling.

**Tuning:** `RandomizedSearchCV` with the same `loss` function, `n_iter=100, cv=5, random_state=42` — **classifier Cell 67**. The narrative reports near-total prediction of class 1 for both the default and tuned models (confusion matrix in **Cell 69**), attributed to severe class imbalance in `is_valuable_arg2`.

### 3.4 Attempt 3: Full-sentence Classifier on Improved Labels (`classifier.ipynb`, Cells 73–79)

**Motivation:** Tests whether the improved annotations from `second_try_annotations.csv` enable a full-sentence classifier to succeed where the first attempt on `full_annot.xlsx` failed.

**Target construction:** `attempt["is_meaningful"] = [1 if sum(attempt.iloc[i, 7:10]) == 3 else 0 for i in range(len(attempt))]` — **classifier Cell 75**. A triple is "meaningful" only if all three of `is_valuable`, `is_valuable_rel`, and `is_valuable_arg2` are 1.

**Features:** The `for_class_attempt(df)` function in **classifier Cell 77** uses the full `sentence` column (not the individual components) encoded with `all-MiniLM-L6-v2`, plus the same 12 POS/length features, plus `confidence`, `passive`, and `negated` one-hot encoded via `pd.get_dummies`. Function source is truncated in the notebook but its structure mirrors `for_class`.

**Training:** `RandomForestClassifier(random_state=42)` with 10-fold cross-validation — **classifier Cell 79**. Results are not shown in a retained output cell, implying unsatisfactory performance.

### 3.5 BART Summarization Baseline (`evaluation.ipynb`, Cells 1–26)

**Algorithm:** DistilBART (`sshleifer/distilbart-cnn-12-6`) — a distilled version of BART fine-tuned on CNN/DailyMail summarization. Used as an **independent reference**, not as part of the concept-map pipeline.

**Instantiation:** `TFBartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6", from_pt=True)` with `AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")` — **evaluation Cell 1**.

**Generation hyperparameters (selected):** `model.generate(inputs["input_ids"], min_length=150, max_length=300)` — chosen in **evaluation Cell 9** after testing six configurations (default, 100/200, **150/300**, 200/400, 400/800, 600/1200). The third configuration with `min_length=150, max_length=300` was selected because its output statistics (677 chars, 116 words, 7 sentences on the test paragraph) most closely matched the mean concept-map summary statistics (mean ≈ 800 chars, 127 words) from the 25-paragraph sample.

**Encapsulated in:** `get_summary(p)` in **evaluation Cell 24**, which wraps the generation call in a try/except to return `pd.NA` on failure.

**Mathematical logic:** BART uses a denoising autoencoder pre-training objective. At generation time, the decoder uses beam search (default `num_beams=4`) to maximise the sequence probability $P(y|x) = \prod_{t=1}^{T} P(y_t | y_{<t}, x)$ where `x` is the tokenized input paragraph and `y` is the generated summary. The `min_length`/`max_length` parameters constrain the decoder to produce between 150 and 300 tokens.

---

## 4. Evaluation & Interpretation

### 4.1 Metrics

**ROUGE-2** is the primary quantitative metric, implemented via the `rouge` Python library in **evaluation.ipynb**.

`compute_rouge(model_summary, bart_summary, stat)` in **evaluation Cell 29** calls `Rouge().get_scores(model_summary, bart_summary)[0]['rouge-2'][stat]` where `stat` ∈ `{'r', 'p', 'f'}`:

- **Recall (`r`):** What fraction of the bigrams in the BART summary also appear in the concept-map summary.
- **Precision (`p`):** What fraction of the bigrams in the concept-map summary also appear in the BART summary.
- **F1 (`f`):** Harmonic mean of recall and precision.

The BART summaries serve as the "reference" (ground truth proxy) and the concept-map text serialisation serves as the "candidate". This is a non-standard use of ROUGE: normally a human-written reference serves as ground truth; using another model's output as reference makes the metric measure overlap between two automatic systems rather than overlap with human text.

**User study (qualitative):** A user study with 18 participants evaluated concept maps on a Likert scale for comprehension, usefulness, trustworthiness, and completeness. Results are reported in the README and Report_NLP.pdf but are not computed in the notebooks.

### 4.2 Validation Logic

**No train/test split exists for the main pipeline** — the concept-map generation functions are applied to the same 19-paragraph `eval_extract.pickle` dataset without any hold-out. The pipeline is purely deterministic given a fixed input DataFrame; there is no learned component tested on unseen data (the classifier was trained and evaluated separately in `classifier.ipynb`).

**For the classifier:**

- **Attempt 1 (full_annot.xlsx):** 5-fold cross-validation (`cv=5`) in **classifier Cell 23**. No stratification specified (default `StratifiedKFold` is used by scikit-learn for classifiers).
- **Attempt 2 — arg1:** 10-fold cross-validation (`cv=10`) in **classifier Cells 45, 50, 51**.
- **Attempt 2 — rel:** 10-fold cross-validation in **Cell 54**; additional 80/20 split (`test_size=0.2, random_state=50`) in **Cell 59** for confusion matrix.
- **Attempt 2 — arg2:** 10-fold cross-validation in **Cell 65**; 80/20 split (`test_size=0.2, random_state=60`) in **Cell 69**.
- **Attempt 3:** 10-fold cross-validation in **Cell 79**.

No cross-validation is applied during inference of `model_file.joblib` in `main_notebook.ipynb`.

### 4.3 Results

**ROUGE-2 on the 250-paragraph sample** (**evaluation Cell 34** output):

| Metric | Mean | Std | Min | Q25 | Median |
|---|---|---|---|---|---|
| ROUGE-2 Recall | 0.1975 | 0.1235 | 0.000 | 0.1093 | 0.1931 |
| ROUGE-2 Precision | 0.1791 | 0.1031 | 0.000 | 0.1148 | 0.1752 |
| ROUGE-2 F1 | 0.1838 | 0.1071 | 0.000 | 0.1144 | 0.1842 |

These scores are computed over 252 valid rows (2 rows returned `pd.NA` and were excluded). The distribution exhibits a spike at 0 (observed in **evaluation Cell 35**) and no significant correlation between concept-map length and F1 (**evaluation Cell 37**).

**ROUGE-2 Precision on the 25-paragraph annotated sample** (**evaluation Cell 42** output):

| Approach | Mean Precision | Std |
|---|---|---|
| Baseline (75th-percentile confidence filter) | 0.1547 | 0.0496 |
| Model (full pipeline with classifier) | 0.1979 | 0.0860 |
| Handmade (human-created maps) | 0.2237 | 0.1019 |

The model outperforms the baseline by approximately 4.3 percentage points in mean ROUGE-2 precision, but falls below human-made maps by approximately 2.6 percentage points.

**Classifier accuracy (from narrative in classifier.ipynb):**

- Arg1 classifier: > 81% 10-fold cross-validated accuracy.
- Rel classifier: Lower accuracy, significant class imbalance issues.
- Arg2 classifier: Near-degenerate (predicts class 1 for almost all samples).

---

## 5. Critical Critique

### 5.1 Logical Leaps and Missing Steps in the Data Pipeline

**Missing coreference resolution step:** The NeuralCoref coreference resolution step (described in the README as applied before OpenIE) is not implemented in any notebook. The `new_sample_solved.csv` file name implies the texts are "solved" (i.e., coreference-resolved), but the resolution process itself is not reproducible from the provided code. Users cannot recreate the `eval_extract.pickle` starting from raw text.

**Missing OpenIE invocation:** The Stanford OpenIE server call that produces `eval_extract.pickle` is not present in any notebook. There is no code cell that calls the OpenIE REST API or command-line tool. The pipeline is therefore only reproducible from `eval_extract.pickle` onward, not from raw text.

**Vectorizer state leak in `merge_concepts` and `remove_similar_phrases`:** Both functions call `vectorizer.fit_transform(...)` where `vectorizer` is a module-level `TfidfVectorizer` instance defined in **Cell 3**. When `cmap_pipeline` is called in a loop over 19 titles (in **Cell 22**), the vectorizer is re-fit from scratch on each call's internal data, but the module-level state is shared across calls. This is not technically a data-leakage problem in this context (because the vectorizer is re-fit each time), but it makes the code non-thread-safe and produces side effects: the vocabulary fit during `merge_concepts` is immediately overwritten by the fit during `remove_similar_phrases`, meaning `merge_concepts` and `remove_similar_phrases` each use a freshly fit vocabulary on their respective inputs, which is the correct behaviour, but the module-level vectorizer object is mutated as a side effect.

**Single-pass cluster merging in `merge_concepts`:** The cluster merging loop in `merge_concepts` (**Cell 3**, lines 86–91) is a single forward pass over the list. If clusters A and B overlap, and B and C overlap (but A and C do not directly), then A ∪ B is formed in the first pass, but B ∪ C is computed relative to the original B. The merged cluster A ∪ B is not propagated back to check transitivity with C. This can result in two separate clusters that should logically be one.

**Confidence threshold is computed on the post-filtering DataFrame:** In **Cell 15**, `df['confidence'].quantile(min_conf_level)` is computed after `remove_similar_phrases` has already modified `df`. The median confidence is therefore computed on a subset of the original relations rather than on the full raw extraction output. This means the effective confidence cutoff is different from what would be obtained if filtering were applied to the raw OpenIE output — a subtle but reproducibility-relevant design choice that is not documented.

**`for_class` in main_notebook uses only `arg1` features but does not include `confidence`:** The inference version of `for_class` (**main_notebook Cell 1**) assembles a feature matrix of shape `(n, 396)` from embedding + POS features. However, the training version in **classifier Cell 37** also does not include `confidence` as a feature. The first-attempt training in **classifier Cell 18–19** did include `confidence`, but the second-attempt `for_class` does not. This is consistent between training and inference, but the exclusion of a potentially informative feature (OpenIE's self-assessed confidence) is not explicitly justified.

### 5.2 Primary Limitations

**Label quality and inter-annotator agreement not measured:** The `full_annot.xlsx` dataset (4,170 rows) was annotated by two annotators, and `second_try_annotations.csv` (150 sentences × ~4 annotators) by four. No inter-annotator agreement metric (Cohen's κ, Fleiss' κ, or Krippendorff's α) is computed or reported anywhere in the notebooks. The narrative acknowledges "high variability" in the annotations as the cause of poor classifier performance but does not quantify it.

**The saved model was not the best-performing model:** `rf_arg1` (default hyperparameters) is saved to `model_file.joblib` in **classifier Cell 47**, but the hyperparameter-tuned `rf_arg1_best` (produced by `RandomizedSearchCV` in **Cell 49**) is neither saved nor compared quantitatively to `rf_arg1` in a retained output cell. It is possible that the tuned model performed worse than the default (suggesting overfitting to the small dataset), but this is not confirmed by the code.

**Near-degenerate rel and arg2 classifiers are not integrated:** Only the `arg1` classifier is integrated into `cmap_pipeline`. The `rel` and `arg2` classifiers, which exhibited near-degenerate behaviour (predicting class 1 for almost all samples), are trained and evaluated but not used in the pipeline. The pipeline applies length-based proxy filters for `rel` (`len(rel.split(' ')) < 6`) and `arg2` (`len(arg2.split(' ')) > 1`) instead. These are coarse heuristics that do not leverage the semantic information available in the embeddings.

**ROUGE-2 reference is another model's output, not a human reference:** The BART summarization output is used as the ROUGE-2 "reference" throughout `evaluation.ipynb`. This means ROUGE-2 measures the bigram overlap between two automatic systems with different summarization strategies (extractive graph-based vs. abstractive neural). High ROUGE-2 scores would indicate that both systems tend to extract similar bigrams from the source, not that either system is accurate relative to human understanding. The handmade maps in `sample25_with_summaries.csv` would be a more valid reference, but they are used only in the 25-sample precision comparison and not for the full 250-sample evaluation.

**No ablation study:** The pipeline applies seven sequential transformation stages. There is no ablation study quantifying the contribution of each individual stage (e.g., how much does ROUGE-2 F1 change if `merge_concepts` is removed? Or if the classifier is removed?). The only comparison available is the baseline vs. model vs. handmade in the 25-sample ROUGE-2 precision table, which conflates all pipeline stages into a single aggregate comparison.

**`eval_extract.pickle` and `sample250.csv` use partially overlapping titles:** The 19 evaluation paragraphs in `eval_extract.pickle` (derived from `new_sample_solved.csv`) are a subset of the 254 Wikipedia articles in `sample250.csv`. No explicit check is made to ensure these 19 are excluded from the 250-sample ROUGE-2 analysis, which could inflate scores if the model was informally tuned against these specific paragraphs.

**Word-split length filtering is inconsistent with token-count length features:** The final filters in **Cell 15** use Python's `str.split(' ')` (whitespace-only split) to count words, while all feature engineering uses `word_tokenize(x)` (NLTK tokenizer, which handles punctuation). This means a phrase like `"end."` counts as 1 word under `split(' ')` but 2 tokens under `word_tokenize`. The inconsistency is minor in practice but introduces a silent discrepancy in the definition of "word length" across the pipeline.
