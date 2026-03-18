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

The following six issues are documented with exact code references, worked examples, assessed impact, and suggested remediation.

---

#### Critique 1 — Missing Coreference Resolution Step

**What the code says:** `data/new_sample_solved.csv` is loaded in **main_notebook Cell 16** via:
```python
eval_sample = pd.read_csv('../data/new_sample_solved.csv')
```
The README states: *"Before applying OpenIE, the text undergoes coreference resolution using NeuralCoref to handle pronouns and link them to their respective entities."* The filename `new_sample_solved.csv` (emphasis on "solved") and the README description together imply that a coreference resolution step was applied to the raw Wikipedia paragraphs before they were stored in this file. However, **no notebook or script in the repository performs this step**. There is no `import neuralcoref`, no `nlp.add_pipe('neuralcoref')` call, and no pipeline cell that takes raw text and outputs coreference-resolved text.

**Why this matters in practice:** Coreference resolution is the process of replacing pronouns and abbreviated references with the full noun phrase they refer to. For example, consider the sentence *"Napoleon was exiled to Elba. He returned in 1815."* Without resolution, OpenIE would extract the triple `(He, returned, in 1815)`, which produces the meaningless map node "he". With NeuralCoref, "He" is resolved to "Napoleon" before extraction, yielding the correct triple `(Napoleon, returned, in 1815)`. In the 19-paragraph evaluation set, topics include historical figures (Brenton Tarrant, Pugachev's Rebellion) and political entities — both domains with dense pronoun use. A pronoun-filled `arg1` would (a) create anonymous, uninformative nodes in the graph, and (b) be correctly flagged as invalid by the `contains_noun` filter or the `arg1` classifier, meaning such triples would be silently dropped rather than correctly attributed. The net effect is that the pipeline generates fewer concept map nodes than it would from the resolved text, reducing recall without the user being aware why.

**What cannot be reproduced:** The raw Wikipedia paragraphs → coreference-resolved paragraphs transformation cannot be repeated from the available code. This means it is impossible to rerun the full end-to-end pipeline on new paragraphs without (a) installing the specific version of NeuralCoref compatible with the spaCy version used (NeuralCoref is only compatible with spaCy v2, which is end-of-life), and (b) writing the resolution script from scratch.

**Suggested fix:** Add a `resolve_coreferences.py` script or a notebook cell with the following structure:
```python
import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def resolve_coreferences(text):
    doc = nlp(text)
    return doc._.coref_resolved  # NeuralCoref attribute

eval_sample['text_resolved'] = eval_sample['text'].apply(resolve_coreferences)
```

---

#### Critique 2 — Missing OpenIE Invocation

**What the code says:** `data/eval_extract.pickle` is loaded in **main_notebook Cell 17** via:
```python
with open('eval_extract.pickle', 'rb') as f:
    eval_sample_extractions = pickle.load(f)
```
This file is a pre-computed Python dictionary where each key is a paragraph title and each value is a DataFrame with columns `['sentence', 'arg1', 'rel', 'arg2', 'confidence']`. **No code in any notebook calls the Stanford OpenIE server** to generate these DataFrames. The README describes the use of *"Open Information Extraction (OpenIE)"* and references the external repository `dair-iitd/OpenIE-standalone`, but there is no HTTP request, no subprocess call, and no Java invocation in any code cell.

**Why this matters in practice:** The Stanford OpenIE standalone server is a Java application. Calling it from Python requires either: (a) starting the server on a given port and sending HTTP POST requests with the text, or (b) invoking it via a command-line wrapper library such as `openie` (PyPI). To reproduce `eval_extract.pickle` from scratch, a user would need to:
1. Download and start the OpenIE standalone JAR file.
2. Send each coreference-resolved paragraph to the server sentence by sentence.
3. Parse the JSON response to extract `arg1`, `rel`, `arg2`, and `confidence` fields.
4. Assemble the results into DataFrames and pickle them.

None of these steps exist in the repository. The missing code gap is therefore **two layers deep**: first coreference resolution (Critique 1), then OpenIE extraction (this critique). The first cell that actually executes code (`relation_preselection`) already assumes all extraction is done.

**Concrete reproducibility failure:** If a researcher wants to apply this pipeline to a new Wikipedia article — say, "The Battle of Stalingrad" — they cannot do so with the provided code alone. They would need to: (a) write and run the coreference resolution step, (b) write and run the OpenIE extraction step, (c) save the result to a format compatible with `eval_extract.pickle`, and only then (d) call `cmap_pipeline`. In contrast, the README implies this is a complete, runnable pipeline.

**Suggested fix:** A `extract_relations.py` script should be added with:
```python
import requests, json, pickle, pandas as pd

# Endpoint for the Stanford OpenIE standalone server (v5.x).
# The server must be started separately:
#   java -mx8g -cp "*" edu.stanford.nlp.naturalli.OpenIE --port 8080
# Adjust the port and endpoint path to match your server configuration.
OPENIE_URL = "http://localhost:8080/getKBP"

def extract_relations(title, text):
    rows = []
    for sentence in text.split('.'):
        if not sentence.strip():
            continue
        response = requests.post(OPENIE_URL, data={'text': sentence})
        for triple in response.json():
            rows.append({
                'sentence': sentence.strip(),
                'arg1': triple['subject'],
                'rel': triple['relation'],
                'arg2': triple['object'],
                'confidence': triple['confidence']
            })
    return pd.DataFrame(rows)
```

---

#### Critique 3 — Vectorizer State Leak in `merge_concepts` and `remove_similar_phrases`

**What the code says:** In **main_notebook Cell 3**, a single module-level `TfidfVectorizer` instance is created:
```python
vectorizer = TfidfVectorizer(ngram_range=(1,2))
```
This same `vectorizer` object is subsequently used in two separate functions:

In `merge_concepts` (**Cell 3**):
```python
similarity_matrix = cosine_similarity(vectorizer.fit_transform(nodes))
```

In `remove_similar_phrases` (**Cell 7**):
```python
sim = cosine_similarity(vectorizer.fit_transform(df['phrase']))
```

Both calls use `fit_transform`, which (1) discards any previously learned vocabulary, (2) learns a new vocabulary from the input data, and (3) transforms the input. Because `merge_concepts` is called *before* `remove_similar_phrases` inside `cmap_pipeline` (**Cell 15**), each call to `cmap_pipeline` results in the following sequence:
- `vectorizer.fit_transform(nodes)` — vectorizer is trained on `arg1` subjects
- `vectorizer.fit_transform(df['phrase'])` — vectorizer vocabulary is overwritten with phrases

**Why this matters:** Consider what happens when `cmap_pipeline` is called in a loop for 19 titles (**Cell 22**):
```python
for i, key in enumerate(eval_sample_extractions.keys()):
    cmap_pipeline(eval_sample_extractions[key], eval_sample, key, 0.5, i)
```
After the first title is processed, `vectorizer.vocabulary_` contains the IDF weights from that title's `df['phrase']`. When the second title's `merge_concepts` is called, `vectorizer.fit_transform` correctly re-fits on the second title's `nodes` — so the output is correct. However, the vocabulary state after the loop finishes reflects the phrases of the last processed title. Any code that runs after the loop and uses `vectorizer.transform(...)` (without re-fitting) would use the wrong vocabulary.

More critically, the shared mutable state means that if this code were ever called from multiple threads (e.g., via `concurrent.futures` to process titles in parallel), the `fit_transform` calls would race: thread A fitting on title 1's nodes while thread B simultaneously uses the partially fitted vocabulary for title 2's phrases. This would produce silently incorrect similarity matrices with no error raised.

**Suggested fix:** Instantiate a fresh `TfidfVectorizer` inside each function call, eliminating the shared state:
```python
def merge_concepts(df, title):
    local_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    nodes = list(df['arg1'].unique()) + [title]
    similarity_matrix = cosine_similarity(local_vectorizer.fit_transform(nodes))
    ...

def remove_similar_phrases(df):
    local_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    df['phrase'] = df['arg1'] + ' ' + df['rel'] + ' ' + df['arg2']
    sim = cosine_similarity(local_vectorizer.fit_transform(df['phrase']))
    ...
```

---

#### Critique 4 — Single-Pass Non-Transitive Cluster Merging in `merge_concepts`

**What the code says:** The cluster merging loop in **main_notebook Cell 3** is:
```python
for i in range(len(set_list)):
    for j in range(i+1, len(set_list)):
        if set_list[i] & set_list[j]:
            set_list[i] = set_list[i] | set_list[j]
            set_list[j] = set()
set_list = [s for s in set_list if s]
```
This is a single-pass forward scan: for each cluster `i`, it checks all clusters `j > i` and merges any overlapping clusters into `i`, setting `j` to empty.

**Why this fails — a concrete worked example:**

Suppose a paragraph has four subject phrases:
- Concept A: `"the United States"`
- Concept B: `"United States of America"`
- Concept C: `"America"`
- Concept D: `"American government"`

The TF-IDF cosine similarity matrix (hypothetical values) might produce:
- `sim(A, B) = 0.85` → above the 0.4 threshold
- `sim(B, C) = 0.50` → above the 0.4 threshold
- `sim(A, C) = 0.12` → **below** the threshold (no shared unigrams other than "America")
- `sim(C, D) = 0.60` → above the threshold
- `sim(A, D) = 0.08` → below threshold

After the initial per-node cluster construction (where `set_list[i]` = `{i} ∪ {j | sim(i,j) > 0.4}` for all `i`):
```
set_list = [{A, B}, {A, B, C}, {B, C}, {C, D}]
```

The merge loop runs as follows:
1. `i=0 ({A,B})`, `j=1 ({A,B,C})`: intersection `{A,B}` is non-empty → merge: `set_list[0] = {A,B,C}`, `set_list[1] = {}`
2. `i=0 ({A,B,C})`, `j=2 ({B,C})`: intersection `{B,C}` is non-empty → merge: `set_list[0] = {A,B,C}`, `set_list[2] = {}`
3. `i=0 ({A,B,C})`, `j=3 ({C,D})`: intersection `{C}` is non-empty → merge: `set_list[0] = {A,B,C,D}`, `set_list[3] = {}`

In this example the single pass works correctly because set `0` accumulates everything. However, consider a different ordering where the initial clusters are:
```
set_list = [{A, B}, {C, D}, {B, C}]  # ordered so the A-B and C-D clusters appear before the bridging B-C cluster
```
1. `i=0 ({A,B})`, `j=1 ({C,D})`: no intersection (A,B do not share with C,D directly) → no merge
2. `i=0 ({A,B})`, `j=2 ({B,C})`: intersection `{B}` → merge: `set_list[0] = {A,B,C}`, `set_list[2] = {}`
3. `i=1 ({C,D})`, `j=2 ({})`: empty set, skipped

**Result:** `set_list = [{A,B,C}, {C,D}]` — concept C appears in **two separate clusters**. When the dictionary `d` is built, `C` would be mapped to the representative of the first cluster encountered, and `D` would be mapped to the representative of the second cluster. The intended merge of all four concepts into one cluster **fails silently**.

The correct algorithm is to compute the **transitive closure** of the overlap relation, which requires iterating until no further merges occur:
```python
changed = True
while changed:
    changed = False
    for i in range(len(set_list)):
        for j in range(i+1, len(set_list)):
            if set_list[i] and set_list[j] and set_list[i] & set_list[j]:
                set_list[i] = set_list[i] | set_list[j]
                set_list[j] = set()
                changed = True
```

---

#### Critique 5 — Confidence Threshold Computed on the Post-Filtering DataFrame

**What the code says:** Inside `cmap_pipeline` in **main_notebook Cell 15**, the confidence filter is applied **after** five preprocessing steps have already reduced the DataFrame:
```python
def cmap_pipeline(df, df_text, title, min_conf_level, index, save_map=True):
    df = relation_preselection(df)   # step 1 — removes rows by POS tag rules
    df = filter_rows(df)             # step 2 — removes duplicates and substrings
    df = merge_concepts(df, title)   # step 3 — collapses similar arg1 nodes
    df = concat_concepts(df)         # step 4 — merges conjunctions
    df = remove_similar_phrases(df)  # step 5 — removes semantically similar rows
    # ↓ threshold is computed here, on the already-reduced df
    df = df[df['confidence'] >= df["confidence"].quantile(min_conf_level)].reset_index(drop=True)
```

The parameter `min_conf_level=0.5` selects the 50th percentile (median) of the **current** confidence distribution — i.e., the distribution of confidence scores among the triples that survived the first five filtering steps.

**Why this changes the effective cutoff — a numerical example:**

Suppose the original OpenIE extraction for a title produces 100 triples with confidence scores uniformly distributed between 0.3 and 1.0. The true median of the raw distribution is approximately 0.65.

After `relation_preselection`, suppose 30 low-confidence triples (many of which have pronouns in `arg1`, which OpenIE tends to extract with lower confidence) are removed. The remaining 70 triples have a skewed-right confidence distribution with a new median of approximately 0.72.

After `remove_similar_phrases`, suppose another 15 medium-confidence triples are removed (similarity-based removal does not preferentially target any confidence level). The remaining 55 triples have a median of approximately 0.74.

The confidence filter at `min_conf_level=0.5` now retains the top 50% of these 55 triples — those above 0.74 — yielding approximately 27 triples. If the same filter had been applied to the raw 100 triples first, it would have retained those above 0.65, yielding 50 triples, and subsequent filtering would have operated on a larger (and potentially higher-recall) input set.

**Impact:** The effective confidence threshold is higher than `min_conf_level=0.5` would suggest if applied to the raw data, because the early filtering steps disproportionately remove low-confidence rows (e.g., pronoun-containing triples that OpenIE extracts with lower confidence). This means the pipeline may silently discard more relations than intended. The `min_conf_level=0.5` parameter is therefore not directly interpretable as "retain the top 50% of all OpenIE extractions" — it means "retain the top 50% of the triples that survived the linguistic filters."

**Suggested fix:** If the intent is to apply a threshold relative to the raw extraction quality, the confidence filter should be applied first (before linguistic filtering), or the documentation should explicitly state that the threshold is relative to the post-filter distribution.

---

#### Critique 6 — `confidence` Excluded from `for_class` Features Without Justification

**What the first-attempt code used (classifier.ipynb, Cell 18–19):**
```python
# Cell 18 — Attempt 1 feature assembly (full_annot.xlsx)
confidence = final.loc[:, ['confidence']].values
len_arg1   = np.expand_dims(final.loc[:, 'len_arg1'].values, axis=1)
len_rel    = np.expand_dims(final.loc[:, 'len_rel'].values, axis=1)
len_arg2   = np.expand_dims(final.loc[:, 'len_arg2'].values, axis=1)

# Cell 19
X = np.hstack([embeddings_arg1, embeddings_rel, embeddings_arg2,
               confidence, len_arg1, len_rel, len_arg2])
```
In the first attempt, `confidence` was explicitly included as one of the 7 scalar features (alongside the three length features).

**What the second-attempt `for_class` function uses (classifier.ipynb, Cell 37 — the function that produces the saved model's training data):**
```python
def for_class(df, x):
    ...
    db = np.hstack([embeddings_arg1, len_arg1, nouns, pronouns,
                    adjectives, verbs, rel_nouns, rel_pronouns,
                    rel_adjectives, rel_verbs, rel_adverbs,
                    rel_determiners, rel_numerals])
    return db, db_aux
```
The `confidence` column is present in the source DataFrame (`second_try_annotations.csv`) but is never extracted or included in the feature matrix. The inference version in **main_notebook Cell 1** matches: it also omits `confidence`.

**Why this matters:** The OpenIE `confidence` score is an explicit, model-derived quality signal for each individual triple. High-confidence extractions are those where the OpenIE model's internal probability is high that the extracted triple faithfully represents the source sentence. It is therefore a direct, numerically measurable proxy for the kind of validity that `is_valuable` is trying to capture (whether a triple makes sense). In the first attempt (Cells 18–19), including `confidence` was a deliberate choice. In the second attempt (Cell 37), the feature was silently dropped without explanation.

**Concrete inconsistency between attempts:** The second attempt replaces `confidence` with 11 POS-tag count and ratio features. This expansion is well-motivated (POS composition captures phrase type), but the removal of `confidence` as a complementary feature is not. A Random Forest with `n=150` training samples and `396` features (of which 384 are dense embedding dimensions) could potentially benefit from having `confidence` as an additional low-noise scalar feature, since the tree-splitting algorithm can directly threshold it with a single decision boundary. The current feature set requires the Random Forest to infer confidence-related distinctions indirectly from the POS composition and embedding of the phrase.

**Impact assessment:** Both training (`classifier Cell 37`) and inference (`main_notebook Cell 1`) omit `confidence` consistently, so there is no train/inference mismatch (which would be a bug). The issue is a missed opportunity: a potentially high-signal feature was available and present in the training data but was not used in the model that was ultimately saved and deployed. The classifier's reported >81% accuracy may have been higher had `confidence` been retained.

### 5.2 Primary Limitations

**Label quality and inter-annotator agreement not measured:** The `full_annot.xlsx` dataset (4,170 rows) was annotated by two annotators, and `second_try_annotations.csv` (150 sentences × ~4 annotators) by four. No inter-annotator agreement metric (Cohen's κ, Fleiss' κ, or Krippendorff's α) is computed or reported anywhere in the notebooks. The narrative acknowledges "high variability" in the annotations as the cause of poor classifier performance but does not quantify it.

**The saved model was not the best-performing model:** `rf_arg1` (default hyperparameters) is saved to `model_file.joblib` in **classifier Cell 47**, but the hyperparameter-tuned `rf_arg1_best` (produced by `RandomizedSearchCV` in **Cell 49**) is neither saved nor compared quantitatively to `rf_arg1` in a retained output cell. It is possible that the tuned model performed worse than the default (suggesting overfitting to the small dataset), but this is not confirmed by the code.

**Near-degenerate rel and arg2 classifiers are not integrated:** Only the `arg1` classifier is integrated into `cmap_pipeline`. The `rel` and `arg2` classifiers, which exhibited near-degenerate behaviour (predicting class 1 for almost all samples), are trained and evaluated but not used in the pipeline. The pipeline applies length-based proxy filters for `rel` (`len(rel.split(' ')) < 6`) and `arg2` (`len(arg2.split(' ')) > 1`) instead. These are coarse heuristics that do not leverage the semantic information available in the embeddings.

**ROUGE-2 reference is another model's output, not a human reference:** The BART summarization output is used as the ROUGE-2 "reference" throughout `evaluation.ipynb`. This means ROUGE-2 measures the bigram overlap between two automatic systems with different summarization strategies (extractive graph-based vs. abstractive neural). High ROUGE-2 scores would indicate that both systems tend to extract similar bigrams from the source, not that either system is accurate relative to human understanding. The handmade maps in `sample25_with_summaries.csv` would be a more valid reference, but they are used only in the 25-sample precision comparison and not for the full 250-sample evaluation.

**No ablation study:** The pipeline applies seven sequential transformation stages. There is no ablation study quantifying the contribution of each individual stage (e.g., how much does ROUGE-2 F1 change if `merge_concepts` is removed? Or if the classifier is removed?). The only comparison available is the baseline vs. model vs. handmade in the 25-sample ROUGE-2 precision table, which conflates all pipeline stages into a single aggregate comparison.

**`eval_extract.pickle` and `sample250.csv` use partially overlapping titles:** The 19 evaluation paragraphs in `eval_extract.pickle` (derived from `new_sample_solved.csv`) are a subset of the 254 Wikipedia articles in `sample250.csv`. No explicit check is made to ensure these 19 are excluded from the 250-sample ROUGE-2 analysis, which could inflate scores if the model was informally tuned against these specific paragraphs.

**Word-split length filtering is inconsistent with token-count length features:** The final filters in **Cell 15** use Python's `str.split(' ')` (whitespace-only split) to count words, while all feature engineering uses `word_tokenize(x)` (NLTK tokenizer, which handles punctuation). This means a phrase like `"end."` counts as 1 word under `split(' ')` but 2 tokens under `word_tokenize`. The inconsistency is minor in practice but introduces a silent discrepancy in the definition of "word length" across the pipeline.
