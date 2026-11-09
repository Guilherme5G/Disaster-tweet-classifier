# Experimental Report — Disaster Tweet Classification

## 1. Research Question and Hypotheses

### Research Question

How do Multinomial Naive Bayes, Logistic Regression, and LinearSVC compare in terms of F1-score when classifying disaster-related tweets using TF-IDF text representations under a group-aware validation protocol?

### Hypotheses

**H0 (Null Hypothesis):**

There is no difference in expected F1-score performance among Multinomial Naive Bayes, Logistic Regression, and LinearSVC when classifying disaster-related tweets with TF-IDF features on the target dataset.

**H1 (Alternative Hypothesis):**

At least one of the evaluated classifiers (Multinomial Naive Bayes, Logistic Regression, or LinearSVC) achieves a different expected F1-score than the others when classifying disaster-related tweets with TF-IDF features on the target dataset.

### Scope

This study focuses on a primary controlled comparison of three linear classifiers — **Multinomial Naive Bayes** (`alpha=1.0`), **Logistic Regression** (`C=1.0`), and **LinearSVC** (`C=1.0`) — using **TF-IDF** (`TfidfVectorizer`) text representations on the **Kaggle "Natural Language Processing with Disaster Tweets"** dataset (7,613 labeled tweets). Separately from this main three-model comparison on the held-out validation split, additional hyperparameter experiments (such as `C in [0.1, 1.0, 10.0]` for Logistic Regression) were conducted strictly within the training partition via group-aware cross-validation. The **F1-score** for the positive class (`target = 1`) serves as the primary evaluation metric, supported by Precision, Recall, Accuracy, and Confusion Matrix counts.

No formal statistical significance test (such as paired hypothesis testing or bootstrap confidence intervals) was conducted; therefore, the hypotheses above frame an empirical model selection protocol rather than a statistical proof of superiority.


---

## 2. Dataset and Experimental Setup

### 2.1 Dataset Description

The experiments use the public **Natural Language Processing with Disaster Tweets** dataset from Kaggle, which consists of 7,613 labeled tweets in `train.csv` and 3,263 unlabeled tweets in `test.csv`. The task is a binary text classification problem: predicting whether a given tweet announces a real disaster (`target = 1`) or does not (`target = 0`, including everyday conversation or figurative use of disaster-related words).

Dataset: https://www.kaggle.com/competitions/nlp-getting-started

| Class | Samples | Percentage |
|---|---:|---:|
| Non-disaster (0) | 4,342 | 57.03% |
| Disaster (1) | 3,271 | 42.97% |
| Total | 7,613 | 100% |

The dataset exhibits a moderate class imbalance favoring the non-disaster class (57.03% vs. 42.97%). Because the negative class is more frequent, a conservative classifier that under-predicts disasters can still achieve reasonably high overall Accuracy while failing to detect a large proportion of real disaster tweets. Consequently, Accuracy alone is insufficient, motivating the use of **F1-score**, **Precision**, and **Recall** on the positive class (`target = 1`).

### 2.2 Feature and Target Selection

To evaluate the baseline predictive power of raw tweet content and maintain a self-contained text processing pipeline, only the raw text string was used as model input:

- Input feature: `text`
- Target variable: `target`
- Unused columns: `id`, `keyword`, `location`

The `id` column is an arbitrary record identifier with no legitimate predictive signal. Excluding `keyword` and `location` simplifies the pipeline and avoids introducing missing-value imputation heuristics or noisy user-typed geographic strings; however, omitting `keyword` also discards potentially informative curator tags that could help disambiguate short tweets.

### 2.3 Train-Validation Split

To evaluate generalization on unseen texts before final submission, the labeled dataset was partitioned into a development training set (`train_df`) and a held-out validation set (`val_df`) using `GroupShuffleSplit`.

Configuration:
- `n_splits=1`
- `test_size=0.2`
- `random_state=42`
- `groups=df["text"]`

| Partition | Samples |
|---|---:|
| Training | 6,083 |
| Validation | 1,530 |
| Total | 7,613 |

Exact text overlap between partitions: **0**

Grouping by `df["text"]` rather than `df["id"]` is essential because every row in `train.csv` has a unique `id`, whereas multiple rows share identical tweet text. Grouping by `id` would treat each row independently and allow identical tweets to appear in both training and validation sets, causing duplicate-text data leakage and overly optimistic validation metrics.

Furthermore, specifying `test_size=0.2` in `GroupShuffleSplit` allocates **20% of the distinct groups** (unique text strings) to the validation set rather than 20% of individual rows. Because text groups vary in size (most appear once, while duplicated tweets appear multiple times) and all rows belonging to a group must remain together, the resulting validation partition contains 1,530 rows (approximately 20.10% of the 7,613 total rows).


---

## 3. Methodology and Leakage Prevention

### 3.1 Text Representation

Tweets are converted into numerical feature vectors using **TF-IDF** (`TfidfVectorizer`), which weights each token by its term frequency in a document scaled by its inverse document frequency across the corpus. TF-IDF is well suited for linear text classification because it produces high-dimensional sparse representations while down-weighting ubiquitous, uninformative words.

During `.fit()`, the vectorizer learns the fixed token vocabulary from the training corpus and computes smoothed inverse document frequency weights using the Scikit-learn formula $\text{IDF}(t) = \ln((1 + N) / (1 + \text{DF}(t))) + 1$, followed by L2 row normalization during transformation.

### 3.2 Preprocessing Leakage Prevention

Fitting `TfidfVectorizer` on the entire dataset prior to splitting (`vectorizer.fit_transform(df["text"])`) would introduce **preprocessing data leakage**: validation texts would influence both the learned vocabulary and the global document-frequency ($\text{DF}$) statistics used to scale training features, even if the classifier itself were only fitted on `X_train`.

To prevent this, `fit_transform()` is executed strictly on `train_df["text"]` (learning 18,621 distinct terms from the 6,083 training tweets), whereas `val_df["text"]` is transformed exclusively via `vectorizer.transform()`. Any out-of-vocabulary tokens appearing only in validation are ignored, preserving an identical 18,621-column feature space across both partitions.

### 3.3 Group-Aware Cross-Validation

For hyperparameter tuning and internal candidate selection within the 6,083-sample training partition, we employ `GroupKFold(n_splits=3)` with `groups=train_df["text"]`. This partitions the training data into three mutually exclusive folds while enforcing that all records sharing an identical text string are assigned to the same fold. Keeping identical texts together prevents internal train-validation leakage across folds, ensuring that cross-validation scores reflect generalization to unseen text groups.

### 3.4 Model Selection Protocol

Candidate configurations were compared using 3-fold `GroupKFold` cross-validation strictly inside `train_df` (`src/select_model.py` and `src/tune_logistic.py`), recording mean F1, Precision, and Recall across folds. Keeping hyperparameter tuning and candidate selection inside `train_df` preserves `val_df` (1,530 records) as an untouched external validation benchmark, avoiding selection bias on the held-out set.

During cross-validation, `TfidfVectorizer()` and the classifier are encapsulated inside a Scikit-learn `Pipeline`. This guarantees that in every fold iteration, the TF-IDF vocabulary and IDF weights are fitted exclusively on the two training folds and applied via `transform()` to the internal validation fold, eliminating cross-fold preprocessing leakage.


---

## 4. Models and Evaluation Metrics

### 4.1 Evaluated Classifiers

**Multinomial Naive Bayes**
- `alpha=1.0`

Serves as a fast probabilistic baseline that estimates class-conditional token likelihoods with Laplace smoothing under the naive assumption of conditional feature independence given the class.

**Logistic Regression**
- `C=1.0`
- `max_iter=1000`
- `random_state=42`

Serves as a discriminative linear classifier that optimizes L2-regularized log-loss (`z = w^T x + b`) to estimate calibrated class probabilities via the sigmoid function without assuming feature independence.

**LinearSVC**
- `C=1.0`
- `max_iter=5000`
- `random_state=42`

Serves as a margin-based discriminative linear classifier that minimizes L2-regularized squared hinge loss, focusing parameter updates on examples near or violating the decision margin.

In addition to this primary three-model baseline comparison, alternative regularization strengths (`C in [0.1, 1.0, 10.0]` for Logistic Regression) were investigated separately using internal 3-fold `GroupKFold` cross-validation on `train_df`.

### 4.2 Evaluation Metrics

**Accuracy**

Measures the overall proportion of correct predictions $(TP + TN) / (TP + TN + FP + FN)$, but can mask poor sensitivity on the minority positive class when classes are imbalanced.

**Precision**

Measures the proportion of predicted disaster tweets that are actual disasters, $TP / (TP + FP)$, quantifying resistance to false alarms.

**Recall**

Measures the proportion of actual disaster tweets correctly identified by the model, $TP / (TP + FN)$, quantifying sensitivity to real disaster reports.

**F1-score**

The harmonic mean of Precision and Recall, $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$. It is used as the primary comparison metric because it penalizes extreme imbalance between false positives and false negatives and matches the official Kaggle evaluation criterion.

**Confusion Matrix**

Decomposes predictions into True Negatives (**TN**: class 0 correctly predicted as 0), False Positives (**FP**: class 0 incorrectly predicted as 1), False Negatives (**FN**: class 1 incorrectly predicted as 0), and True Positives (**TP**: class 1 correctly predicted as 1).


---

## 5. Experimental Results

### 5.1 Validation Performance

All models were evaluated on the same validation partition
containing 1,530 observations.

| Metric | Naive Bayes | Logistic Regression | LinearSVC |
|---|---:|---:|---:|
| Accuracy | 0.7863 | 0.7954 | 0.7948 |
| Precision | 0.8643 | 0.8092 | 0.7891 |
| Recall | 0.6124 | 0.7027 | 0.7308 |
| F1-score | 0.7169 | 0.7522 | 0.7588 |

The three classifiers exhibit distinct operating profiles: **Multinomial Naive Bayes** achieves the highest Precision (`0.8643`) but suffers from low Recall (`0.6124`), yielding the lowest F1-score (`0.7169`). **Logistic Regression** improves Recall to `0.7027` while maintaining high Precision (`0.8092`), reaching an F1-score of `0.7522`. **LinearSVC** achieves the highest observed **F1-score (`0.7588`)** and the highest **Recall (`0.7308`)**.

Although Logistic Regression achieved a marginally higher Accuracy (`0.7954` vs. `0.7948` for LinearSVC, a difference of 1 correct prediction out of 1,530), Accuracy is dominated by the majority non-disaster class (`target = 0`). LinearSVC is preferred under the project's primary criterion because its higher Recall on the positive class yields a better Precision-Recall balance (F1 = `0.7588`).

### 5.2 Confusion Matrix Comparison

| Result | Naive Bayes | Logistic Regression | LinearSVC |
|---|---:|---:|---:|
| True Negatives | 789 | 742 | 722 |
| False Positives | 65 | 112 | 132 |
| False Negatives | 262 | 201 | 182 |
| True Positives | 414 | 475 | 494 |

Moving from Naive Bayes to Logistic Regression and LinearSVC progressively shifts the decision boundary toward higher sensitivity for class 1: True Positives increase from 414 (Naive Bayes) to 475 (Logistic Regression) and 494 (LinearSVC), while False Negatives drop from 262 to 201 and 182, respectively. Conversely, False Positives increase from 65 to 112 and 132.

### 5.3 Model Selection

`LinearSVC(C=1.0)` was selected as the final pipeline architecture because it achieved the highest F1-score both in internal 3-fold group-aware cross-validation on `train_df` (`f1_mean = 0.7460`) and on the held-out validation set (`F1 = 0.7588`), while also providing the highest Recall (`0.7308`). Prioritizing F1-score and Recall aligns with the goal of reducing missed disaster reports (`FN`) without excessively degrading Precision.

However, the F1-score difference between `LinearSVC` (`0.7588`) and `LogisticRegression` (`0.7522`) is small (`0.0066`), and internal cross-validation showed close performance across discriminative linear candidates. Therefore, this selection represents a practical engineering decision under our predefined protocol rather than proof of statistically significant superiority.

### 5.4 Kaggle Submission Result

The selected TF-IDF + LinearSVC Pipeline was refitted
using all 7,613 labeled training observations.

Official Kaggle submission:
- Predictions generated: 3,263
- Leaderboard F1-score: **0.79313**

The Kaggle leaderboard score (`0.79313`) and our local validation score (`0.7588`) are not directly comparable because they reflect different evaluation settings: the submission pipeline was trained on all 7,613 labeled examples (incorporating the 1,530 validation rows into the vocabulary and classifier fit), and the Kaggle test partition may have a different distribution or degree of text overlap with the training set than our strict group-disjoint split (`Exact text overlap = 0`).

Furthermore, the Kaggle score measures performance on a static competition test set from the same historical collection and must not be interpreted as an independent estimate of real-world emergency deployment performance.


---

## 6. Error Analysis and Trade-offs

### 6.1 False Positives

In disaster tweet classification, a **False Positive (FP)** occurs when a non-disaster tweet (`target = 0`) is classified as a disaster (`prediction = 1`). In a downstream monitoring workflow, false positives generate false alarms that consume human triage capacity and, if excessive, can lead to operator alarm fatigue.

### 6.2 False Negatives

A **False Negative (FN)** occurs when an actual disaster-related tweet (`target = 1`) is classified as non-disaster (`prediction = 0`). In an emergency monitoring context, false negatives are critical errors because relevant incident reports go undetected.

### 6.3 Precision-Recall Trade-off

Compared with Naive Bayes, LinearSVC produced:
- 80 fewer false negatives.
- 67 additional false positives.
- 11.84 percentage points higher Recall.

This trade-off shows that Naive Bayes behaves overly conservatively on class 1, predicting only 479 positives in total (`414 TP + 65 FP`) and missing 262 real disaster tweets. LinearSVC adopts a more sensitive decision boundary that recovers 80 additional true disaster tweets (+11.84 percentage points in Recall) at the cost of 67 extra false alarms (-7.52 percentage points in Precision).

Because the harmonic mean (F1-score) heavily penalizes the low Recall of Naive Bayes (`0.6124`), the substantial gain in Recall achieved by LinearSVC (`0.7308`) outweighs its moderate Precision reduction (`0.7891` vs. `0.8643`), raising the overall F1-score from `0.7169` to `0.7588`.

### 6.4 Qualitative Error Analysis

Inspection of the 132 False Positives and 182 False Negatives extracted by `src/analyze_errors.py` (including validation records such as IDs 53, 117, 7, and 328) highlights two main sources of error:
1. **Model and representation limitations:** Because unigram TF-IDF treats tokens independently without broader semantic context, tweets containing disaster-related keywords used figuratively, metaphorically, or in non-emergency contexts trigger False Positives. Conversely, genuine disaster tweets that rely on implicit context, abbreviations, or less frequent phrasing fail to accumulate sufficient positive weight, resulting in False Negatives.
2. **Annotation noise:** Several misclassified examples involve short or ambiguous tweets where the ground-truth label in the Kaggle dataset is itself questionable, requiring careful distinction between model failure and dataset label inconsistency.


---

## 7. Limitations and Threats to Validity

### 7.1 Dataset Limitations

Analysis of the training partition (`src/label_conflicts.py`) revealed **13 distinct tweet texts (associated with 41 rows, including 7 exact ties)** that carry conflicting ground-truth labels (`target = 0` and `target = 1` for the exact same text). This confirms the presence of annotation noise in the dataset, which imposes an upper bound on achievable classification accuracy and reliability. Additionally, the dataset represents a static snapshot of English-language tweets and may not generalize to other languages, regions, or evolving social media slang.

### 7.2 Feature Representation Limitations

Standard unigram `TfidfVectorizer` follows a bag-of-words assumption that discards word order, syntactic structure, and negation. It cannot distinguish literal hazard reports from figurative expressions sharing the same vocabulary, nor can it capture semantic similarity between synonymous words that do not share exact surface forms.

### 7.3 Evaluation Limitations

Our primary model comparison relies on a single held-out validation split of 1,530 tweets (`random_state=42`) and 3-fold internal cross-validation. Because no formal statistical hypothesis test was performed and the F1 margin between `LinearSVC` (`0.7588`) and `LogisticRegression` (`0.7522`) is small (`0.0066`), we cannot claim statistically significant superiority between these two models.

Additionally, while `GroupShuffleSplit` and `GroupKFold` eliminate **exact** string duplicates across partitions, they do not filter out **near-duplicates** (tweets with identical core text that differ only by a trailing URL, username mention, punctuation, or capitalization), which may still allow partial information leakage between training and validation sets.

### 7.4 Real-World Applicability

A text classifier trained on historical Kaggle tweets is an educational demonstration and **not** a production-ready disaster detection or emergency response system. Real-world emergency systems require multi-source verification, geolocation validation, calibrated probability thresholds tailored to operational triage capacity, continuous monitoring for concept drift, and human-in-the-loop oversight before any high-stakes action is taken.


---

## 8. Reproducibility

### 8.1 Environment

- Python: 3.11.9
- Dependencies: `requirements.txt`
- Main ML library: Scikit-learn
- Model persistence: Joblib
- Automated testing: Pytest

Pinning library versions in `requirements.txt` and fixing `random_state=42` across data splitting (`GroupShuffleSplit`) and model initialization (`LogisticRegression`, `LinearSVC`) ensure that partition assignments, vocabulary indices, and optimization trajectories are deterministic and reproducible across runs.

### 8.2 Dataset Preparation

Download the official dataset files (`train.csv`, `test.csv`, and `sample_submission.csv`) from the [Kaggle competition page](https://www.kaggle.com/competitions/nlp-getting-started/data) and place them inside the `data/raw/` directory at the root of the repository.

### 8.3 Reproducing the Experiments

From the repository root, with the Python environment activated:

```bash
python -m pip install -r requirements.txt
python -m src.compare_models
python -m src.select_model
python -m src.analyze_errors
python -m pytest -q
```

- `python -m pip install -r requirements.txt`: Installs all required Python packages.
- `python -m src.compare_models`: Runs the controlled validation comparison of `MultinomialNB`, `LogisticRegression`, and `LinearSVC` on the 1,530-sample validation split.
- `python -m src.select_model`: Executes 3-fold `GroupKFold` cross-validation on the 6,083 training records across the four candidate configurations.
- `python -m src.analyze_errors`: Extracts and displays the 132 False Positives and 182 False Negatives from the validation partition.
- `python -m pytest -q`: Runs the automated test suite.

Note: The original Kaggle CSV files in `data/raw/` are required to run the experiment scripts (`src.*`), whereas the unit test suite (`pytest`) runs on self-contained synthetic data (skipping only local artifact integration tests if `.joblib` models have not yet been generated).

### 8.4 Model Training and Persistence

```bash
python -m src.final_pipeline
python -m src.verify_saved_pipeline
python -m src.train_submission
```

- `src.final_pipeline` trains the **development Pipeline** on the 6,083-sample training split (`train_df`) and saves it to `models/linear_svm_pipeline.joblib`, allowing `src.verify_saved_pipeline` to verify that reloading the artifact reproduces the exact validation metrics (`F1 = 0.7588`) on the 1,530 held-out validation samples.
- `src.train_submission` trains the **full-data submission Pipeline** on all 7,613 labeled records in `train.csv` and saves it to `models/linear_svm_submission.joblib` for Kaggle submission generation (`src.generate_submission`) and the interactive web application.

### 8.5 Automated Verification

Latest verified test results:
- Tests passed: 34
- Tests failed: 0
- Execution time: 2.61 seconds

The 34 automated tests cover three verification layers:
1. **Unit tests:** Validate group-aware data splitting (`test_data_split.py`), TF-IDF vocabulary isolation (`test_text_pipeline.py`), conflict detection (`test_label_conflicts.py`), metric calculation (`test_evaluate_baseline.py`), model training/comparison (`test_train_logistic.py`, `test_compare_models.py`, `test_tune_logistic.py`, `test_select_model.py`), submission formatting (`test_train_submission.py`, `test_generate_submission.py`), and input sanitization/inference logic (`test_inference.py`).
2. **Artifact integration tests:** Verify that serialized `.joblib` pipelines preserve predictions after reloading and execute end-to-end inference (`test_final_pipeline.py` and real-pipeline inference tests).
3. **UI integration tests:** Use Streamlit `AppTest` (`test_streamlit_integration.py`) to simulate user interactions for positive, negative, and empty tweet submissions without opening a browser.

Passing all 34 software tests confirms code correctness, pipeline integrity, and protection against implementation bugs, but does not establish predictive generalization to new real-world tweets.

### 8.6 Public Artifacts

GitHub repository:
https://github.com/Guilherme5G/Disaster-tweet-classifier

Public Kaggle notebook:
https://www.kaggle.com/code/guiamoreira/disaster-tweets-tf-idf-linear-svm

Published model:
https://huggingface.co/GuiAraujo05/disaster-tweets-tfidf-linear-svm

Live demonstration:
https://huggingface.co/spaces/GuiAraujo05/disaster-tweet-classifier

These public artifacts enable independent inspection and end-to-end reproducibility: the GitHub repository provides the modular source code and automated test suite, the Kaggle notebook executes the complete training and submission workflow directly on cloud competition data, the Hugging Face Model Hub hosts the serialized `Pipeline` artifact, and the Hugging Face Space provides an interactive Streamlit interface for live testing.


---

## 9. Conclusions

### 9.1 Main Findings

Under a group-aware validation protocol (`GroupShuffleSplit`, 0 exact text overlap), discriminative linear classifiers (`LogisticRegression` with `F1 = 0.7522` and `LinearSVC` with `F1 = 0.7588`) outperformed the generative `MultinomialNB` baseline (`F1 = 0.7169`). `LinearSVC (C=1.0)` achieved the highest observed validation F1-score (`0.7588`), the highest validation Recall (`0.7308`), the highest internal 3-fold cross-validation mean F1 (`0.7460`), and a Kaggle Leaderboard F1-score of `0.79313` when refitted on all 7,613 training tweets.

### 9.2 Engineering Implications

This project demonstrated that trustworthy ML engineering requires more than optimizing a single metric: grouping by text (`df["text"]`) prevented duplicate-text leakage, encapsulating `TfidfVectorizer` inside a Scikit-learn `Pipeline` prevented preprocessing leakage across cross-validation folds and inference, and a 34-test automated suite ensured consistency across data processing, model persistence, and the deployed Streamlit interface.

### 9.3 Future Work

1. **Repeated or stratified group cross-validation:** Evaluate candidates across more folds and multiple random seeds to quantify variance and compute confidence intervals.
2. **Label conflict resolution:** Investigate strategies to handle the 13 conflicting text groups (41 rows) in the training set without leaking validation information.
3. **Near-duplicate detection:** Normalize URLs, user mentions, and punctuation before grouping to prevent near-duplicate leakage across partitions.
4. **Richer text representations:** Compare unigram TF-IDF against n-gram features (`ngram_range=(1, 2)`), character n-grams, or contextual embeddings (e.g., pretrained Transformer models) to better handle figurative language and context.
5. **Probability calibration and cost-sensitive thresholds:** Compare calibrated `LinearSVC` (`CalibratedClassifierCV`) and `LogisticRegression` across custom decision thresholds tailored to specific False Positive vs. False Negative operational costs.

### 9.4 Final Statement

Within the scope of the evaluated TF-IDF representations and linear classifiers on the Kaggle Disaster Tweets dataset, `LinearSVC (C=1.0)` provided the best observed balance between Precision and Recall (`F1 = 0.7588` on validation; `0.79313` on the Kaggle Leaderboard), though its small margin over `LogisticRegression` (`F1 = 0.7522`) does not constitute statistical proof of universal superiority.