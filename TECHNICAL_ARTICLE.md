# Beyond Accuracy: Building and Evaluating a Disaster Tweet Classifier with TF-IDF and LinearSVC

## 1. Introduction — Why Tweet Classification Matters

During unfolding crises, social media streams mix eyewitness reports of real hazards with jokes, news commentary, and everyday metaphors. A sentence like *"The sky is ablaze above the highway"* might describe a wildfire or a sunset. Separating literal disaster reports from figurative language is a text classification challenge because surface keywords alone rarely tell the full story.

In this project, we explore that problem using the public Kaggle **Natural Language Processing with Disaster Tweets** dataset (`train.csv` with 7,613 labeled tweets and `test.csv` with 3,263 unlabeled tweets).

Rather than claiming to build a real-world emergency detection system, our goal was to design a disciplined, reproducible Machine Learning engineering workflow: preventing subtle data leakage, comparing classifiers under a controlled protocol, analyzing Precision-Recall trade-offs, and packaging the selected Pipeline with automated tests and an interactive browser demonstration.

---

## 2. Dataset and the Data Leakage Problem

The labeled dataset contains **7,613 tweets** distributed across two classes: **4,342 non-disaster tweets (`target = 0`, 57.03%)** and **3,271 disaster tweets (`target = 1`, 42.97%)**.

To keep the Pipeline self-contained and evaluate the predictive signal of raw text alone, we used `text` as our sole input feature and excluded `id`, `keyword`, and `location`.

Early exploratory analysis revealed two important pitfalls that a naive row-level `train_test_split` could overlook:

1. **Duplicate tweets across rows:** While every row has a unique `id`, multiple rows share identical `text` strings. Splitting randomly by row—or grouping by `id`—can allow identical tweets to appear in both training and validation sets, potentially inflating validation scores through memorization. Inspecting the training partition (`src/label_conflicts.py`) also uncovered **13 distinct tweet texts spanning 41 rows, including 7 exact ties**, with conflicting ground-truth labels (`0` and `1` for identical texts). These inconsistencies introduce irreducible ambiguity for a deterministic text-only classifier when identical inputs have contradictory labels.

2. **Preprocessing leakage in TF-IDF:** Calling `vectorizer.fit_transform(df["text"])` before splitting allows validation tokens to influence the learned vocabulary and uses validation document frequencies to compute global IDF weights. Even if the classifier is subsequently fitted only on `X_train`, its feature representations have already been influenced by validation data.

To address these issues, we partitioned the dataset using `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`, grouped by `df["text"]`.

Because `GroupShuffleSplit` allocates approximately **20% of unique text groups** rather than 20% of individual rows, our resulting split contained **6,083 training rows** and **1,530 validation rows (approximately 20.10%)**, with **zero exact text overlap** between partitions.

We then fitted `TfidfVectorizer` exclusively on the 6,083 training tweets, learning a vocabulary of **18,621 features**, and transformed the validation set using `vectorizer.transform()` without refitting.

This prevented the identified exact-duplicate overlap and preprocessing leakage, although near-duplicate texts remained a potential limitation.

---

## 3. Three Models, One Evaluation Protocol

With a leakage-aware feature pipeline in place, we compared three classical text classifiers that represent different modeling assumptions over sparse TF-IDF vectors:

- **Multinomial Naive Bayes (`alpha=1.0`):** A generative baseline that estimates class-conditional feature distributions with additive smoothing under the assumption of conditional feature independence. Although traditionally associated with count-based representations, MultinomialNB can also operate on non-negative TF-IDF features.

- **Logistic Regression (`C=1.0`, `max_iter=1000`, `random_state=42`):** A discriminative linear classifier trained with regularized log-loss that produces probability estimates through the logistic function without assuming independent input features. Probability calibration was not evaluated separately.

- **LinearSVC (`C=1.0`, `max_iter=5000`, `random_state=42`):** A margin-based discriminative classifier that minimizes a regularized squared hinge loss, with training penalties concentrated on examples within or violating the classification margin.

Because non-disaster tweets account for 57.03% of the dataset, **Accuracy** alone can be misleading. An overly conservative model that frequently predicts class `0` may achieve reasonable overall Accuracy while missing a substantial proportion of disaster-related tweets.

We therefore selected the positive-class **F1-score**, the harmonic mean of Precision and Recall, as our primary evaluation metric. This also matches the official Kaggle competition metric.

For hyperparameter exploration and internal model selection (`src/tune_logistic.py` and `src/select_model.py`), we performed **3-fold `GroupKFold` cross-validation** within the 6,083-row training partition, placing `TfidfVectorizer` and the classifier inside a Scikit-learn `Pipeline`.

This ensured that vocabulary and IDF statistics were learned independently within each training fold and that identical text groups did not cross the internal fold boundaries.

Among the four candidate configurations evaluated during internal model selection, **LinearSVC (`C=1.0`) achieved the highest observed mean cross-validation F1-score (`0.7471`)**.

---

## 4. Results: What the Numbers Actually Show

Evaluating the three primary classifiers on the same validation partition of 1,530 tweets revealed different Precision-Recall trade-offs.

| Metric | Naive Bayes (`alpha=1.0`) | Logistic Regression (`C=1.0`) | LinearSVC (`C=1.0`) |
|---|---:|---:|---:|
| **Accuracy** | 0.7863 | 0.7954 | 0.7948 |
| **Precision** | **0.8643** | 0.8092 | 0.7891 |
| **Recall** | 0.6124 | 0.7027 | **0.7308** |
| **F1-score** | 0.7169 | 0.7522 | **0.7588** |
| **True Negatives (TN)** | 789 | 742 | 722 |
| **False Positives (FP)** | 65 | 112 | 132 |
| **False Negatives (FN)** | 262 | 201 | 182 |
| **True Positives (TP)** | 414 | 475 | 494 |

Looking beyond individual metrics helps explain the behavior of each classifier.

**Why Accuracy and F1-score disagree**

Logistic Regression achieved slightly higher Accuracy than LinearSVC, with **20 additional True Negatives but 19 fewer True Positives**.

This resulted in just **one additional correct prediction across 1,530 validation samples**.

However, LinearSVC achieved higher Recall and F1-score for the positive class, making it the preferred configuration under our primary evaluation criterion.

**The Precision-Recall trade-off**

Naive Bayes showed the most conservative behavior when predicting class `1`, achieving high Precision (`0.8643`) with only 65 False Positives, but missing 262 disaster-related tweets (`Recall = 0.6124`).

Compared with Naive Bayes, LinearSVC produced:

- **80 fewer False Negatives** (`262 → 182`).
- **67 additional False Positives** (`65 → 132`).
- **11.84 percentage points higher Recall**.

The increase in Recall more than compensated for the reduction in Precision under the F1-score calculation, raising F1 from `0.7169` to `0.7588`.

This illustrates why selecting a classifier requires considering the trade-off between missed positive cases and false alarms rather than maximizing a single metric in isolation.

**Error analysis**

Running `src/analyze_errors.py` identified **132 False Positives and 182 False Negatives** for LinearSVC.

Figurative disaster-related vocabulary, ambiguous phrasing, context-dependent tweets, and annotation inconsistencies are plausible explanations for these errors. However, their individual contributions were not systematically quantified, so they should not be treated as confirmed causes.

**Official Kaggle submission**

After model selection, we refitted the TF-IDF + LinearSVC Pipeline using all **7,613 labeled tweets** and generated predictions for the **3,263 competition test tweets**.

The official submission achieved a **Kaggle Leaderboard F1-score of `0.79313`**.

This result should not be directly compared with the local validation F1-score of `0.7588`.

The submission model was trained on all labeled observations, while the local development model used only 6,083 training records. The evaluation datasets also differ, and the Kaggle competition split may not enforce the same exact-text group separation.

These scores represent distinct evaluation settings, not evidence that the model automatically generalizes better in real-world environments.

---

## 5. From Experiment to Deployment

To move beyond notebook experimentation, we organized the project into reusable Python modules, persisted trained Pipelines, implemented automated tests, and built separate local and browser-based interfaces.

**Reproducible Pipelines**

We serialized two trained configurations:

- **Development Pipeline:** `models/linear_svm_pipeline.joblib`, trained on 6,083 observations and verified through `src/verify_saved_pipeline.py` to reproduce the validation F1-score of `0.7588`.
- **Submission Pipeline:** `models/linear_svm_submission.joblib`, trained on all 7,613 labeled observations for competition submission and application inference.

Keeping `TfidfVectorizer` and `LinearSVC` together inside a Scikit-learn `Pipeline` ensures that the saved classifier uses its corresponding learned vocabulary, IDF statistics, and feature transformation during inference.

**34 Automated Tests**

Our Pytest suite completed with **34 tests passed in 2.61 seconds** in the verified local environment.

These tests cover:

- Group-aware data splitting and vocabulary isolation.
- Label-conflict detection and classification metrics.
- Model training, selection, persistence, and inference.
- Input validation, including empty, invalid, long, and Unicode text.
- Integration with the serialized Joblib Pipeline.
- Streamlit UI interactions using `AppTest`, including positive predictions, negative predictions, and empty-input warnings.

In a fresh environment, the local submission model must first be generated or obtained from a trusted source before running integration tests that depend on its `.joblib` artifact.

Passing these tests provides evidence that the tested software components behave as intended. It does not establish predictive generalization to unseen social media streams.

**Local and Browser-Based Interfaces**

For local interactive use, we developed a **Streamlit** application (`streamlit_app.py`) that loads the serialized Python Pipeline and performs predictions using the existing inference module.

For public access, we deployed a separate **static HTML/CSS/JavaScript application on Hugging Face Spaces**.

The browser implementation performs TF-IDF transformation and linear classification using parameters exported from the trained Pipeline, without requiring a Python inference backend.

This separation allowed us to demonstrate the model publicly while preserving a reusable Python inference workflow for development and testing.

---

## 6. Limitations and Key Lessons

Interpreting these findings responsibly requires recognizing what the experiments do **not** establish.

**1. No demonstrated statistical superiority**

Although LinearSVC achieved the highest observed validation F1-score (`0.7588`), its margin over Logistic Regression (`0.7522`) was only `0.0066`.

No formal statistical significance test was conducted to establish whether this difference reflects a reliable performance advantage beyond the evaluated data.

Furthermore, although the held-out validation partition was excluded from internal cross-validation tuning, its metrics were inspected during development. Therefore, it should not be considered a completely untouched final test set.

**2. Near-duplicate leakage remains possible**

Grouping by exact `df["text"]` values eliminated exact text overlap between training and validation partitions.

However, tweets differing only in URLs, username mentions, punctuation, or capitalization may remain separate groups and could introduce near-duplicate information overlap.

Further normalization and similarity analysis would be required to investigate this risk.

**3. Limitations of bag-of-words representations**

Unigram TF-IDF ignores word order and does not explicitly model the scope of negation or broader discourse context.

Although individual negation words may appear as features, the representation does not directly encode how those words modify the meaning of surrounding phrases.

This limits its ability to represent compositional meaning, metaphors, sarcasm, and context-dependent language.

**4. Not a real-world emergency detection system**

A classifier trained on a static historical Kaggle dataset is an educational benchmark, not a production-ready crisis-response tool.

Real-world applications would require additional independent evaluation, multi-source verification, geographic context, decision-threshold assessment, human oversight, and continuous monitoring for changes in the data distribution.

The observed benchmark results alone are insufficient to justify high-stakes operational use.

---

## 7. Conclusion and Resources

Building a reliable Machine Learning baseline involves more than optimizing leaderboard scores.

Throughout this project, we investigated repeated texts and conflicting annotations, implemented group-aware data splitting, prevented preprocessing leakage, compared multiple classification approaches, analyzed confusion matrices, and verified important software behaviors through automated tests.

Within the evaluated configurations and validation protocol, **TF-IDF combined with LinearSVC (`C=1.0`) achieved the highest observed F1-score**, with:

- **0.7471** mean F1-score in internal 3-fold group-aware cross-validation.
- **0.7588** F1-score on the local validation partition.
- **0.79313** F1-score on the official Kaggle Leaderboard after retraining on all labeled observations.

These results supported our engineering decision to select LinearSVC, but they do not demonstrate universal or statistically significant superiority over alternative algorithms.

The main lesson is that **reproducibility, leakage prevention, meaningful evaluation, and honest interpretation of limitations are as important as model selection itself**.

The project's source code, experimental report, published model, and interactive demonstration are available through the following resources:

- **GitHub Repository:** https://github.com/Guilherme5G/Disaster-tweet-classifier
- **Full Experimental Report:** https://github.com/Guilherme5G/Disaster-tweet-classifier/blob/main/EXPERIMENT_REPORT.md
- **Public Kaggle Notebook:** https://www.kaggle.com/code/guiamoreira/disaster-tweets-tf-idf-linear-svm
- **Published Model (Hugging Face Hub):** https://huggingface.co/GuiAraujo05/disaster-tweets-tfidf-linear-svm
- **Live Browser Demo (Hugging Face Spaces):** https://huggingface.co/spaces/GuiAraujo05/disaster-tweet-classifier

---

**Author:** Guilherme de Araújo Moreira
