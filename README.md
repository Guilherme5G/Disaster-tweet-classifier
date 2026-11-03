# Disaster Tweet Classifier

A reproducible Machine Learning project for binary text classification, developed using the Kaggle **Natural Language Processing with Disaster Tweets** competition.

## Objective

Classify whether a tweet refers to a real disaster (1) or not (0).

The project focuses on building a reliable text classification pipeline, investigating data quality issues, preventing data leakage, training baseline models, and evaluating predictive performance.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Pytest
- Git and GitHub

## Project Status

Work in progress.

- [x] Python virtual environment setup
- [x] Dataset download and loading
- [x] Initial data exploration
- [x] Data quality analysis
- [x] Group-aware train/validation splitting
- [x] TF-IDF text vectorization
- [x] Mathematical foundations of Naive Bayes and Logistic Regression
- [x] Investigation of conflicting target labels
- [x] Baseline model training with Multinomial Naive Bayes
- [x] Baseline evaluation using Accuracy, Precision, Recall and F1-score
- [x] Confusion matrix and error analysis
- [x] Automated tests for data processing and evaluation
- [ ] Model comparison and hyperparameter experiments
- [ ] Kaggle submission
- [ ] Hugging Face deployment

## Dataset

Source: [Kaggle - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

Download the dataset and place `train.csv` inside:

`data/raw/train.csv`

The dataset files are not tracked by Git and must be downloaded separately.

The training dataset contains 7,613 observations and five columns:

- `id`: Unique tweet identifier.
- `keyword`: Keyword associated with the tweet, when available.
- `location`: Location information, when available.
- `text`: Original tweet content.
- `target`: Binary classification label.

**Target interpretation:**

- `0`: Not a real disaster.
- `1`: Real disaster.

The current baseline uses only the `text` column as input features and `target` as the prediction target.

## Current Implementation

The project currently includes the following modules:

### Data Exploration and Mathematical Foundations

- `src/explore.py`: Loads the training dataset and performs exploratory data analysis, including missing values, duplicate detection, label conflicts, class distribution, and consistency checks.
- `src/text_basics.py`: Demonstrates basic tokenization and Bag-of-Words word frequencies.
- `src/tfidf_basics.py`: Demonstrates TF-IDF vectorization using a small example corpus.
- `src/classifier_math_basics.py`: Demonstrates the mathematical foundations of Naive Bayes and Logistic Regression through numerical examples.

### Data Processing

- `src/data_split.py`: Implements reusable group-aware training and validation splitting using `GroupShuffleSplit`.
- `src/text_pipeline.py`: Implements reusable TF-IDF text vectorization, fitting the vocabulary exclusively on training data.
- `src/label_conflicts.py`: Investigates conflicting target labels in the training split, including class counts and tied labels.

### Model Training and Evaluation

- `src/train_baseline.py`: Trains the initial Multinomial Naive Bayes classifier and generates predictions for the validation dataset.
- `src/evaluate_baseline.py`: Evaluates the classifier using Accuracy, Precision, Recall, F1-score, a confusion matrix, and a classification report.

The evaluation module provides a reusable `evaluate_predictions()` function that calculates classification metrics independently of the model training procedure.

### Automated Tests

- `tests/test_data_split.py`: Verifies that identical texts remain in the same data split.
- `tests/test_text_pipeline.py`: Tests feature dimensionality, vocabulary integrity, and vocabulary stability.
- `tests/test_label_conflicts.py`: Validates conflicting-label detection using synthetic data.
- `tests/test_evaluate_baseline.py`: Validates classification metrics and confusion matrix values using known examples.

## Exploratory Data Analysis

### Dataset Overview

The dataset contains 7,613 observations and five columns, designed for binary classification of disaster-related tweets.

The target distribution is moderately imbalanced:

| Target | Samples | Percentage |
|---|---:|---:|
| Non-disaster (0) | 4,342 | 57.03% |
| Disaster (1) | 3,271 | 42.97% |
| Total | 7,613 | 100% |

Because the classes are not perfectly balanced, accuracy alone is insufficient to evaluate the classifier.

Precision, Recall, F1-score, and the confusion matrix are also considered.

### Data Quality Findings

The exploratory analysis identified the following characteristics:

- 2,533 missing values in `location` (33.27%).
- 61 missing values in `keyword` (0.80%).
- No missing values in `text` or `target`.
- No fully duplicated rows.
- 110 additional occurrences of previously seen text.
- 18 distinct texts with conflicting target labels.
- No empty or whitespace-only texts.
- No duplicated IDs.

These findings highlight potential annotation inconsistencies and repeated observations that require careful consideration during preprocessing and validation.

### Conflicting Target Labels

Some identical tweet texts appear with different target labels.

For example, the same text may be classified as a disaster in one observation and as a non-disaster in another.

This creates an annotation inconsistency because identical input features may correspond to contradictory target values.

The complete dataset contains 18 distinct texts associated with conflicting target labels.

A separate investigation was performed using only the training split.

**Observed training results:**

- 13 distinct texts with conflicting labels.
- 41 training observations associated with these conflicts.
- 7 conflicting texts with tied class counts.
- 6 conflicting texts with unequal class counts.

The initial baseline preserves the original training labels, including conflicting observations.

This decision maintains a consistent reference experiment before investigating alternative label-cleaning strategies.

No automatic relabeling or removal of conflicting observations was applied.

## Data Splitting and Text Vectorization

### Group-Aware Splitting

The dataset is split using Scikit-learn's `GroupShuffleSplit` with:

- `n_splits=1`
- `test_size=0.2`
- `random_state=42`
- Group identifier: exact tweet text

The validation fraction is applied to distinct groups rather than directly to individual rows.

Exact duplicate texts are assigned to the same split to prevent identical messages from appearing in both training and validation.

**Observed results:**

| Metric | Training | Validation |
|---|---:|---:|
| Samples | 6,083 | 1,530 |
| Class 0 | 57.34% | 55.82% |
| Class 1 | 42.66% | 44.18% |

The observed intersection of exact text values between the two splits is zero.

Class distributions are inspected after splitting because `GroupShuffleSplit` does not guarantee class stratification.

### TF-IDF Vectorization

Text features are extracted using Scikit-learn's `TfidfVectorizer`.

The vectorizer is fitted exclusively on the training texts using `fit_transform()`.

Validation texts are processed using `transform()`, preserving the vocabulary and IDF values learned during training.

This prevents validation data from influencing the fitted text representation.

**Feature matrix dimensions:**

- Training: `(6083, 18621)`
- Validation: `(1530, 18621)`
- Learned vocabulary: 18,621 terms

Both matrices use the same feature space, with identical vocabulary ordering.

The feature matrices are kept in sparse format to avoid unnecessary memory consumption.

The current text classification pipeline uses only the `text` column.

## Baseline Model

### Multinomial Naive Bayes

The first supervised learning baseline uses Scikit-learn's `MultinomialNB`.

**Configuration:**

- Algorithm: Multinomial Naive Bayes
- `alpha=1.0`
- Input representation: TF-IDF
- Training samples: 6,083
- Validation samples: 1,530
- Features: 18,621

Multinomial Naive Bayes was selected as an initial baseline because it is computationally efficient and commonly used for text classification tasks.

The classifier estimates class-related feature probabilities and uses them to generate predictions.

The `alpha=1.0` parameter applies additive smoothing, preventing zero estimated probabilities for features that were not observed in a particular class during training.

The model is trained exclusively using the training features and labels.

The validation dataset is not used in model fitting.

### Training Workflow

The baseline follows these steps:

1. Load the original training dataset.
2. Split the data using `GroupShuffleSplit`.
3. Fit the TF-IDF vectorizer using training texts only.
4. Transform training and validation texts into numerical feature matrices.
5. Train `MultinomialNB(alpha=1.0)` using the training dataset.
6. Generate predictions for the validation dataset.
7. Compare predictions with the true validation labels.
8. Calculate classification metrics and analyze prediction errors.

This workflow separates model training from evaluation and prevents direct leakage of validation information into the fitted classifier.

## Evaluation Results

### Baseline Performance

The initial Multinomial Naive Bayes classifier produced the following results on the local validation dataset:

| Metric | Score |
|---|---:|
| Accuracy | 0.7863 |
| Precision (class 1) | 0.8643 |
| Recall (class 1) | 0.6124 |
| F1-score (class 1) | 0.7169 |

These values were obtained from the local validation split.

They are **not Kaggle leaderboard scores**.

### Confusion Matrix

The confusion matrix is organized with true labels as rows and predicted labels as columns.

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 789 (TN) | 65 (FP) |
| Actual 1 | 262 (FN) | 414 (TP) |

**Interpretation:**

- True Negatives (TN): 789 non-disaster tweets correctly classified.
- False Positives (FP): 65 non-disaster tweets incorrectly classified as disasters.
- False Negatives (FN): 262 disaster tweets incorrectly classified as non-disasters.
- True Positives (TP): 414 disaster tweets correctly identified.

The confusion matrix contains all 1,530 validation observations.

### Performance Interpretation

The classifier achieved a precision of 86.43%, indicating that most tweets predicted as disasters were correctly classified.

However, recall was substantially lower at 61.24%.

This means the model identified approximately six out of every ten disaster-related tweets in the validation dataset.

Among the 676 actual disaster tweets:

- 414 were correctly identified.
- 262 were incorrectly classified as non-disasters.

Approximately 38.76% of the disaster-related tweets were missed.

This suggests that the initial classifier is relatively conservative when predicting the positive class.

For applications involving emergency monitoring, false negatives may be particularly concerning because relevant events could remain undetected.

However, the relative costs of false positives and false negatives depend on the intended application.

### F1-score Interpretation

The F1-score combines Precision and Recall using their harmonic mean.

The baseline achieved an F1-score of 0.7169 for the positive class.

This provides a useful reference point for future model comparisons.

A higher F1-score does not automatically imply that a classifier is suitable for every application, because different applications may assign different costs to false positives and false negatives.

The Kaggle competition uses the F1-score as its evaluation metric.

## Known Limitations

### Conflicting Labels

The original dataset contains 18 distinct texts with contradictory target labels.

Within the training split, 13 distinct conflicting texts were identified, involving 41 observations.

These inconsistencies were preserved in the initial baseline rather than automatically corrected.

Future experiments may investigate alternative strategies for handling conflicting annotations.

### False Negatives

The model incorrectly classified 262 actual disaster tweets as non-disasters.

Its recall of 0.6124 indicates that a substantial proportion of positive observations remains undetected.

Reducing false negatives is a potential objective for future experiments, depending on the intended use case.

### Limited Model Comparison

The current baseline uses only Multinomial Naive Bayes.

A systematic comparison with Logistic Regression and other classifiers has not yet been completed.

Therefore, the current results do not establish that Multinomial Naive Bayes is the best algorithm for this dataset.

### Validation Strategy

The current experiment uses one fixed group-aware train/validation split.

This provides a consistent reference for the initial baseline but does not measure performance variability across multiple splits.

Repeatedly selecting models or hyperparameters using the same validation set can also lead to overfitting to that validation set.

Future experiments should consider group-aware cross-validation within the training data and independent final evaluation.

### Potential Near-Duplicate Leakage

Group-aware splitting prevents exact tweet text overlap between training and validation.

However, it does not guarantee protection against all forms of data leakage.

Near-duplicate texts, paraphrases, or other highly similar observations may still appear in different splits.

### Feature Limitations

The initial model uses only tweet text.

The `keyword` and `location` columns are not included in the feature representation.

Additional features may be investigated in future experiments.

### Generalization

The reported scores describe performance on the selected local validation dataset.

They do not establish performance on unseen Kaggle test data or in a real-world production environment.

## Automated Tests

The project uses Pytest to verify important properties of data splitting, text vectorization, label-conflict analysis, and model evaluation.

The current test suite covers:

- Group-aware separation of identical texts.
- Compatible training and validation feature dimensions.
- Vocabulary restricted to training data.
- Vocabulary stability when transforming validation texts.
- Correct detection of conflicting target labels.
- Correct classification metrics and confusion matrix values using synthetic examples.

The tests use small synthetic datasets to make expected behavior predictable and reduce dependence on external files.

Floating-point metrics are compared using `pytest.approx()` to account for numerical precision.

The evaluation implementation also checks:

- Matching numbers of true labels and predictions.
- Valid predicted class labels.
- Consistency between confusion matrix counts and sample totals.
- Agreement between Scikit-learn's F1-score and its mathematical formula.

**Latest verified execution:**

- Python: 3.11.9
- Pytest: 9.2.0
- Tests collected: 6
- Tests passed: 6
- Tests failed: 0

These tests validate specific implementation properties.

They do not prove that the classifier generalizes well or that every possible source of data leakage has been eliminated.

## Reproducibility

### Requirements

The project uses Python and the dependencies listed in `requirements.txt`.

The latest verified execution used Python 3.11.9.

For consistent results, use the project's documented dependency versions and the same dataset.

### Environment Setup

Clone the repository:

```bash
git clone https://github.com/Guilherme5G/Disaster-tweet-classifier.git
cd Disaster-tweet-classifier
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows using Git Bash:

```bash
source .venv/Scripts/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

### Dataset Setup

Download the dataset from the Kaggle competition:

https://www.kaggle.com/competitions/nlp-getting-started

Place `train.csv` inside:

```text
data/
└── raw/
    └── train.csv
```

The dataset is excluded from Git version control.

### Running Data Exploration

From the project root:

```bash
python src/explore.py
```

### Running the Data Split

```bash
python -m src.data_split
```

### Running TF-IDF Vectorization

```bash
python -m src.text_pipeline
```

### Investigating Label Conflicts

```bash
python -m src.label_conflicts
```

### Training the Baseline

```bash
python -m src.train_baseline
```

This command trains the Multinomial Naive Bayes classifier and generates predictions for the validation dataset.

### Evaluating the Baseline

```bash
python -m src.evaluate_baseline
```

This command executes the training and evaluation workflow and prints:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- True Negative, False Positive, False Negative, and True Positive counts
- Classification report

### Running Automated Tests

```bash
python -m pytest -v
```

Run these commands from the repository root with the virtual environment activated.

The reported baseline metrics were obtained using the original training dataset, the specified group-aware split, and the current TF-IDF and Multinomial Naive Bayes configurations.

A fresh-environment installation has not yet been independently verified.

## Engineering Implications

The project prioritizes reproducibility, separation of responsibilities, data integrity, and reliable model evaluation.

The data splitting, text vectorization, label-conflict investigation, and classification evaluation logic are organized into reusable functions.

This allows important parts of the pipeline to be tested independently using synthetic datasets.

Repeated texts are grouped during splitting to prevent exact overlapping examples from producing potentially optimistic validation results.

The TF-IDF vectorizer is fitted exclusively on training data to avoid incorporating validation information into the learned representation.

The baseline model provides a measurable reference for comparing future classification approaches.

The evaluation results demonstrate why accuracy alone is insufficient to understand predictive performance.

Although the classifier achieved relatively high precision, its lower recall and 262 false negatives reveal an important limitation.

Future experiments will investigate alternative classifiers, hyperparameters, decision thresholds, and data-quality strategies.

Model selection should be performed carefully to avoid repeatedly optimizing decisions against the same validation dataset.

The project will continue documenting experimental configurations, results, limitations, and testing decisions to support reproducibility and meaningful model comparisons.

## Next Steps

- Compare Multinomial Naive Bayes against Logistic Regression.
- Investigate model hyperparameters and classification thresholds.
- Evaluate strategies for handling conflicting labels.
- Introduce more robust group-aware model-selection procedures.
- Generate predictions for Kaggle's test dataset.
- Submit predictions to the Kaggle competition.
- Prepare a deployable inference workflow.
- Deploy an interactive demonstration using Hugging Face.
- Document experimental comparisons and final conclusions.

## Author

Guilherme de Araújo Moreira
