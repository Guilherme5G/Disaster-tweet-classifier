# Disaster Tweet Classifier

A reproducible Machine Learning project for binary text classification, developed using the Kaggle **Natural Language Processing with Disaster Tweets** competition.

## Objective

Classify whether a tweet refers to a real disaster (1) or not (0).

The project focuses on building a reliable text classification pipeline, investigating data quality issues, preventing data leakage, training and comparing classification models, optimizing hyperparameters, and evaluating predictive performance.

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
- [x] Logistic Regression model training
- [x] Automated comparison between MultinomialNB and Logistic Regression
- [x] Group-aware cross-validation with GroupKFold
- [x] Hyperparameter optimization using GridSearchCV
- [x] Automated tests for preprocessing, evaluation, model training, comparison, and tuning
- [ ] Independent final model evaluation
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

The current classification experiments use only the `text` column as input features and `target` as the prediction target.

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
- `src/evaluate_baseline.py`: Evaluates predictions using Accuracy, Precision, Recall, F1-score, a confusion matrix, and a classification report.
- `src/train_logistic.py`: Implements reusable Logistic Regression training and evaluates the classifier using the existing TF-IDF workflow.
- `src/compare_models.py`: Trains and compares MultinomialNB and Logistic Regression using the same data split and TF-IDF representation.
- `src/tune_logistic.py`: Performs group-aware hyperparameter optimization using `Pipeline`, `GroupKFold`, and `GridSearchCV`.

The evaluation module provides a reusable `evaluate_predictions()` function that calculates classification metrics independently of the model training procedure.

The Logistic Regression module provides a reusable `train_logistic_model()` function.

The comparison workflow uses these existing functions to avoid unnecessary duplication and maintain consistent evaluation procedures.

### Automated Tests

- `tests/test_data_split.py`: Verifies that identical texts remain in the same data split.
- `tests/test_text_pipeline.py`: Tests feature dimensionality, vocabulary integrity, and vocabulary stability.
- `tests/test_label_conflicts.py`: Validates conflicting-label detection using synthetic data.
- `tests/test_evaluate_baseline.py`: Validates classification metrics and confusion matrix values using known examples.
- `tests/test_train_logistic.py`: Tests Logistic Regression training and prediction behavior.
- `tests/test_compare_models.py`: Tests the model comparison workflow and verifies metric consistency.
- `tests/test_tune_logistic.py`: Tests group integrity during cross-validation and verifies the hyperparameter search procedure.

## Exploratory Data Analysis

### Dataset Overview

The dataset contains 7,613 observations and five columns, designed for binary classification of disaster-related tweets.

The target distribution is moderately imbalanced:

| Target | Samples | Percentage |
|---|---:|---:|
| Non-disaster (0) | 4,342 | 57.03% |
| Disaster (1) | 3,271 | 42.97% |
| Total | 7,613 | 100% |

Because the classes are not perfectly balanced, accuracy alone is insufficient to evaluate the classifiers.

Precision, Recall, F1-score, and confusion matrices are also considered.

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

The initial experiments preserve the original training labels, including conflicting observations.

This decision maintains consistent reference experiments before investigating alternative label-cleaning strategies.

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

For the initial model training and comparison workflows, the vectorizer is fitted exclusively on the training texts using `fit_transform()`.

Validation texts are processed using `transform()`, preserving the vocabulary and IDF values learned during training.

This prevents validation data from influencing the fitted text representation.

**Feature matrix dimensions:**

- Training: `(6083, 18621)`
- Validation: `(1530, 18621)`
- Learned vocabulary: 18,621 terms

Both matrices use the same feature space, with identical vocabulary ordering.

The feature matrices are kept in sparse format to avoid unnecessary memory consumption.

The current classification workflow uses only the `text` column.

### TF-IDF During Cross-Validation

The hyperparameter optimization workflow uses a different approach to prevent internal cross-validation leakage.

Instead of vectorizing all training texts before cross-validation, `TfidfVectorizer` is included inside a Scikit-learn `Pipeline`.

This ensures that vocabulary and IDF statistics are fitted exclusively on the training portion of each cross-validation fold.

The validation portion of each fold is transformed using the representation learned from that fold's training data.

This separation is necessary because fitting TF-IDF on all 6,083 training observations before internal cross-validation would expose the training representation to information from the internal validation folds.

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

## Model Comparison

### Experimental Setup

Two classifiers were evaluated using the same group-aware train/validation split and TF-IDF feature representation.

The comparison included:

- Multinomial Naive Bayes with `alpha=1.0`.
- Logistic Regression with `C=1.0`, `max_iter=1000`, and `random_state=42`.

Both models were trained using the same 6,083 training observations and evaluated against the same 1,530 validation observations.

The comparison workflow is implemented in `src/compare_models.py`, which reuses the existing preprocessing and evaluation components.

Both models receive the same TF-IDF matrices, allowing their predictive behavior to be compared under consistent experimental conditions.

### Validation Results

| Metric | MultinomialNB | Logistic Regression |
|---|---:|---:|
| Accuracy | 0.7863 | 0.7954 |
| Precision (class 1) | 0.8643 | 0.8092 |
| Recall (class 1) | 0.6124 | 0.7027 |
| F1-score (class 1) | 0.7169 | 0.7522 |
| True Negatives | 789 | 742 |
| False Positives | 65 | 112 |
| False Negatives | 262 | 201 |
| True Positives | 414 | 475 |

### Comparison Interpretation

Logistic Regression achieved a higher F1-score, recall, and accuracy than Multinomial Naive Bayes on the selected validation split.

Compared with the Naive Bayes baseline, Logistic Regression:

- Increased the positive-class F1-score by approximately 0.0353.
- Increased recall by approximately 0.0902.
- Correctly identified 61 additional disaster-related tweets.
- Reduced false negatives from 262 to 201.
- Increased false positives from 65 to 112.

These findings demonstrate a trade-off between precision and recall.

Logistic Regression identified more actual disasters but generated more false alarms.

For an emergency-monitoring application, reducing false negatives could be valuable when missed emergencies have higher consequences than false alarms.

However, the appropriate balance depends on the operational context and the relative costs of each error type.

The observed improvement cannot be attributed exclusively to one mathematical difference between the algorithms.

Naive Bayes and Logistic Regression differ in their learning objectives, assumptions, parameter estimation, and regularization mechanisms.

### Comparison Limitations

These results were obtained using one fixed local validation split.

They do not establish that Logistic Regression will consistently outperform Naive Bayes on unseen data.

The comparison also does not provide statistical evidence of superiority across different data partitions.

The Logistic Regression configuration used in this initial comparison was `C=1.0`.

Subsequent hyperparameter optimization was performed separately using internal cross-validation.

## Hyperparameter Optimization

### Objective

A group-aware cross-validation procedure was implemented to investigate the effect of regularization strength on Logistic Regression performance.

The experiment searches for the value of `C` that maximizes the mean F1-score across internal validation folds.

The `C` parameter controls the inverse of regularization strength.

Smaller values apply stronger regularization, while larger values apply weaker regularization.

Changing `C` can influence model complexity, generalization, precision, and recall.

However, weaker regularization does not guarantee better performance.

### Methodology

Hyperparameter optimization is implemented in `src/tune_logistic.py` using:

- `Pipeline` with `TfidfVectorizer` and `LogisticRegression`.
- `GroupKFold(n_splits=3)`.
- `GridSearchCV`.
- Candidate values `C = [0.1, 1.0, 10.0]`.
- `scoring="f1"`.
- `refit=True`.
- `max_iter=1000`.
- `random_state=42` for Logistic Regression.

Only the original training partition, containing 6,083 observations, was used during hyperparameter selection.

Tweet texts were used as group identifiers to keep identical texts together during internal cross-validation.

### Group-Aware Cross-Validation

The experiment uses `GroupKFold` with three folds.

During each iteration:

1. One fold is reserved for internal validation.
2. The remaining folds are used for training.
3. The TF-IDF vectorizer is fitted using only the training folds.
4. Logistic Regression is trained on the resulting feature representation.
5. Predictions are generated for the internal validation fold.
6. F1-score is calculated using the true labels and predictions.

The process is repeated for each candidate value of `C`.

The mean F1-score across the three folds is used to select the best configuration.

`GroupKFold` prevents identical text groups from appearing simultaneously in the training and validation portions of the same fold.

It does not guarantee identical class distributions across folds.

### Pipeline and Data Leakage Prevention

The TF-IDF vectorizer is placed inside the Scikit-learn `Pipeline`.

This ensures that the vocabulary and IDF statistics are learned exclusively from the training portion of each fold.

Without this separation, fitting TF-IDF on the complete training partition before cross-validation would allow internal validation texts to influence the representation used during training.

The original 1,530-observation validation partition was not used during hyperparameter selection.

### GridSearchCV Configuration

The search evaluates three candidate values:

- `C=0.1`
- `C=1.0`
- `C=10.0`

Each configuration is evaluated using three cross-validation folds.

This produces nine internal cross-validation fits.

Because `refit=True`, the best configuration is subsequently fitted again using all 6,083 original training observations.

This final refit uses the entire training partition but does not include the original validation partition.

### Cross-Validation Results

| C | Mean F1-score | Standard Deviation |
|---|---:|---:|
| 0.1 | 0.5964 | 0.0109 |
| 1.0 | 0.7383 | 0.0143 |
| 10.0 | 0.7446 | 0.0073 |

**Selected hyperparameter:** `C=10.0`

**Best mean cross-validation F1-score:** `0.7446`

Among the evaluated candidates, `C=10.0` achieved the highest mean F1-score.

The improvement over `C=1.0` was:

`0.7446 - 0.7383 = 0.0063`

This corresponds to approximately 0.63 percentage points.

### Experimental Interpretation

The configuration with `C=0.1` achieved the lowest mean F1-score in the tested grid.

This suggests that stronger regularization was less suitable for this particular experiment.

The configurations `C=1.0` and `C=10.0` produced substantially higher mean F1-scores.

Although `C=10.0` performed best among the tested candidates, the observed improvement over `C=1.0` was small.

The experiment used only three folds, and the reported standard deviations describe variability across those folds.

These results do not establish that `C=10.0` will consistently generalize better to unseen tweets.

### Internal Cross-Validation vs. Original Validation

The best mean cross-validation F1-score of `0.7446` was obtained using internal folds from the original training partition.

It must not be confused with the F1-score obtained on the original validation partition.

The initial Logistic Regression configuration with `C=1.0` achieved an F1-score of `0.7522` on the original validation partition.

These values come from different evaluation procedures and should not be directly interpreted as a head-to-head comparison of `C=1.0` and `C=10.0`.

The tuned model with `C=10.0` has not yet been evaluated on the original validation partition.

### Hyperparameter Optimization Limitations

The search was limited to three candidate values and three internal folds.

Therefore, `C=10.0` is the best configuration among the tested candidates, not necessarily the globally optimal value.

The original validation partition was already used during earlier model comparisons.

Although it remained outside the hyperparameter search, it is not a completely untouched final test set for the entire project.

Future experiments should consider additional candidates, robust evaluation procedures, and an independent final test strategy.

## Known Limitations

### Conflicting Labels

The original dataset contains 18 distinct texts with contradictory target labels.

Within the training split, 13 distinct conflicting texts were identified, involving 41 observations.

These inconsistencies were preserved rather than automatically corrected.

Future experiments may investigate alternative strategies for handling conflicting annotations.

### False Negatives

The initial Naive Bayes baseline incorrectly classified 262 actual disaster tweets as non-disasters.

Logistic Regression reduced this number to 201 on the same validation partition.

Although this represents an improvement, a substantial number of disaster-related tweets were still missed.

Reducing false negatives remains a potential objective depending on the intended use case.

### Limited Model Coverage

The project has evaluated Multinomial Naive Bayes and Logistic Regression.

Both classifiers use the same TF-IDF text representation.

Additional algorithms, text representations, and feature engineering strategies have not yet been systematically evaluated.

The current results do not establish which approach is best across different datasets or operational conditions.

### Validation Strategy

The initial model comparison uses one fixed group-aware train/validation split.

Hyperparameter selection uses three-fold group-aware cross-validation within the training partition.

This provides a more controlled selection procedure but does not guarantee stable performance across all possible partitions.

Repeatedly selecting models using the same external validation results can lead to overfitting to that validation set.

An independent final evaluation is still needed for stronger generalization claims.

### Potential Near-Duplicate Leakage

Group-aware splitting prevents exact tweet text overlap between training and validation.

However, it does not guarantee protection against all forms of data leakage.

Near-duplicate texts, paraphrases, or highly similar observations may still appear in different groups.

### Feature Limitations

The current classification experiments use only tweet text.

The `keyword` and `location` columns are not included in the feature representation.

Additional features may be investigated in future experiments.

### Generalization

The reported validation scores describe performance on the selected local validation dataset.

The reported cross-validation scores describe performance across internal training folds.

Neither directly establishes performance on unseen Kaggle test data or in a real-world production environment.

## Automated Tests

The project uses Pytest to verify important properties of data splitting, text vectorization, label-conflict analysis, model training, model comparison, hyperparameter optimization, and evaluation.

The current test suite covers:

- Group-aware separation of identical texts.
- Compatible training and validation feature dimensions.
- Vocabulary restricted to training data.
- Vocabulary stability when transforming validation texts.
- Correct detection of conflicting target labels.
- Correct classification metrics and confusion matrix values using synthetic examples.
- Logistic Regression training and prediction behavior.
- Model comparison output structure and metric consistency.
- Group integrity during internal cross-validation.
- Correct execution and structure of GridSearchCV results.

The tests use small synthetic datasets to make expected behavior predictable and reduce dependence on external files.

Floating-point metrics are compared using `pytest.approx()` where appropriate.

The evaluation implementation also checks:

- Matching numbers of true labels and predictions.
- Valid predicted class labels.
- Consistency between confusion matrix counts and sample totals.
- Agreement between Scikit-learn's F1-score and its mathematical formula.

### Latest Verified Execution

- Python: 3.11.9
- Pytest: 9.2.0
- Tests collected: 10
- Tests passed: 10
- Tests failed: 0

These tests validate specific implementation properties.

They do not prove that the classifiers generalize well or that every possible source of data leakage has been eliminated.

## Reproducibility

### Requirements

The project uses Python and the dependencies listed in `requirements.txt`.

The latest verified execution used Python 3.11.9.

Verified core dependencies include:

- Pandas 3.1.0
- NumPy 2.4.6
- Scikit-learn 1.9.1
- Pytest 9.2.0

The environment dependency check completed without reporting broken requirements.

For consistent results, use the project's specified dependency versions and the same dataset.

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

### Training the Naive Bayes Baseline

```bash
python -m src.train_baseline
```

This command trains the Multinomial Naive Bayes classifier and generates predictions for the validation dataset.

### Evaluating the Naive Bayes Baseline

```bash
python -m src.evaluate_baseline
```

This command prints Accuracy, Precision, Recall, F1-score, the confusion matrix, and a classification report.

### Training Logistic Regression

```bash
python -m src.train_logistic
```

This command trains Logistic Regression using the existing group-aware data split and TF-IDF representation.

It also prints the validation metrics and confusion matrix values.

### Comparing Classification Models

```bash
python -m src.compare_models
```

This command trains and evaluates Multinomial Naive Bayes and Logistic Regression using the same training and validation data.

It prints the classification metrics, confusion matrix summaries, and differences between the models.

### Running Hyperparameter Optimization

```bash
python -m src.tune_logistic
```

This command performs group-aware cross-validation using `Pipeline`, `GroupKFold`, and `GridSearchCV`.

It prints:

- The selected value of `C`.
- The best mean cross-validation F1-score.
- Mean F1-score for each candidate.
- Standard deviation across folds for each candidate.

### Running Automated Tests

```bash
python -m pytest -v
```

Run these commands from the repository root with the virtual environment activated.

The reported baseline and comparison metrics were obtained using the original training dataset, the specified group-aware split, and the documented configurations.

The hyperparameter search uses the original training partition only.

A fresh-environment installation has not yet been independently verified.

## Engineering Implications

The project prioritizes reproducibility, separation of responsibilities, data integrity, and reliable model evaluation.

Data splitting, text vectorization, label-conflict investigation, model training, and classification evaluation have been organized into reusable functions.

This allows important parts of the pipeline to be tested independently using synthetic datasets.

Repeated texts are grouped during splitting to prevent exact overlapping examples from producing potentially optimistic validation results.

Preprocessing components are fitted exclusively on the appropriate training data.

During hyperparameter optimization, the complete text representation and classifier are placed inside a Pipeline to prevent leakage across internal cross-validation folds.

The Naive Bayes baseline provides a measurable reference for evaluating alternative classifiers.

The initial Logistic Regression experiment demonstrated improved recall and F1-score on the selected validation partition, alongside an increase in false positives.

This illustrates why evaluating multiple metrics is necessary when the consequences of prediction errors differ.

Group-aware cross-validation provides a structured procedure for selecting hyperparameters without using the original validation labels during the search.

However, small improvements in cross-validation metrics should not automatically be interpreted as evidence of superiority on unseen data.

The project distinguishes between baseline evaluation, model comparison, hyperparameter selection, and independent final evaluation.

All preprocessing, testing, and experimental decisions are documented to support reproducibility and meaningful model comparisons.

## Next Steps

- Evaluate the selected hyperparameter configuration using an appropriate held-out evaluation procedure.
- Investigate additional Logistic Regression hyperparameters and classification thresholds.
- Compare alternative classification models and text representations.
- Evaluate strategies for handling conflicting labels.
- Investigate potential near-duplicate texts and remaining data leakage risks.
- Strengthen the final evaluation strategy to support generalization claims.
- Generate predictions for Kaggle's test dataset.
- Submit predictions to the Kaggle competition.
- Prepare a deployable inference workflow.
- Deploy an interactive demonstration using Hugging Face.
- Document experimental comparisons and final conclusions.

## Author

Guilherme de Araújo Moreira