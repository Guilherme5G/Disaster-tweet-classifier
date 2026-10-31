# Disaster Tweet Classifier

A reproducible Machine Learning project for binary text classification, developed using the Kaggle Natural Language Processing with Disaster Tweets competition.

## Objective

Classify whether a tweet refers to a real disaster (1) or not (0).

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Pytest

## Project Status

Work in progress.

- [x] Python virtual environment setup
- [x] Dataset download and loading
- [x] Initial data exploration
- [x] Data quality analysis
- [x] Group-aware train/validation splitting
- [x] TF-IDF text vectorization
- [x] Automated tests for data splitting and vectorization
- [ ] Model training and evaluation
- [ ] Kaggle submission
- [ ] Hugging Face deployment

## Dataset

Source: [Kaggle - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

Download the dataset and place the CSV files inside `data/raw/`.

The dataset files are not tracked by Git and must be downloaded separately.

## Current Implementation

The project currently includes the following modules:

- `src/explore.py`: Loads the training dataset and performs exploratory data analysis, including missing values, duplicate detection, label conflicts, class distribution, and consistency checks.
- `src/text_basics.py`: Demonstrates basic tokenization and Bag-of-Words word frequencies.
- `src/tfidf_basics.py`: Demonstrates TF-IDF vectorization using a small example corpus.
- `src/data_split.py`: Implements reusable group-aware training and validation splitting using `GroupShuffleSplit`.
- `src/text_pipeline.py`: Implements reusable TF-IDF text vectorization, fitting the vocabulary exclusively on training data.
- `tests/test_data_split.py`: Tests group-aware splitting and verifies that identical texts do not appear in both datasets.
- `tests/test_text_pipeline.py`: Tests feature dimensionality, vocabulary integrity, and vocabulary stability.

Run the data exploration from the project root:

```bash
python src/explore.py
```

Run the group-aware data splitting:

```bash
python -m src.data_split
```

Run the text vectorization workflow:

```bash
python -m src.text_pipeline
```

## Exploratory Data Analysis

### Dataset Overview

The dataset contains 7,613 observations and 5 columns, designed for binary classification of disaster-related tweets.

The target distribution is relatively balanced, with 57.03% non-disaster tweets (class 0) and 42.97% disaster-related tweets (class 1).

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

The current text vectorization workflow uses only the `text` column. A classification model has not yet been trained.

### Known Data Quality Limitations

Exploratory analysis identified 18 distinct text values associated with conflicting target labels. These conflicts have not yet been resolved.

Group-aware splitting prevents exact text overlap but does not eliminate all possible forms of data leakage, including potential near-duplicate messages.

Label inconsistencies and potential near-duplicates require further investigation before final model evaluation.

The current splitting strategy also does not guarantee identical class proportions between training and validation.

## Automated Tests

The project uses Pytest to validate data splitting and text vectorization behavior.

The current test suite covers:

- Group-aware separation of identical texts.
- Compatible training and validation feature dimensions.
- Vocabulary restricted to training data.
- Vocabulary stability when transforming validation texts.

The tests use small synthetic datasets to make expected behavior predictable and reduce dependence on external files.

Run the test suite from the project root:

```bash
python -m pytest -v
```

**Latest verified execution:** 4 tests passed using Python 3.11.9 and Pytest 9.2.0.

These tests validate specific preprocessing properties. They do not yet establish classification performance or complete protection against every form of data leakage.

## Engineering Implications

The project prioritizes reproducibility, separation of responsibilities, and reliable evaluation.

The data splitting and text vectorization logic have been organized into reusable functions, allowing them to be tested independently.

The current text vectorization workflow uses only the `text` column. The classifier has not yet been trained.

Repeated texts are grouped during splitting to prevent exact overlapping examples from producing potentially optimistic evaluation results.

Preprocessing components are fitted exclusively on training data to avoid incorporating validation information into the learned representation.

Conflicting target labels require further investigation before selecting an appropriate treatment strategy.

Future model evaluation will prioritize the F1-score, the competition's evaluation metric, while also considering precision, recall, and confusion matrices.

All preprocessing, testing, and validation decisions will be documented to support reproducibility and reliable model evaluation.

## Author

Guilherme de Araújo Moreira