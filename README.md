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
- [ ] Model training and evaluation
- [ ] Kaggle submission
- [ ] Hugging Face deployment

## Dataset

Source: [Kaggle - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

Download the dataset and place the CSV files inside `data/raw/`.

## Current Implementation

The `src/explore.py` script loads the training dataset and performs exploratory data analysis, including missing values, duplicate detection, label conflicts, class distribution, and basic consistency checks.

Run the analysis from the project root:

`python src/explore.py`

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

### Engineering Implications

The initial classification pipeline will use only the `text` column, avoiding unnecessary preprocessing of `keyword` and `location`.

Repeated texts must be handled carefully when separating training and validation data to prevent overlapping examples from producing overly optimistic evaluation results.

Conflicting target labels will require further investigation before selecting an appropriate treatment strategy.

Model evaluation will prioritize the F1-score, the competition's evaluation metric, while also considering precision, recall, and confusion matrices.

All preprocessing and validation decisions will be documented to support reproducibility and reliable model evaluation.

## Author

Guilherme de Araújo Moreira