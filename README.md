
# Preprocessor Tuning with Scikit-learn Pipelines

This project focuses on building and optimizing machine learning preprocessing pipelines using **scikit-learn**.

## Overview

The notebook uses a tumor classification dataset containing two classes: malignant and benign. The main objective is to identify as many malignant tumors as possible by building an effective preprocessing and classification pipeline.

## Project Workflow

### 1. Building a Pipeline
- Handle missing values using `KNNImputer`.
- Scale numerical features using `MinMaxScaler`.
- Train a `LogisticRegression` classifier.
- Combine all steps into a single scikit-learn `Pipeline`.

### 2. Hyperparameter Tuning
- Optimize the `n_neighbors` parameter of `KNNImputer`.
- Compare the values `2`, `5`, and `10`.
- Use `GridSearchCV` with 5-fold cross-validation.
- Select **recall** as the scoring metric to prioritize detecting malignant tumors.

### 3. Model Evaluation
- Evaluate the optimized pipeline using 5-fold cross-validation.
- Measure performance with recall.
- Validate the results using the provided challenge tests.

### 4. Prediction
- Fit the optimized pipeline on the dataset.
- Predict the class of a new tumor sample.

## Results

- Best number of neighbors: **5**
- Cross-validation recall: **0.9245**
- New sample prediction: **Malignant**

## Tools and Libraries

- Python
- pandas
- scikit-learn
- Jupyter Notebook

## Repository Structure

```text
.
├── 01-Preprocessor-Tuning.ipynb
├── tests/
├── .gitignore
├── Makefile
└── README.md
```

## Purpose

This repository is part of my machine learning learning archive. It demonstrates practical experience with preprocessing pipelines, missing-value imputation, feature scaling, hyperparameter optimization, and model evaluation.
