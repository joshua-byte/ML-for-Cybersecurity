# DGA Domain Detection Using Traditional Machine Learning

## Overview

This project explores the detection of Domain Generation Algorithm (DGA) domains using traditional machine learning techniques and handcrafted lexical features.

## Methodology

The notebook follows a machine learning pipeline consisting of data preprocessing, feature extraction, exploratory data analysis, model training, hyperparameter tuning and evaluation.

### 1. Data Preprocessing

The dataset is loaded from `dga2.csv` and undergoes the following preprocessing steps:

- Removal of missing values.
- Conversion of domain names and labels to lowercase.
- Removal of leading and trailing whitespace.
- Removal of empty domain strings and duplicate records.
- Validation of class labels.

A balanced sample of 10,000 benign domains and 10,000 DGA domains is selected for experimentation, resulting in a dataset of 20,000 records.

### 2. Feature Engineering

Ten handcrafted lexical features are extracted from each domain:

| Feature | Description |
|---|---|
| Domain Length | Total number of characters |
| Digit Count | Number of numerical characters |
| Vowel Count | Number of vowels |
| Consonant Count | Number of consonants |
| Digit Ratio | Proportion of numerical characters |
| Vowel Ratio | Proportion of vowels |
| Unique Character Ratio | Proportion of unique characters |
| Shannon Entropy | Character distribution entropy |
| Hyphen Count | Number of hyphens |
| Maximum Consonant Sequence | Longest uninterrupted sequence of consonants |

These features represent the structural and statistical characteristics of domain names.

### 3. Machine Learning Models

Three supervised machine learning algorithms are implemented and compared:

- **Logistic Regression:** A linear classification algorithm used as a baseline model.
- **Random Forest:** An ensemble learning algorithm that combines multiple decision trees.
- **Histogram-Based Gradient Boosting:** A boosting algorithm that builds an ensemble of decision trees.

### 4. Model Training and Hyperparameter Tuning

The dataset is divided into training and testing subsets using an 80:20 stratified split.

Hyperparameter optimisation is performed using GridSearchCV with three-fold Stratified K-Fold cross-validation. F1-score is used as the optimisation metric.

Feature scaling for Logistic Regression is incorporated into a Scikit-learn pipeline to prevent data leakage during cross-validation.

### 5. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Classification reports and confusion matrices are generated to examine the models' performance across benign and DGA classes.
