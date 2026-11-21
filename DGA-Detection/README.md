# DGA Domain Detection Using Traditional Machine Learning

## Overview

This project explores the detection of Domain Generation Algorithm (DGA) domains using traditional machine learning algorithms and handcrafted lexical features.


## Dataset

**Source:** [ExtraHop DGA Detection Training Dataset](https://github.com/ExtraHop/DGA-Detection-Training-Dataset)

## Methodology

### 1. Data Preprocessing

The dataset undergoes the following preprocessing steps:

- Removal of missing values.
- Conversion of domain names and labels to lowercase.
- Removal of leading and trailing whitespace.
- Removal of empty domain strings and duplicate records.
- Validation of class labels.

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

These features capture the structural and statistical characteristics of domain names.

### 3. Exploratory Data Analysis

Exploratory analysis is performed to examine feature distributions and differences between benign and DGA domains.

A feature correlation matrix is also calculated to explore relationships between the extracted lexical features.

### 4. Machine Learning Models

ML models chosen for comparison:
- **Logistic Regression**
- **Random Forest**
- **Histogram-Based Gradient Boosting**

### 5. Model Training and Hyperparameter Tuning

The dataset is divided into training and testing subsets using an 80:20 stratified split.

Hyperparameter optimisation is performed using GridSearchCV with three-fold Stratified K-Fold cross-validation. F1-score is used as the optimisation metric.

Feature scaling for Logistic Regression is incorporated into a Scikit-learn pipeline to avoid data leakage during cross-validation.

### 6. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Results

| Model | Accuracy | DGA Precision | DGA Recall | DGA F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.40% | 84.62% | 76.75% | 80.49% |
| Random Forest | 83.90% | 86.26% | **80.65%** | 83.36% |
| HistGradientBoosting | **84.23%** | **87.80%** | 79.50% | **83.44%** |


### Confusion Matrices

The confusion matrices below show the performance of each model on the test set.

**Logistic Regression**

<img width="577" height="470" alt="output_LOG" src="https://github.com/user-attachments/assets/e633faa8-2660-4a2a-aa85-0de6bd564c80" />


**Random Forest**

<img width="577" height="470" alt="output_RANDOM" src="https://github.com/user-attachments/assets/a44f3759-b540-43b7-89d7-eb787c021b5b" />


**Histogram-Based Gradient Boosting**

<img width="577" height="470" alt="output_grad" src="https://github.com/user-attachments/assets/f52b2e26-eb7d-4278-9465-b094a8d42ade" />
