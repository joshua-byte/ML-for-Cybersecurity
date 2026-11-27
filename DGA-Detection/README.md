# DGA Domain Detection Using Machine Learning

## Overview

This project implements a supervised machine learning system for detecting Domain Generation Algorithm (DGA) domains using lexical and character-level analysis.

The project explores two approaches to domain classification:

- **Handcrafted lexical feature analysis:** Extracting statistical and structural characteristics from domain names and using traditional machine learning classifiers.
- **Character n-gram analysis:** Learning character-level patterns from domain strings using TF-IDF vectorisation and Logistic Regression.

The trained models are integrated into a Streamlit application that supports single-domain analysis, batch classification, model comparison and downloadable detection reports.

## Dataset

**Source:** [ExtraHop DGA Detection Training Dataset](https://github.com/ExtraHop/DGA-Detection-Training-Dataset)

The dataset contains benign and DGA-generated domain samples.

For the initial lexical-feature experiment, an equal number of samples from both classes were selected to construct a balanced dataset of 20,000 domains.

The classification convention used throughout the project is:

- `0` – Benign
- `1` – DGA

## Methodology

### 1. Data Preprocessing

The dataset undergoes the following preprocessing operations:

- Removal of missing values.
- Conversion of domain names and labels to lowercase.
- Removal of leading and trailing whitespace.
- Removal of empty domain strings.
- Removal of duplicate records.
- Validation of class labels.
- Sampling and class balancing for the initial experiment.

### 2. Feature Engineering: Lexical Analysis

Ten handcrafted lexical features are extracted from each domain.

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

These features capture structural and statistical characteristics of domain names that may help distinguish algorithmically generated domains from legitimate domains.

### 3. Character n-gram Analysis

In addition to handcrafted lexical features, the project implements a character-level classification approach.

Character n-grams represent consecutive sequences of characters within a domain name. TF-IDF vectorisation is used to transform these sequences into numerical representations.

Logistic Regression is then used to classify domains based on their character-level patterns.

Unlike handcrafted lexical features, character n-gram analysis can learn recurring character sequences and structural patterns directly from the training data.

The application integrates this classifier alongside the numeric-feature models.

### 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to investigate the characteristics of benign and DGA domains.

The analysis includes:

- Feature distribution analysis.
- Comparison of lexical feature values across classes.
- Statistical summaries of extracted features.
- Feature correlation analysis.
- Examination of relationships between domain characteristics and classification labels.

### 5. Machine Learning Models

The project explores the following supervised classification approaches:

**Numeric-feature models**

- Logistic Regression
- Random Forest
- Histogram-Based Gradient Boosting

**Character-level model**

- Character n-gram Logistic Regression

The numeric-feature models operate on the ten handcrafted lexical features, whereas the character n-gram classifier operates on TF-IDF representations of domain strings.

### 6. Model Training and Hyperparameter Optimisation

The initial numeric-feature experiment uses an 80:20 stratified training-testing split.

Hyperparameter optimisation is performed using GridSearchCV with three-fold Stratified K-Fold cross-validation. F1-score is used as the optimisation metric.

Feature scaling for Logistic Regression is incorporated into a Scikit-learn pipeline to prevent data leakage during cross-validation.

The fitted classifiers are exported into a consolidated Joblib model bundle for use by the Streamlit application.

## Application Features

The trained models are integrated into a Streamlit-based detection interface.

### 1. Single-Domain Analysis

Users can submit an individual domain for classification.

The application provides:

- Predicted classification.
- DGA probability estimate.
- Extracted lexical characteristics.
- Model-agreement information.
- Prediction explanations where supported.

### 2. Batch Domain Detection

Users can upload datasets containing multiple domains.

Supported formats:

- CSV files.
- TXT files containing one domain per line.

The application automatically identifies recognised domain columns in CSV files, including `domain`, `domain_name`, `host` and `hostname`.

Batch inference is processed in chunks to support larger datasets.

**Maximum input size:** 1,000,000 domain rows.

### 3. Domain Validation

Before classification, domain names undergo validation checks, including:

- Empty domain detection.
- Whitespace detection.
- Maximum domain length.
- DNS label length restrictions.
- Empty label detection.
- Invalid character detection.
- Leading and trailing hyphen checks.
- Internationalised domain name conversion and validation.

Invalid entries are rejected separately rather than silently classified.

### 4. Configurable Classification Threshold

The application allows users to adjust the DGA probability threshold.

The default threshold is 0.50.

Changing this threshold affects the predicted classification without changing the underlying probability estimate.

### 5. Evaluation Mode

The application supports evaluation using labelled CSV datasets.

Recognised domain columns include `domain`, `domain_name`, `host` and `hostname`.

Recognised label columns include `isDGA`, `label`, `class` and `target`.

The supported label conventions include:

- `0`, `benign`, `legit`, `legitimate`, `normal`
- `1`, `dga`, `malicious`, `malware`

Evaluation utilities support classification metrics and confusion matrix analysis.

### 6. Interactive Dashboard

The application provides visual representations of detection results, including:

- Prediction distribution.
- Probability distribution.
- Highest-scoring suspicious domains.
- Model comparison and evaluation visualisations.

### 7. Reporting and Export

The application supports downloadable detection outputs.

Available outputs include:

- Complete classification results in CSV format.
- Rejected domain entries in CSV format.
- HTML detection reports containing summary statistics and domain-level results.

## Initial Model Comparison

The following results represent the initial experiment comparing the three handcrafted lexical-feature classifiers.

| Model | Accuracy | DGA Precision | DGA Recall | DGA F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.40% | 84.62% | 76.75% | 80.49% |
| Random Forest | 83.90% | 86.26% | **80.65%** | 83.36% |
| Histogram-Based Gradient Boosting | **84.23%** | **87.80%** | 79.50% | **83.44%** |

Histogram-Based Gradient Boosting achieved the highest accuracy and DGA F1-score in this initial comparison, while Random Forest achieved the highest DGA recall.

These figures describe the initial numeric-feature experiment and should not be interpreted as the evaluation results of every model currently integrated into the application.

### Confusion Matrices

The following confusion matrices illustrate the initial classifiers' performance on the test set.

**Logistic Regression**

![Logistic Regression Confusion Matrix](https://github.com/user-attachments/assets/e633faa8-2660-4a2a-aa85-0de6bd564c80)

**Random Forest**

![Random Forest Confusion Matrix](https://github.com/user-attachments/assets/a44f3759-b540-43b7-89d7-eb787c021b5b)

**Histogram-Based Gradient Boosting**

![Histogram-Based Gradient Boosting Confusion Matrix](https://github.com/user-attachments/assets/f52b2e26-eb7d-4278-9465-b094a8d42ade)

## Limitations

- Classification is based on learned lexical and character-level patterns rather than live DNS or network traffic analysis.
- Legitimate domains can exhibit characteristics similar to DGA-generated domains.
- A domain's predicted probability is a model estimate, not a verified measure of maliciousness.
- Performance may vary across datasets, DGA families and previously unseen domain-generation techniques.
- The initial benchmark results should not be assumed to generalise to all real-world DNS environments.
- The application does not independently establish whether a domain is actively involved in malicious communication.
