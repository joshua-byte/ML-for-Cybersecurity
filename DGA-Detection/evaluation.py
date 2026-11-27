
"""
Evaluation utilities for supervised DGA detection.

Responsibilities:
- Parse labelled CSV datasets.
- Normalise ground-truth labels.
- Calculate binary classification metrics.
- Generate confusion matrix data.
- Compare multiple supervised classifiers.

Label convention:
    0 = Benign
    1 = DGA
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from detector_engine import (
    MAX_DOMAINS,
    CSV_READ_CHUNK_SIZE,
    normalise_domain,
)


# ============================================================
# 1. LABEL CONFIGURATION
# ============================================================

BENIGN_LABELS = {
    "0",
    "benign",
    "legitimate",
    "normal",
}

DGA_LABELS = {
    "1",
    "dga",
    "malicious",
    "malware",
}


# ============================================================
# 2. LABEL NORMALISATION
# ============================================================

def normalise_label(value: object) -> int:
    """
    Convert a ground-truth label into binary format.

    Returns:
        0 = Benign
        1 = DGA

    Raises:
        ValueError for missing or unsupported labels.
    """

    if value is None:
        raise ValueError("Missing ground-truth label.")

    label = str(value).strip().lower()

    if label in BENIGN_LABELS:
        return 0

    if label in DGA_LABELS:
        return 1

    raise ValueError(
        f"Unrecognised label '{value}'. "
        "Use 0/benign or 1/dga."
    )


# ============================================================
# 3. LABELLED CSV PARSING
# ============================================================

def parse_labeled_csv(
    uploaded_file,
    label_column: str = "label",
) -> tuple[list[str], np.ndarray, list[str]]:
    """
    Parse a CSV containing domain and ground-truth label columns.

    Required columns:
        domain
        label

    Accepted labels:
        Benign: 0, benign, legitimate, normal
        DGA:    1, dga, malicious, malware

    Returns:
        domains: Normalised domain strings.
        labels: Binary NumPy array.
        warnings: Ingestion warnings.

    The original row order is preserved so labels can be
    aligned with detection results after domain validation.
    """

    filename = str(uploaded_file.name).lower()

    if not filename.endswith(".csv"):
        raise ValueError(
            "Evaluation mode requires a CSV file."
        )

    raw = uploaded_file.getvalue()

    # --------------------------------------------------------
    # Read CSV header
    # --------------------------------------------------------

    try:
        decoded = raw.decode("utf-8-sig")

        header = next(
            csv.reader(io.StringIO(decoded)),
            None,
        )

    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(
            f"Could not read CSV header: {exc}"
        ) from exc

    if not header:
        raise ValueError("The CSV file is empty.")

    normalised_headers = [
        str(column).strip().lower()
        for column in header
    ]

    domain_positions = [
        index
        for index, column in enumerate(normalised_headers)
        if column == "domain"
    ]

    label_positions = [
        index
        for index, column in enumerate(normalised_headers)
        if column == label_column.strip().lower()
    ]

    if len(domain_positions) != 1:
        raise ValueError(
            "CSV must contain exactly one column named 'domain'."
        )

    if len(label_positions) != 1:
        raise ValueError(
            f"CSV must contain exactly one column named "
            f"'{label_column}'."
        )

    domain_name = header[domain_positions[0]]
    label_name = header[label_positions[0]]

    # --------------------------------------------------------
    # Read CSV in chunks
    # --------------------------------------------------------

    domains = []
    labels = []
    warnings = []

    try:

        reader = pd.read_csv(
            io.BytesIO(raw),
            usecols=[domain_name, label_name],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            chunksize=CSV_READ_CHUNK_SIZE,
            skip_blank_lines=False,
        )

        for chunk in reader:

            for domain_value, label_value in zip(
                chunk[domain_name].tolist(),
                chunk[label_name].tolist(),
            ):

                domain = normalise_domain(domain_value)
                label = normalise_label(label_value)

                domains.append(domain)
                labels.append(label)

                if len(domains) > MAX_DOMAINS:
                    raise ValueError(
                        f"Please upload no more than "
                        f"{MAX_DOMAINS:,} rows at a time."
                    )

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            f"Could not parse labelled CSV: {exc}"
        ) from exc

    if not domains:
        raise ValueError(
            "No evaluation rows were found in the CSV."
        )

    if len(domains) != len(labels):
        raise RuntimeError(
            "Domain and label counts do not match."
        )

    empty_domains = sum(
        not domain for domain in domains
    )

    if empty_domains:
        warnings.append(
            f"{empty_domains:,} empty domain rows were retained "
            "and will be excluded during domain validation."
        )

    return (
        domains,
        np.asarray(labels, dtype=np.int8),
        warnings,
    )


# ============================================================
# 4. INPUT VALIDATION
# ============================================================

def validate_evaluation_inputs(
    y_true,
    y_probability,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate labels and model probabilities before evaluation.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    if y_true.ndim != 1 or y_probability.ndim != 1:
        raise ValueError(
            "Labels and probabilities must be one-dimensional."
        )

    if len(y_true) != len(y_probability):
        raise ValueError(
            "Labels and probabilities must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    if not np.isfinite(y_probability).all():
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    if (
        (y_probability < 0).any()
        or (y_probability > 1).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    try:
        numeric_labels = y_true.astype(float)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Labels must be encoded as 0 (benign) or 1 (DGA)."
        ) from exc

    if not np.isfinite(numeric_labels).all():
        raise ValueError("Ground-truth labels contain invalid values.")

    if not np.isin(numeric_labels, [0, 1]).all():
        raise ValueError(
            "Labels must contain only 0 (benign) and 1 (DGA)."
        )

    y_true = numeric_labels.astype(np.int8)

    return y_true, y_probability


# ============================================================
# 5. SINGLE MODEL EVALUATION
# ============================================================

def evaluate_predictions(
    y_true,
    y_probability,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a model's probability predictions.

    Returns:
        Precision
        Recall
        F1-score
        False-positive rate
        Confusion matrix components
        Accuracy

    Assumes:
        0 = Benign
        1 = DGA
    """

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Decision threshold must be between 0 and 1."
        )

    y_true, y_probability = validate_evaluation_inputs(
        y_true,
        y_probability,
    )

    y_pred = (
        y_probability >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    total = len(y_true)

    accuracy = (
        (tp + tn) / total
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    return {
        "threshold": float(threshold),

        "total_samples": int(total),

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),

        "false_positive_rate": float(fpr),
        "specificity": float(specificity),
    }


# ============================================================
# 6. MULTI-MODEL EVALUATION
# ============================================================

def evaluate_multiple_models(
    y_true,
    probabilities_by_model: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate multiple classifiers on the same labelled dataset.

    Args:
        y_true:
            Ground-truth binary labels.

        probabilities_by_model:
            Dictionary mapping model names to probability arrays.

        threshold:
            Shared classification threshold.

    Returns:
        Dictionary containing metrics for each model.

    All models must be evaluated against the same rows.
    """

    if not probabilities_by_model:
        raise ValueError(
            "No model probabilities were supplied."
        )

    y_true = np.asarray(y_true)

    evaluations = {}

    expected_length = len(y_true)

    for model_name, probabilities in probabilities_by_model.items():

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if len(probabilities) != expected_length:
            raise ValueError(
                f"Model '{model_name}' produced "
                "a different number of predictions."
            )

        evaluations[model_name] = evaluate_predictions(
            y_true=y_true,
            y_probability=probabilities,
            threshold=threshold,
        )

    return evaluations


# ============================================================
# 7. CONFUSION MATRIX DATA
# ============================================================

def get_confusion_matrix(
    metrics: dict,
) -> pd.DataFrame:
    """
    Convert evaluation results into a labelled confusion matrix.
    """

    required = {
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }

    missing = required - set(metrics)

    if missing:
        raise ValueError(
            f"Missing confusion matrix values: {sorted(missing)}"
        )

    return pd.DataFrame(
        [
            [
                metrics["true_negatives"],
                metrics["false_positives"],
            ],
            [
                metrics["false_negatives"],
                metrics["true_positives"],
            ],
        ],
        index=[
            "Actual benign",
            "Actual DGA",
        ],
        columns=[
            "Predicted benign",
            "Predicted DGA",
        ],
    )


# ============================================================
# 8. EVALUATION SUMMARY
# ============================================================

def build_evaluation_summary(
    evaluations: dict,
) -> pd.DataFrame:
    """
    Convert multiple model evaluations into a comparison table.
    """

    rows = []

    for model_name, metrics in evaluations.items():

        rows.append({
            "Model": model_name,
            "Samples": metrics["total_samples"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1_score"],
            "False-positive rate": metrics["false_positive_rate"],
            "Specificity": metrics["specificity"],
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
