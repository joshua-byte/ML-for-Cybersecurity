
"""Loading, parsing, feature extraction and supervised DGA inference."""

from __future__ import annotations

import io
import math
import re
from collections import Counter
from pathlib import Path
import csv
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/dga_supervised_models.joblib")

MAX_DOMAINS = 1_000_000
CSV_READ_CHUNK_SIZE = 50_000
INFERENCE_BATCH_SIZE = 10_000

DEFAULT_FEATURE_COLUMNS = [
    "domain_length",
    "digit_count",
    "vowel_count",
    "consonant_count",
    "digit_ratio",
    "vowel_ratio",
    "unique_char_ratio",
    "entropy",
    "hyphen_count",
    "max_consonant_sequence",
]

VOWEL_PATTERN = r"[aeiou]"
CONSONANT_PATTERN = r"[bcdfghjklmnpqrstvwxyz]"

# Valid ASCII DNS label:
# - Starts and ends with an alphanumeric character.
# - May contain alphanumeric characters and hyphens internally.
VALID_ASCII_LABEL = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"
)

NGRAM_MODEL_NAME = "Character n-gram Logistic Regression"

# Provisional probability bands for analyst-facing severity labels.
# These are not detection thresholds and should be reviewed against
# validation data before being treated as operational risk levels.
SEVERITY_THRESHOLDS = {
    "medium": 0.25,
    "high": 0.50,
    "critical": 0.75,
}

# Accepted textual labels when reading a labelled evaluation CSV.
BENIGN_LABELS = {"0", "benign", "legit", "legitimate", "normal"}
DGA_LABELS = {"1", "dga", "malicious", "malware"}


# ============================================================
# 2. MODEL LOADING
# ============================================================

def load_model_bundle(
    model_path: str | Path = MODEL_PATH,
) -> dict:
    """
    Load a trusted Joblib bundle exported by the training notebook.

    Expected bundle structure:

    {
        "numeric_models": {...},
        "ngram_model": fitted_model,
        "feature_columns": [...],
        "label_mapping": {...},
        "evaluation_metrics": {...},
        "ngram_metrics": {...}
    }

    The evaluation fields are optional.
    """

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'. "
            "Run the notebook's final export cell first."
        )

    # Only load Joblib files from a trusted source.
    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError(
            "Invalid model bundle. Expected a dictionary."
        )

    if "numeric_models" not in bundle:
        raise ValueError(
            "Model bundle does not contain 'numeric_models'."
        )

    if "ngram_model" not in bundle:
        raise ValueError(
            "Model bundle does not contain 'ngram_model'."
        )

    if not isinstance(bundle["numeric_models"], dict):
        raise ValueError(
            "'numeric_models' must be a dictionary."
        )

    if not bundle["numeric_models"]:
        raise ValueError(
            "No numeric-feature models were found in the bundle."
        )

    if bundle["ngram_model"] is None:
        raise ValueError(
            "The character n-gram model is missing from the bundle."
        )

    if "feature_columns" in bundle:
        feature_columns = bundle["feature_columns"]

        if not isinstance(feature_columns, (list, tuple)):
            raise ValueError(
                "'feature_columns' must be a list or tuple."
            )

        if not feature_columns:
            raise ValueError(
                "'feature_columns' cannot be empty."
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError(
                "'feature_columns' contains duplicate names."
            )

    return bundle


# ============================================================
# 3. DOMAIN NORMALISATION
# ============================================================

def normalise_domain(value: object) -> str:
    """
    Convert an input value into a normalised domain string.

    Lowercase conversion and whitespace stripping are retained
    to remain consistent with the original feature extractor.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip().lower()


# ============================================================
# 4. DOMAIN VALIDATION
# ============================================================

def validate_domain(value: str) -> str | None:
    """
    Validate a domain name.

    Checks:
    - Empty values
    - Whitespace
    - Maximum DNS name length
    - Empty labels
    - Individual label length
    - Invalid characters
    - Leading and trailing hyphens
    - Internationalised domain names

    Returns None if valid, otherwise a rejection reason.

    Note:
    Python's built-in IDNA codec is used for Unicode-to-ASCII
    conversion. It is not a complete implementation of every
    modern IDNA2008/UTS #46 validation rule.
    """

    if not value:
        return "Empty domain"

    if any(char.isspace() for char in value):
        return "Contains whitespace"

    if len(value) > 254:
        return "Exceeds maximum domain length"

    # A trailing dot represents an absolute DNS name.
    # Preserve it in the original value for feature extraction.
    validation_value = (
        value[:-1] if value.endswith(".") else value
    )

    if not validation_value:
        return "Empty domain"

    if validation_value.startswith("."):
        return "Invalid dot placement"

    if ".." in validation_value:
        return "Contains empty domain label"

    labels = validation_value.split(".")

    ascii_labels = []

    for label in labels:

        if not label:
            return "Contains empty domain label"

        # Convert Unicode labels into their IDNA ASCII form
        # for DNS length and character validation.
        try:
            ascii_label = (
                label.encode("idna")
                .decode("ascii")
                .lower()
            )

        except UnicodeError:
            return "Invalid internationalised domain label"

        if len(ascii_label) > 63:
            return "Domain label exceeds 63 characters"

        if not VALID_ASCII_LABEL.fullmatch(ascii_label):
            return "Contains invalid domain label characters"

        ascii_labels.append(ascii_label)

    ascii_domain = ".".join(ascii_labels)

    if len(ascii_domain) > 253:
        return "Exceeds maximum DNS name length"

    return None


# ============================================================
# 5. FILE PARSING
# ============================================================


def _normalise_column_name(value: object) -> str:
    """Normalise a CSV header for case- and punctuation-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _resolve_csv_column(
    headers: list[str],
    *,
    column_kind: str,
    requested_column: str | None = None,
) -> int:
    """
    Resolve a CSV column by an explicit selection or recognised aliases.

    Domain aliases include names containing 'domain' and common hostname
    names. When both 'domain' and 'host' are present, prefer 'host' because
    it commonly contains the complete hostname (including the TLD).

    Label aliases include isDGA, label, class and target.
    """
    normalised = [_normalise_column_name(header) for header in headers]

    if requested_column is not None:
        requested = _normalise_column_name(requested_column)
        matches = [i for i, name in enumerate(normalised) if name == requested]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Selected {column_kind} column '{requested_column}' "
                f"was not found. Available columns: {headers}"
            )
        raise ValueError(
            f"Selected {column_kind} column '{requested_column}' occurs "
            "more than once. Rename the duplicate columns."
        )

    if column_kind == "domain":
        # Prefer complete-hostname fields over shorter domain components.
        preferred_aliases = (
            "fullhost",
            "hostname",
            "host",
            "domain",
            "domainname",
            "sourcedomain",
            "domainstring",
        )
        for alias in preferred_aliases:
            matches = [i for i, name in enumerate(normalised) if name == alias]
            if matches:
                if len(matches) == 1:
                    return matches[0]
                raise ValueError(
                    f"CSV contains multiple possible '{column_kind}' columns "
                    f"named like '{alias}'. Available columns: {headers}. "
                    "Pass an explicit domain_column or rename the columns."
                )

        # Also accept other descriptive headers containing "domain".
        matches = [
            i for i, name in enumerate(normalised)
            if "domain" in name
        ]
    elif column_kind == "label":
        preferred_aliases = (
            "isdga",
            "label",
            "class",
            "target",
            "groundtruth",
            "groundtruthlabel",
            "is_dga",
        )
        for alias in preferred_aliases:
            matches = [i for i, name in enumerate(normalised) if name == alias]
            if matches:
                if len(matches) == 1:
                    return matches[0]
                raise ValueError(
                    f"CSV contains multiple possible '{column_kind}' columns "
                    f"named like '{alias}'. Available columns: {headers}. "
                    "Pass an explicit label_column or rename the columns."
                )
        matches = []
    else:
        raise ValueError(f"Unsupported CSV column type: {column_kind}")

    if not matches:
        if column_kind == "domain":
            raise ValueError(
                "Could not identify a domain column. Use a header such as "
                "'domain', 'domain_name', 'host' or 'hostname'. "
                f"Available columns: {headers}"
            )
        raise ValueError(
            "Could not identify a ground-truth label column. Use a header "
            "such as 'label', 'isDGA', 'class' or 'target'. "
            f"Available columns: {headers}"
        )

    if len(matches) > 1:
        candidate_names = [headers[i] for i in matches]
        raise ValueError(
            f"CSV contains multiple possible {column_kind} columns: "
            f"{candidate_names}. Pass an explicit {column_kind}_column "
            "or rename the columns."
        )
    return matches[0]


def parse_uploaded_file(
    uploaded_file,
    domain_column: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Parse a CSV or TXT file for domain detection.

    CSV domain headers are detected automatically. Recognised examples:
    domain, domain_name, source_domain, domain_string, host, hostname,
    and full_host. If several candidates exist, pass domain_column
    explicitly rather than silently choosing the wrong field.

    TXT files are interpreted as one domain per non-empty, non-comment line.

    Returns:
        domains
        ingestion_warnings
    """
    filename = str(uploaded_file.name).lower()
    raw = uploaded_file.getvalue()

    warnings: list[str] = []
    domains: list[str] = []

    def add_domains(values):
        for value in values:
            domains.append(normalise_domain(value))
            if len(domains) > MAX_DOMAINS:
                raise ValueError(
                    f"Please upload no more than {MAX_DOMAINS:,} "
                    "domain rows at a time."
                )

    if filename.endswith(".csv"):
        try:
            decoded_header = raw.decode("utf-8-sig")
            header_reader = csv.reader(io.StringIO(decoded_header))
            header = next(header_reader, None)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ValueError(f"Could not read CSV header: {exc}") from exc

        if not header:
            raise ValueError("The CSV file is empty.")

        domain_position = _resolve_csv_column(
            [str(column) for column in header],
            column_kind="domain",
            requested_column=domain_column,
        )

        try:
            reader = pd.read_csv(
                io.BytesIO(raw),
                usecols=[domain_position],
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
                chunksize=CSV_READ_CHUNK_SIZE,
                skip_blank_lines=False,
            )
            for chunk in reader:
                add_domains(chunk.iloc[:, 0].tolist())
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not parse CSV file: {exc}") from exc

        empty_count = sum(not domain for domain in domains)
        if empty_count:
            warnings.append(
                f"{empty_count:,} empty CSV rows were retained for "
                "validation and will be rejected."
            )

    elif filename.endswith(".txt"):
        if domain_column is not None:
            raise ValueError(
                "domain_column can only be specified for CSV files."
            )
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("TXT file must use UTF-8 encoding.") from exc

        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            add_domains([line])
    else:
        raise ValueError(
            "Unsupported file type. Upload a .csv or .txt file."
        )

    if not domains:
        raise ValueError(
            "No domain values were found in the uploaded file."
        )

    return domains, warnings


def parse_labeled_csv(
    uploaded_file,
    label_column: str | None = None,
    domain_column: str | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """
    Parse a labelled evaluation CSV.

    Domain column aliases include domain, domain_name, source_domain,
    domain_string, host, hostname and full_host. If both 'domain' and
    'host' are present, 'host' is preferred as the complete hostname.

    Label column aliases include isDGA, label, class and target.

    Labels are normalised to:
        0 = benign (0, benign, legit, legitimate, normal)
        1 = DGA    (1, dga, malicious, malware)

    Explicit column names can be supplied when automatic detection is
    ambiguous. Returns domains, binary labels and ingestion warnings.
    """
    filename = str(uploaded_file.name).lower()
    if not filename.endswith(".csv"):
        raise ValueError("Evaluation mode requires a CSV file.")

    raw = uploaded_file.getvalue()
    try:
        header = next(
            csv.reader(io.StringIO(raw.decode("utf-8-sig"))),
            None,
        )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV header: {exc}") from exc

    if not header:
        raise ValueError("The CSV file is empty.")

    headers = [str(column) for column in header]
    domain_position = _resolve_csv_column(
        headers,
        column_kind="domain",
        requested_column=domain_column,
    )
    label_position = _resolve_csv_column(
        headers,
        column_kind="label",
        requested_column=label_column,
    )

    if domain_position == label_position:
        raise ValueError(
            "The domain column and label column must be different."
        )

    domains: list[str] = []
    labels: list[int] = []
    nonstandard_positive_labels: set[str] = set()

    try:
        reader = pd.read_csv(
            io.BytesIO(raw),
            usecols=[domain_position, label_position],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            chunksize=CSV_READ_CHUNK_SIZE,
            skip_blank_lines=False,
        )

        for chunk in reader:
            # usecols may return columns in source order; use positional
            # indices within the selected two-column frame.
            selected_positions = sorted(
                [domain_position, label_position]
            )
            domain_offset = selected_positions.index(domain_position)
            label_offset = selected_positions.index(label_position)

            domain_values = chunk.iloc[:, domain_offset].tolist()
            label_values = chunk.iloc[:, label_offset].tolist()

            for domain_value, label_value in zip(
                domain_values, label_values
            ):
                domains.append(normalise_domain(domain_value))

                label_key = str(label_value).strip().lower()
                if label_key in BENIGN_LABELS:
                    labels.append(0)
                elif label_key in DGA_LABELS:
                    labels.append(1)
                else:
                    # Evaluation is binary: zero denotes benign and any
                    # positive integer class denotes DGA. This supports
                    # datasets whose DGA families are encoded as 1, 2, 3, ...
                    # while rejecting negative, fractional and non-numeric
                    # labels that cannot safely be interpreted.
                    try:
                        numeric_label = float(label_key)
                    except (TypeError, ValueError):
                        numeric_label = float("nan")

                    if (
                        math.isfinite(numeric_label)
                        and numeric_label.is_integer()
                        and numeric_label == 0
                    ):
                        labels.append(0)
                    elif (
                        math.isfinite(numeric_label)
                        and numeric_label.is_integer()
                        and numeric_label > 0
                    ):
                        labels.append(1)
                        if numeric_label != 1:
                            nonstandard_positive_labels.add(label_key)
                    else:
                        raise ValueError(
                            f"Unrecognised or empty label '{label_value}' "
                            f"in column '{headers[label_position]}'. "
                            "For binary evaluation, use 0 for benign and "
                            "a positive integer for DGA, or use recognised "
                            "text labels such as benign, legit and dga."
                        )

                if len(domains) > MAX_DOMAINS:
                    raise ValueError(
                        f"Please upload no more than {MAX_DOMAINS:,} "
                        "rows at a time."
                    )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not parse labelled CSV: {exc}"
        ) from exc

    if not domains:
        raise ValueError(
            "No evaluation rows were found in the CSV file."
        )

    warnings: list[str] = []
    if nonstandard_positive_labels:
        mapped_values = ", ".join(sorted(nonstandard_positive_labels))
        warnings.append(
            f"Numeric label value(s) {mapped_values} were mapped to DGA (1). "
            "Binary evaluation treats 0 as benign and every positive integer "
            "class as DGA. Confirm that this matches your dataset's label scheme."
        )

    invalid_count = sum(
        bool(validate_domain(domain)) for domain in domains
    )
    if invalid_count:
        warnings.append(
            f"{invalid_count:,} rows contain invalid domains; these rows "
            "must be excluded from evaluation or handled explicitly."
        )

    return domains, np.asarray(labels, dtype=int), warnings


# ============================================================
# 6. SHANNON CHARACTER ENTROPY
# ============================================================

def calculate_entropy(domain: str) -> float:
    """
    Calculate Shannon entropy of the domain string.

    H(X) = -sum(p(x) * log2(p(x)))
    """

    if not domain:
        return 0.0

    counts = Counter(domain)
    length = len(domain)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


# ============================================================
# 7. LEXICAL FEATURE EXTRACTION
# ============================================================

def extract_domain_features(domain: str) -> dict:
    """
    Mirrors extract_domain_features() in the supplied notebook.

    IMPORTANT:
    Keep this implementation synchronised with the notebook.
    Do not modify feature definitions without retraining models.
    """

    domain = (
        ""
        if pd.isna(domain)
        else str(domain).lower().strip()
    )

    length = len(domain)

    digit_count = sum(
        char.isdigit() for char in domain
    )

    vowel_count = len(
        re.findall(VOWEL_PATTERN, domain)
    )

    consonant_count = len(
        re.findall(CONSONANT_PATTERN, domain)
    )

    hyphen_count = domain.count("-")

    return {
        "domain_length": length,

        "digit_count": digit_count,

        "vowel_count": vowel_count,

        "consonant_count": consonant_count,

        "digit_ratio": (
            digit_count / length if length else 0.0
        ),

        "vowel_ratio": (
            vowel_count / length if length else 0.0
        ),

        "unique_char_ratio": (
            len(set(domain)) / length if length else 0.0
        ),

        "entropy": calculate_entropy(domain),

        "hyphen_count": hyphen_count,

        "max_consonant_sequence": max(
            (
                len(sequence)
                for sequence in re.findall(
                    CONSONANT_PATTERN + r"+",
                    domain,
                )
            ),
            default=0,
        ),
    }


# ============================================================
# 8. FEATURE FRAME CONSTRUCTION
# ============================================================

def build_feature_frame(
    domains: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Construct the numerical feature matrix.

    Feature column order follows the exported model bundle.
    """

    frame = pd.DataFrame(
        [
            extract_domain_features(domain)
            for domain in domains
        ]
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Feature extraction is missing columns: "
            f"{missing_columns}"
        )

    return frame.reindex(columns=feature_columns)


# ============================================================
# 9. IDENTIFY THE DGA PROBABILITY COLUMN
# ============================================================

def get_dga_probability_index(
    estimator,
    bundle: dict,
) -> int:
    """
    Identify the probability column belonging to the DGA class.

    Does not assume that DGA is always the second column.

    Expected label mapping example:
        {0: "benign", 1: "dga"}
    """

    classes = getattr(estimator, "classes_", None)

    if classes is None:
        raise ValueError(
            "The selected estimator does not expose classes_. "
            "Cannot identify the DGA probability column."
        )

    label_mapping = bundle.get(
        "label_mapping",
        {0: "benign", 1: "dga"},
    )

    if not isinstance(label_mapping, dict):
        raise ValueError(
            "The bundle's 'label_mapping' must be a dictionary."
        )

    dga_indices = []

    for index, class_value in enumerate(classes):

        mapped_label = label_mapping.get(
            class_value,
            label_mapping.get(str(class_value), ""),
        )

        mapped_label = str(mapped_label).strip().lower()

        if mapped_label == "dga":
            dga_indices.append(index)

        elif mapped_label == "":
            if str(class_value).strip().lower() == "dga":
                dga_indices.append(index)

            elif class_value == 1:
                dga_indices.append(index)

    if len(dga_indices) != 1:
        raise ValueError(
            "Could not uniquely identify the DGA class "
            f"in estimator classes: {list(classes)}"
        )

    return dga_indices[0]


# ============================================================
# 10. BATCH MODEL INFERENCE
# ============================================================

def predict_dga_probabilities(
    estimator,
    feature_frame: pd.DataFrame,
    domains: list[str],
    model_name: str,
    bundle: dict,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    """
    Generate DGA probabilities in batches.

    Supports:
    - Numeric lexical classifiers
    - Character n-gram Logistic Regression

    The original feature extraction and model pipeline
    are retained.
    """

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    if len(feature_frame) != len(domains):
        raise ValueError(
            "Feature count does not match domain count."
        )

    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            "The selected estimator does not support predict_proba()."
        )

    probability_index = get_dga_probability_index(
        estimator,
        bundle,
    )

    probability_batches = []

    for start in range(0, len(domains), batch_size):

        end = min(
            start + batch_size,
            len(domains),
        )

        batch_features = feature_frame.iloc[start:end]

        try:

            if model_name == NGRAM_MODEL_NAME:

                # The n-gram pipeline receives the original domain
                # alongside the lexical features, matching the
                # input structure used in the original application.
                model_input = batch_features.copy()

                model_input.insert(
                    0,
                    "domain",
                    domains[start:end],
                )

            else:

                model_input = batch_features

            batch_probabilities = estimator.predict_proba(
                model_input
            )

        except Exception as exc:

            raise RuntimeError(
                f"Inference failed for rows "
                f"{start + 1}–{end}: {exc}"
            ) from exc

        batch_probabilities = np.asarray(
            batch_probabilities
        )

        if batch_probabilities.ndim != 2:
            raise ValueError(
                "predict_proba() returned an unexpected shape."
            )

        if batch_probabilities.shape[0] != end - start:
            raise ValueError(
                "Prediction count does not match batch size."
            )

        if probability_index >= batch_probabilities.shape[1]:
            raise ValueError(
                "DGA probability column is outside the "
                "returned probability matrix."
            )

        dga_probabilities = batch_probabilities[
            :, probability_index
        ]

        if not np.isfinite(dga_probabilities).all():
            raise ValueError(
                "Model produced non-finite probabilities."
            )

        if (
            (dga_probabilities < 0).any()
            or (dga_probabilities > 1).any()
        ):
            raise ValueError(
                "Model produced probabilities outside [0, 1]."
            )

        probability_batches.append(
            dga_probabilities
        )

    if not probability_batches:
        return np.array([], dtype=float)

    return np.concatenate(probability_batches).astype(float)


# ============================================================
# 11. MODEL AGREEMENT, SEVERITY AND EXPLANATIONS
# ============================================================

def get_all_model_estimators(bundle: dict) -> dict:
    """Return all available supervised estimators, with n-gram first."""
    estimators = {}
    ngram_model = bundle.get("ngram_model")
    if ngram_model is not None:
        estimators[NGRAM_MODEL_NAME] = ngram_model
    estimators.update(bundle.get("numeric_models", {}))
    return estimators


def classify_severity(probability: float) -> str:
    """
    Convert a model probability into a provisional severity band.

    Low:      < 0.25
    Medium:   0.25 to < 0.50
    High:     0.50 to < 0.75
    Critical: >= 0.75

    Severity is deliberately independent of the user-selected
    classification threshold. These bands require validation before
    being used as operational risk ratings.
    """
    probability = float(probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be finite and between 0 and 1.")
    if probability >= SEVERITY_THRESHOLDS["critical"]:
        return "Critical"
    if probability >= SEVERITY_THRESHOLDS["high"]:
        return "High"
    if probability >= SEVERITY_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def calculate_model_agreement(
    feature_frame: pd.DataFrame,
    domains: list[str],
    bundle: dict,
    threshold: float = 0.5,
    precomputed_probabilities: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Run every available supervised model and report its decision.

    Failed models are excluded from the per-row denominator and their
    errors are retained in ``DataFrame.attrs['model_failures']``.
    Agreement is model consensus, not calibrated confidence.
    """
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")
    if len(feature_frame) != len(domains):
        raise ValueError("Feature count does not match domain count.")

    estimators = get_all_model_estimators(bundle)
    if not estimators:
        raise ValueError("No supervised models are available.")

    # Reuse probabilities already computed by detect_domains. This avoids
    # running the selected classifier twice on the same batch.
    probabilities: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}

    if precomputed_probabilities:
        for model_name, values in precomputed_probabilities.items():
            if model_name not in estimators:
                continue
            values = np.asarray(values, dtype=float)
            if len(values) != len(domains):
                raise ValueError(
                    f"Precomputed probabilities for '{model_name}' "
                    "do not match the domain count."
                )
            probabilities[model_name] = values

    for model_name, estimator in estimators.items():
        if model_name in probabilities:
            continue
        try:
            model_probabilities = predict_dga_probabilities(
                estimator=estimator,
                feature_frame=feature_frame,
                domains=domains,
                model_name=model_name,
                bundle=bundle,
            )
            if len(model_probabilities) != len(domains):
                raise RuntimeError("Model returned an unexpected prediction count.")
            probabilities[model_name] = model_probabilities
        except Exception as exc:
            failures[model_name] = str(exc)

    if not probabilities:
        raise RuntimeError(f"Every model failed during inference: {failures}")

    probability_frame = pd.DataFrame(probabilities)
    flag_frame = probability_frame.ge(threshold)
    agreement = pd.DataFrame(index=probability_frame.index)
    agreement["models_flagging"] = flag_frame.sum(axis=1).astype(int)
    agreement["models_total"] = len(probabilities)
    agreement["model_agreement"] = (
        agreement["models_flagging"] / agreement["models_total"]
    )
    agreement["models_flagging_display"] = (
        agreement["models_flagging"].astype(str)
        + "/"
        + agreement["models_total"].astype(str)
    )
    for model_name in probability_frame.columns:
        agreement[f"prob_{model_name}"] = probability_frame[model_name]

    agreement.attrs["model_failures"] = failures
    return agreement.reset_index(drop=True)


def explain_ngram_prediction(
    estimator,
    domain: str,
    feature_row: pd.DataFrame,
    bundle: dict,
    top_n: int = 5,
) -> list[dict]:
    """
    Return the largest positive character-feature contributions for DGA.

    This supports a scikit-learn Pipeline whose final step is a linear
    estimator with ``coef_`` and whose preceding transformer exposes
    ``transform`` and ``get_feature_names_out``. Unsupported pipeline
    structures return an empty list rather than fabricated explanations.
    """
    if top_n <= 0 or not hasattr(estimator, "named_steps"):
        return []

    try:
        classifier = list(estimator.named_steps.values())[-1]
        if not hasattr(classifier, "coef_"):
            return []

        preprocessing = estimator[:-1]
        model_input = feature_row.copy()
        model_input.insert(0, "domain", [domain])
        transformed = preprocessing.transform(model_input)
        feature_names = preprocessing.get_feature_names_out()

        coefficients = np.asarray(classifier.coef_)
        dga_index = get_dga_probability_index(estimator, bundle)
        classes = list(classifier.classes_)

        if coefficients.shape[0] == 1:
            # Binary sklearn linear classifiers orient coef_ toward classes_[1].
            if dga_index == 0:
                coefficients = -coefficients
            weights = coefficients.ravel()
        else:
            weights = coefficients[dga_index].ravel()

        if transformed.shape[1] != len(weights) or len(feature_names) != len(weights):
            return []

        # Convert to CSR so both dense and sparse transformer output work.
        from scipy.sparse import csr_matrix
        transformed_sparse = csr_matrix(transformed)
        contribution_row = transformed_sparse.multiply(weights).getrow(0)

        ranked = [
            (str(feature_names[index]), float(value))
            for index, value in zip(
                contribution_row.indices, contribution_row.data
            )
            if value > 0
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [
            {"ngram": name, "contribution": value}
            for name, value in ranked[:top_n]
        ]
    except Exception:
        return []


# ============================================================
# 12. MAIN DOMAIN DETECTION FUNCTION
# ============================================================

def detect_domains(
    domains: list[str],
    bundle: dict,
    model_name: str,
    threshold: float = 0.5,
    include_model_agreement: bool = True,
    include_explanations: bool = True,
    defer_explanations: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classify domains using a selected supervised model.

    Results include probability, binary prediction, provisional severity,
    lexical features, model consensus and (for the n-gram model) top
    positive character-feature contributions.

    Returns:
        results, valid_feature_frame, rejected_rows

    Original input row numbers are preserved.

    Set defer_explanations=True for large batches. This avoids transforming
    every domain separately for character n-gram explanations; the UI can
    call explain_ngram_prediction() only for the domain the analyst selects.
    """
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")
    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle. Expected a dictionary.")

    estimators = get_all_model_estimators(bundle)
    if model_name not in estimators:
        raise ValueError(f"Unknown classifier: {model_name}")
    estimator = estimators[model_name]

    accepted: list[tuple[int, str]] = []
    rejected: list[dict] = []
    for index, raw in enumerate(domains):
        domain = normalise_domain(raw)
        reason = validate_domain(domain)
        if reason:
            rejected.append({"row": index + 1, "domain": str(raw), "reason": reason})
        else:
            accepted.append((index, domain))

    rejected_frame = pd.DataFrame(rejected, columns=["row", "domain", "reason"])
    feature_columns = bundle.get("feature_columns", DEFAULT_FEATURE_COLUMNS)

    if not accepted:
        empty_results = pd.DataFrame(columns=[
            "input_row", "domain", "dga_probability", "prediction", "severity",
            "models_flagging_display", "model_agreement",
        ])
        empty_features = pd.DataFrame(columns=feature_columns)
        return empty_results, empty_features, rejected_frame

    clean_domains = [domain for _, domain in accepted]
    feature_frame = build_feature_frame(clean_domains, feature_columns).reset_index(drop=True)

    probabilities = predict_dga_probabilities(
        estimator=estimator,
        feature_frame=feature_frame,
        domains=clean_domains,
        model_name=model_name,
        bundle=bundle,
    )
    if len(probabilities) != len(clean_domains):
        raise RuntimeError("Prediction count does not match valid domain count.")

    results = pd.DataFrame({
        "input_row": [index + 1 for index, _ in accepted],
        "domain": clean_domains,
        "dga_probability": probabilities,
        "prediction": np.where(
            probabilities >= threshold, "Potential DGA", "Likely benign"
        ),
        "severity": [classify_severity(p) for p in probabilities],
    })

    # Add the lexical features used by numeric models and useful to analysts.
    for column in feature_frame.columns:
        results[column] = feature_frame[column].to_numpy()

    if include_model_agreement:
        agreement = calculate_model_agreement(
            feature_frame=feature_frame,
            domains=clean_domains,
            bundle=bundle,
            threshold=threshold,
            precomputed_probabilities={model_name: probabilities},
        )
        for column in agreement.columns:
            if column not in results.columns:
                results[column] = agreement[column].to_numpy()
        results.attrs["model_failures"] = agreement.attrs.get("model_failures", {})
    else:
        results["models_flagging"] = np.nan
        results["models_total"] = 0
        results["models_flagging_display"] = "Not calculated"
        results["model_agreement"] = np.nan

    if (
        include_explanations
        and not defer_explanations
        and model_name == NGRAM_MODEL_NAME
    ):
        # Eager explanations are retained for small/single-domain analyses.
        # For large batches, callers should set defer_explanations=True and
        # explain only the domain selected by the analyst in the UI.
        results["top_ngrams"] = [
            explain_ngram_prediction(
                estimator=estimator,
                domain=domain,
                feature_row=feature_frame.iloc[[index]],
                bundle=bundle,
            )
            for index, domain in enumerate(clean_domains)
        ]
    else:
        results["top_ngrams"] = [[] for _ in clean_domains]

    if defer_explanations and include_explanations:
        results.attrs["explanations_deferred"] = True

    return results, feature_frame, rejected_frame


def detect_single_domain(
    domain: str,
    bundle: dict,
    model_name: str = NGRAM_MODEL_NAME,
    threshold: float = 0.5,
    include_model_agreement: bool = True,
    include_explanations: bool = True,
):
    """Run the standard detection pipeline for one domain."""
    results, features, rejected = detect_domains(
        domains=[domain],
        bundle=bundle,
        model_name=model_name,
        threshold=threshold,
        include_model_agreement=include_model_agreement,
        include_explanations=include_explanations,
    )
    if not results.empty:
        return results.iloc[0], features.iloc[0], None
    rejection = rejected.iloc[0].to_dict() if not rejected.empty else {
        "reason": "Domain could not be processed."
    }
    return None, None, rejection
