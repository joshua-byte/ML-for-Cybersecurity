
import hashlib
import math
import html

import pandas as pd
import streamlit as st

from detector_engine import (
    MODEL_PATH,
    MAX_DOMAINS,
    NGRAM_MODEL_NAME,
    detect_domains,
    detect_single_domain,
    get_all_model_estimators,
    explain_ngram_prediction,
    load_model_bundle,
    parse_uploaded_file,
    parse_labeled_csv,
)

from dashboard import (
    prediction_distribution,
    probability_distribution,
)

from report import (
    build_summary,
    results_csv_bytes,
    rejected_csv_bytes,
    report_html,
)

from evaluation import (
    evaluate_multiple_models,
    get_confusion_matrix,
    build_evaluation_summary,
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DGA Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM STYLING — MUTED SLATE + TEAL
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background-color: #0F172A;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Typography */

    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    h1, h2, h3, h4 {
        color: #E2E8F0 !important;
        letter-spacing: -0.3px;
    }

    p, label {
        color: #CBD5E1;
    }

    [data-testid="stCaptionContainer"] {
        color: #94A3B8;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #E2E8F0 !important;
    }

    /* Header */

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        padding: 18px 22px;
        background: linear-gradient(
            120deg,
            #1E293B 0%,
            #172033 100%
        );
        border: 1px solid #334155;
        border-radius: 14px;
        margin-bottom: 25px;
    }

    .brand-title {
        font-size: 25px;
        font-weight: 750;
        letter-spacing: 1.2px;
        color: #F1F5F9;
        margin: 0;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 12px;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 20px;
        background: #263A38;
        border: 1px solid #52786D;
        color: #A3C4B7;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.7px;
        white-space: nowrap;
    }

    /* Section headings */

    .section-heading {
        color: #E2E8F0;
        font-size: 19px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 14px;
    }

    .section-description {
        color: #94A3B8;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 16px;
    }

    /* KPI cards */

    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 19px;
        min-height: 112px;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 550;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #F1F5F9;
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -0.6px;
        line-height: 1.2;
    }

    .metric-accent {
        height: 3px;
        width: 34px;
        border-radius: 5px;
        margin-top: 13px;
    }

    /* Information panels */

    .info-panel {
        background: #1A2638;
        border: 1px solid #334155;
        border-radius: 11px;
        padding: 15px 17px;
        margin-bottom: 12px;
    }

    .info-label {
        color: #94A3B8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 5px;
    }

    .info-value {
        color: #E2E8F0;
        font-size: 14px;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    /* Upload area */

    [data-testid="stFileUploader"] {
        background: #1A2638;
        border: 1px dashed #475569;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #6B9B8A;
    }

    /* Buttons */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 40px;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background: #6B9B8A;
        border: 1px solid #83AA9B;
        color: #0F172A;
    }

    .stButton > button[kind="primary"]:hover {
        background: #83AA9B;
        border-color: #A3C4B7;
        color: #0F172A;
    }

    .stDownloadButton > button {
        background: #1E293B;
        border: 1px solid #475569;
        color: #E2E8F0;
    }

    .stDownloadButton > button:hover {
        border-color: #6B9B8A;
        color: #F1F5F9;
    }

    /* Inputs */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #1E293B;
        border-color: #475569;
        border-radius: 8px;
    }

    /* Dataframes */

    [data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Dividers */

    hr {
        border-color: #334155 !important;
    }

    /* Expanders */

    [data-testid="stExpander"] {
        background: #1A2638;
        border: 1px solid #334155;
        border-radius: 10px;
    }

    /* Footer */

    .app-footer {
        color: #94A3B8;
        font-size: 11px;
        text-align: center;
        border-top: 1px solid #334155;
        padding-top: 18px;
        margin-top: 35px;
        line-height: 1.8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HELPER FUNCTIONS FOR UI
# ============================================================

def render_metric_card(label, value, accent="#6B9B8A"):
    """Render a consistent dashboard KPI card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-accent"
                 style="background:{accent};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_panel(label, value):
    """Render a small information panel."""

    st.markdown(
        f"""
        <div class="info-panel">
            <div class="info-label">{html.escape(str(label))}</div>
            <div class="info-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="brand-title">◈ DGA DETECTION</div>
            <div class="brand-subtitle">
                SUPERVISED LEXICAL DGA DETECTION
            </div>
        </div>
        <div class="status-pill">● DETECTION SYSTEM</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. LOAD EXPORTED MODEL BUNDLE
# ============================================================

@st.cache_resource
def get_bundle():
    return load_model_bundle(MODEL_PATH)


try:
    bundle = get_bundle()

except Exception as exc:

    st.error(f"Could not load the exported models: {exc}")

    st.info(
        "Run the training notebook and its final "
        "**Export fitted supervised models** cell. "
        "It should create "
        "`models/dga_supervised_models.joblib`."
    )

    st.stop()

# ============================================================
# 6. MODEL SELECTION — N-GRAM IS THE DEFAULT
# ============================================================

model_names = [NGRAM_MODEL_NAME] + [
    name for name in bundle["numeric_models"].keys()
    if name != NGRAM_MODEL_NAME
]

if not model_names:
    st.error("No supervised classifiers are available.")
    st.stop()


# ============================================================
# 7. SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.markdown("## Detection configuration")
    st.caption("Configure supervised lexical analysis.")
    st.divider()

    st.markdown("#### Classification model")
    model_name = st.selectbox(
        "Select classifier",
        model_names,
        index=0,
    )

    st.markdown("#### Decision threshold")
    threshold = st.slider(
        "DGA probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        format="%.2f",
    )

    st.caption(
        "The threshold controls the predicted label. "
        "It does not change the model's probability estimate."
    )

    st.divider()
    st.markdown("#### Severity bands")
    st.caption(
        "Provisional probability bands: Low <25%, Medium 25–49%, "
        "High 50–74%, Critical ≥75%. These are separate from the "
        "decision threshold and require validation before operational use."
    )

    st.divider()
    st.markdown("#### Input requirements")
    st.markdown(
        """
        **Batch CSV/TXT**
        - CSV domain columns: `domain`, `domain_name`, `host`, `hostname`, etc.
        - TXT requires one domain per line.

        **Evaluation CSV**
        - Domain column is detected automatically (`host` is preferred when present).
        - Label columns: `isDGA`, `label`, `class`, or `target`.
        - Labels: `0`/`benign`/`legit` or `1`/`dga`.
        """
    )

    st.divider()
    render_info_panel("Maximum input size", f"{MAX_DOMAINS:,} domain rows")
    st.caption(
        "The current inference pipeline uses supervised classifiers. "
        "The bundled Isolation Forest artifact is not loaded or used."
    )


# ============================================================
# 8. ANALYSIS MODE AND INPUT
# ============================================================

st.markdown(
    '<div class="section-heading">Analysis mode</div>',
    unsafe_allow_html=True,
)

mode_col, help_col = st.columns([1, 2])

with mode_col:
    single_domain_mode = st.checkbox(
        "Single-domain analysis",
        value=False,
        help="Analyse one domain and inspect its probability and contributing features.",
    )

with help_col:
    evaluation_mode = False
    if not single_domain_mode:
        evaluation_mode = st.checkbox(
            "Evaluation mode (labelled CSV)",
            value=False,
            help="Evaluate model predictions against ground-truth labels in a CSV.",
        )

st.divider()

domains = None
labels = None
ingest_warnings = []
uploaded_file = None
single_domain_value = ""

if single_domain_mode:
    st.markdown(
        '<div class="section-heading">Single-domain analysis</div>',
        unsafe_allow_html=True,
    )
    single_domain_value = st.text_input(
        "Domain to analyse",
        placeholder="example.com",
        help="Enter one bare domain name. URL and hostname normalization is a separate robustness improvement.",
    ).strip()
    run_label = "Analyse domain"

else:
    st.markdown(
        '<div class="section-heading">Dataset ingestion</div>',
        unsafe_allow_html=True,
    )
    if evaluation_mode:
        st.markdown(
            '<div class="section-description">Upload a labelled CSV. The domain and ground-truth label columns are detected automatically.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload labelled evaluation CSV",
            type=["csv"],
            help="Domain headers may include domain, domain_name, host or hostname. Label headers may include isDGA, label, class or target. Labels may be 0/1, benign/legit or DGA.",
            key="evaluation_upload",
        )
    else:
        st.markdown(
            '<div class="section-description">Upload a domain dataset for classification.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload domain dataset",
            type=["csv", "txt"],
            help=f"CSV: domain column. TXT: one domain per line. Maximum {MAX_DOMAINS:,} rows.",
            key="batch_upload",
        )
    run_label = "Run evaluation" if evaluation_mode else "Run DGA detection"


# ============================================================
# 9. INPUT PARSING AND ANALYSIS IDENTITY
# ============================================================

file_fingerprint = None
input_name = "Single-domain input"

if single_domain_mode:
    current_input_key = (
        "single",
        single_domain_value,
        model_name,
        float(threshold),
    )
elif uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_fingerprint = hashlib.sha256(file_bytes).hexdigest()
    input_name = uploaded_file.name
    current_input_key = (
        "evaluation" if evaluation_mode else "batch",
        file_fingerprint,
        model_name,
        float(threshold),
    )
else:
    current_input_key = None

if uploaded_file is not None:
    try:
        if evaluation_mode:
            # Let detector_engine identify the domain and label columns.
            # This supports datasets such as host/isDGA as well as
            # domain/label without requiring users to rename CSV headers.
            domains, labels, ingest_warnings = parse_labeled_csv(uploaded_file)
        else:
            domains, ingest_warnings = parse_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"Could not ingest this file: {exc}")
        st.stop()

    file_col, count_col = st.columns([2, 1])
    with file_col:
        render_info_panel("Uploaded dataset", uploaded_file.name)
    with count_col:
        render_info_panel("Rows found", f"{len(domains):,}")

    for warning in ingest_warnings:
        st.warning(warning)


# ============================================================
# 10. RUN DETECTION / EVALUATION
# ============================================================

can_run = (
    (single_domain_mode and bool(single_domain_value))
    or (not single_domain_mode and uploaded_file is not None)
)

if st.button(
    run_label,
    type="primary",
    use_container_width=True,
    disabled=not can_run,
):
    try:
        if single_domain_mode:
            with st.spinner("Analysing domain..."):
                result, feature_row, rejection = detect_single_domain(
                    domain=single_domain_value,
                    bundle=bundle,
                    model_name=model_name,
                    threshold=threshold,
                    include_model_agreement=True,
                    include_explanations=True,
                )

            if rejection is not None:
                st.error(
                    f"Domain rejected: {rejection.get('reason', 'Invalid domain')}"
                )
                st.stop()

            results = pd.DataFrame([result.to_dict()])
            features = pd.DataFrame([feature_row.to_dict()])
            rejected = pd.DataFrame(columns=["row", "domain", "reason"])
            evaluation_labels = None

        else:
            with st.spinner(
                f"Analysing {len(domains):,} domain rows across available models..."
            ):
                results, features, rejected = detect_domains(
                    domains=domains,
                    bundle=bundle,
                    model_name=model_name,
                    threshold=threshold,
                    include_model_agreement=True,
                    include_explanations=True,
                    defer_explanations=True,
                )

            evaluation_labels = labels if evaluation_mode else None

        st.session_state["dga_results"] = results
        st.session_state["dga_features"] = features
        st.session_state["dga_rejected"] = rejected
        st.session_state["dga_analysis_key"] = current_input_key
        st.session_state["dga_model_name"] = model_name
        st.session_state["dga_threshold"] = float(threshold)
        st.session_state["dga_filename"] = input_name
        st.session_state["dga_mode"] = (
            "single" if single_domain_mode
            else "evaluation" if evaluation_mode
            else "batch"
        )
        st.session_state["dga_labels"] = evaluation_labels
        st.session_state["dga_page"] = 0

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()


# ============================================================
# 11. CHECK STORED RESULTS AND STALE SETTINGS
# ============================================================

if "dga_results" not in st.session_state:
    if single_domain_mode and not single_domain_value:
        st.info("Enter a domain above to begin single-domain analysis.")
    elif not single_domain_mode and uploaded_file is None:
        st.info("Upload a dataset to begin analysis.")
    else:
        st.info(f"Input ready. Select **{run_label}** to continue.")
    st.stop()

if st.session_state.get("dga_analysis_key") != current_input_key:
    st.warning(
        "The input or detection settings have changed since the previous run. "
        "Run the analysis again to refresh the results."
    )
    st.stop()

results = st.session_state["dga_results"]
features = st.session_state["dga_features"]
rejected = st.session_state["dga_rejected"]
used_model = st.session_state["dga_model_name"]
used_threshold = st.session_state["dga_threshold"]
used_filename = st.session_state["dga_filename"]
used_mode = st.session_state["dga_mode"]
stored_labels = st.session_state.get("dga_labels")

if results.empty:
    st.warning("No valid domains were available for classification.")
    if not rejected.empty:
        st.dataframe(rejected, use_container_width=True, hide_index=True)
    st.stop()


# ============================================================
# 12. DETECTION OVERVIEW
# ============================================================

summary = build_summary(results, rejected)

st.divider()
st.markdown(
    '<div class="section-heading">Detection overview</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    render_metric_card(
        "DOMAINS ANALYSED",
        f"{summary['total_processed']:,}",
        "#8B9BB4",
    )

with m2:
    render_metric_card(
        "POTENTIAL DGA",
        f"{summary['potential_dga']:,}",
        "#C58B75",
    )

with m3:
    render_metric_card(
        "LIKELY BENIGN",
        f"{summary['likely_benign']:,}",
        "#6B9B8A",
    )

with m4:
    render_metric_card(
        "MEAN DGA PROBABILITY",
        f"{summary['mean_dga_probability']:.1%}",
        "#8B9BB4",
    )

st.caption(
    f"Model: **{used_model}**  ·  "
    f"Threshold: **{used_threshold:.0%}**  ·  "
    f"Input: **{used_filename}**"
)

# Model agreement execution diagnostics.
model_failures = results.attrs.get("model_failures", {})
if model_failures:
    with st.expander("Model execution warnings"):
        st.warning(
            "Some classifiers did not produce predictions. "
            "Agreement is calculated only across models that succeeded."
        )
        st.json(model_failures)


# ============================================================
# 13. CLASSIFICATION ANALYTICS
# ============================================================

st.divider()
st.markdown(
    '<div class="section-heading">Classification analytics</div>',
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown("#### Classification split")
    st.plotly_chart(
        prediction_distribution(results),
        use_container_width=True,
    )

with chart_right:
    st.markdown("#### DGA probability distribution")
    st.plotly_chart(
        probability_distribution(results, threshold=used_threshold),
        use_container_width=True,
    )


# ============================================================
# 14. HIGHEST-SCORING DOMAINS
# ============================================================

st.divider()
st.markdown(
    '<div class="section-heading">Highest-scoring domains</div>',
    unsafe_allow_html=True,
)

top_columns = [
    column for column in [
        "domain",
        "dga_probability",
        "prediction",
        "severity",
        "models_flagging_display",
    ]
    if column in results.columns
]

top_domains = (
    results.sort_values(
        "dga_probability",
        ascending=False,
        kind="mergesort",
    )[top_columns]
    .head(15)
    .reset_index(drop=True)
)

top_domains.insert(0, "Rank", range(1, len(top_domains) + 1))

st.dataframe(
    top_domains,
    column_config={
        "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
        "domain": st.column_config.TextColumn("Domain", width="large"),
        "dga_probability": st.column_config.ProgressColumn(
            "DGA probability", format="%.1f%%", min_value=0, max_value=1
        ),
        "prediction": st.column_config.TextColumn("Classification"),
        "severity": st.column_config.TextColumn("Severity"),
        "models_flagging_display": st.column_config.TextColumn("Models flagging"),
    },
    use_container_width=True,
    hide_index=True,
    height=440,
)


# ============================================================
# 15. RESULTS EXPLORER
# ============================================================

st.divider()
st.markdown(
    '<div class="section-heading">Detection results explorer</div>',
    unsafe_allow_html=True,
)

filter_col, search_col, sort_col, page_size_col = st.columns(
    [1.15, 1.5, 1.25, 0.9]
)

with filter_col:
    filter_choice = st.selectbox(
        "Prediction filter",
        ["All domains", "Potential DGA only", "Likely benign only"],
    )

with search_col:
    search_query = st.text_input(
        "Search domain",
        placeholder="Enter a domain...",
    ).strip()

with sort_col:
    sort_choice = st.selectbox(
        "Sort results",
        [
            "Highest DGA probability",
            "Lowest DGA probability",
            "Domain A–Z",
            "Input order",
        ],
    )

with page_size_col:
    page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)

view = results.copy()

if filter_choice == "Potential DGA only":
    view = view[view["prediction"] == "Potential DGA"]
elif filter_choice == "Likely benign only":
    view = view[view["prediction"] == "Likely benign"]

if search_query:
    view = view[
        view["domain"].astype(str).str.contains(
            search_query,
            case=False,
            regex=False,
            na=False,
        )
    ]

if sort_choice == "Highest DGA probability":
    view = view.sort_values("dga_probability", ascending=False, kind="mergesort")
elif sort_choice == "Lowest DGA probability":
    view = view.sort_values("dga_probability", ascending=True, kind="mergesort")
elif sort_choice == "Domain A–Z":
    view = view.sort_values("domain", ascending=True, kind="mergesort")
else:
    view = view.sort_values("input_row", ascending=True, kind="mergesort")

view_signature = (filter_choice, search_query, sort_choice, int(page_size))
if st.session_state.get("dga_view_signature") != view_signature:
    st.session_state["dga_page"] = 0
    st.session_state["dga_view_signature"] = view_signature

total_rows = len(view)
total_pages = max(1, math.ceil(total_rows / page_size))
current_page = min(st.session_state.get("dga_page", 0), total_pages - 1)
st.session_state["dga_page"] = current_page

start_index = current_page * page_size
end_index = min(start_index + page_size, total_rows)
page_view = view.iloc[start_index:end_index]

if total_rows:
    st.caption(
        f"Showing {start_index + 1:,}–{end_index:,} of {total_rows:,} "
        f"matching domains · Page {current_page + 1} of {total_pages}"
    )
else:
    st.caption("No domains match the selected filters.")

display_columns = [
    column for column in [
        "input_row",
        "domain",
        "dga_probability",
        "prediction",
        "severity",
        "models_flagging_display",
    ]
    if column in page_view.columns
]

st.dataframe(
    page_view[display_columns],
    column_config={
        "input_row": st.column_config.NumberColumn("Input row", format="%d"),
        "domain": st.column_config.TextColumn("Domain", width="large"),
        "dga_probability": st.column_config.ProgressColumn(
            "DGA probability", format="%.1f%%", min_value=0, max_value=1
        ),
        "prediction": st.column_config.TextColumn("Classification"),
        "severity": st.column_config.TextColumn("Severity"),
        "models_flagging_display": st.column_config.TextColumn("Models flagging"),
    },
    use_container_width=True,
    hide_index=True,
    height=440,
)

previous_col, page_info_col, next_col = st.columns([1, 2, 1])

with previous_col:
    if st.button("← Previous", disabled=current_page == 0, use_container_width=True):
        st.session_state["dga_page"] = current_page - 1
        st.rerun()

with page_info_col:
    st.markdown(
        f"<div style='text-align:center;padding-top:10px;color:#94A3B8;'>"
        f"Page {current_page + 1} of {total_pages}</div>",
        unsafe_allow_html=True,
    )

with next_col:
    if st.button(
        "Next →",
        disabled=current_page >= total_pages - 1,
        use_container_width=True,
    ):
        st.session_state["dga_page"] = current_page + 1
        st.rerun()


# ============================================================
# 16. DOMAIN EXPLANATION / FEATURE INSPECTION
# ============================================================

st.divider()
st.markdown(
    '<div class="section-heading">Why was this domain flagged?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-description">'
    'Inspect the lexical measurements and, when the n-gram model is selected, '
    'the strongest positive character-feature contributions.'
    '</div>',
    unsafe_allow_html=True,
)

if used_mode == "single":
    selected_domain = str(results.iloc[0]["domain"])
    st.caption(f"Inspecting: {selected_domain}")
else:
    domain_options = results["domain"].astype(str).tolist()
    selected_domain = st.selectbox(
        "Select a domain to inspect",
        domain_options,
        index=0,
    )

selected_result_index = results.index[
    results["domain"].astype(str) == selected_domain
][0]
selected_row = results.loc[selected_result_index]

feature_columns = [
    "domain_length",
    "entropy",
    "digit_ratio",
    "max_consonant_sequence",
    "digit_count",
    "vowel_count",
    "consonant_count",
    "unique_char_ratio",
    "hyphen_count",
]

available_features = [
    column for column in feature_columns if column in selected_row.index
]

feature_display = pd.DataFrame(
    {
        "Feature": available_features,
        "Value": [selected_row[column] for column in available_features],
    }
)

feature_col, explanation_col = st.columns([1, 1])

with feature_col:
    st.markdown("#### Lexical features")
    st.dataframe(feature_display, hide_index=True, use_container_width=True)

with explanation_col:
    st.markdown("#### Model explanation")
    ngram_explanation = selected_row.get("top_ngrams", [])

    # Batch explanations are deferred during inference for performance.
    # Calculate the explanation only for the domain currently selected.
    if (
        used_model == NGRAM_MODEL_NAME
        and (not isinstance(ngram_explanation, list) or not ngram_explanation)
    ):
        selected_estimator = get_all_model_estimators(bundle).get(used_model)
        if selected_estimator is not None:
            ngram_explanation = explain_ngram_prediction(
                estimator=selected_estimator,
                domain=selected_domain,
                feature_row=features.iloc[[selected_result_index]],
                bundle=bundle,
            )

    if isinstance(ngram_explanation, list) and ngram_explanation:
        explanation_frame = pd.DataFrame(ngram_explanation)
        st.dataframe(
            explanation_frame.rename(
                columns={
                    "ngram": "Character n-gram",
                    "contribution": "Positive contribution",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "These are positive coefficient-weighted feature contributions, "
            "not a complete causal explanation of the prediction."
        )
    elif used_model == NGRAM_MODEL_NAME:
        st.info(
            "The selected pipeline did not expose compatible feature "
            "contribution information for this domain."
        )
    else:
        st.info(
            "Top character n-gram contributions are available when the "
            "Character n-gram Logistic Regression model is selected."
        )

st.caption(
    f"Selected probability: {float(selected_row['dga_probability']):.2%} · "
    f"Classification: {selected_row['prediction']} · "
    f"Severity band: {selected_row.get('severity', 'Unavailable')} · "
    f"Models flagging: {selected_row.get('models_flagging_display', 'Unavailable')}"
)


# ============================================================
# 17. EVALUATION MODE
# ============================================================

if used_mode == "evaluation":
    st.divider()
    st.markdown(
        '<div class="section-heading">Evaluation against ground-truth labels</div>',
        unsafe_allow_html=True,
    )

    if stored_labels is None:
        st.error("Ground-truth labels were not retained for this evaluation.")
    else:
        # The engine preserves original input row numbers. Use them to align
        # labels after invalid domain rows have been excluded.
        valid_positions = results["input_row"].astype(int).to_numpy() - 1
        if (
            len(valid_positions) != len(results)
            or (valid_positions < 0).any()
            or (valid_positions >= len(stored_labels)).any()
        ):
            st.error(
                "Could not align ground-truth labels with the valid input rows. "
                "Please check the uploaded CSV and rerun evaluation."
            )
            st.stop()

        y_true = pd.Series(stored_labels).iloc[valid_positions].to_numpy()

        # Collect probability outputs for every model that successfully ran.
        # The engine preserves the original input row in `input_row`, so labels
        # remain aligned after invalid domains have been rejected.
        probabilities_by_model = {
            name: results[f"prob_{name}"].to_numpy()
            for name in get_all_model_estimators(bundle)
            if f"prob_{name}" in results.columns
        }

        if probabilities_by_model:
            evaluations = evaluate_multiple_models(
                y_true=y_true,
                probabilities_by_model=probabilities_by_model,
                threshold=used_threshold,
            )
            comparison = build_evaluation_summary(evaluations)

            st.markdown("#### Model comparison on this labelled dataset")
            st.dataframe(
                comparison,
                column_config={
                    "Precision": st.column_config.NumberColumn(
                        "Precision", format="%.2%"
                    ),
                    "Recall": st.column_config.NumberColumn(
                        "Recall", format="%.2%"
                    ),
                    "F1-score": st.column_config.NumberColumn(
                        "F1-score", format="%.2%"
                    ),
                    "False-positive rate": st.column_config.NumberColumn(
                        "False-positive rate", format="%.2%"
                    ),
                    "Specificity": st.column_config.NumberColumn(
                        "Specificity", format="%.2%"
                    ),
                    "Accuracy": st.column_config.NumberColumn(
                        "Accuracy", format="%.2%"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

            chosen_metrics = evaluations.get(used_model)
            if chosen_metrics is not None:
                st.markdown(f"#### Confusion matrix — {used_model}")
                confusion = get_confusion_matrix(chosen_metrics)
                st.dataframe(confusion, use_container_width=True)

                metric_left, metric_right = st.columns(2)
                with metric_left:
                    st.metric(
                        "Accuracy",
                        f"{chosen_metrics['accuracy']:.2%}",
                    )
                    st.metric(
                        "Precision",
                        f"{chosen_metrics['precision']:.2%}",
                    )
                with metric_right:
                    st.metric(
                        "Recall",
                        f"{chosen_metrics['recall']:.2%}",
                    )
                    st.metric(
                        "F1-score",
                        f"{chosen_metrics['f1_score']:.2%}",
                    )
            else:
                st.warning(
                    f"No probability output was available for the selected model "
                    f"({used_model})."
                )

            st.caption(
                f"Evaluation rows: {len(y_true):,}. "
                f"Invalid domains excluded: {len(rejected):,}. "
                "All model metrics use the same valid rows and selected threshold. "
                "Results describe this uploaded labelled dataset and are not "
                "necessarily independent test-set metrics."
            )
        else:
            st.warning(
                "No model probability columns were available for evaluation."
            )


# ============================================================
# 18. EXPORT ANALYSIS
# ============================================================

st.divider()
st.markdown(
    '<div class="section-heading">Export analysis</div>',
    unsafe_allow_html=True,
)

download_left, download_middle, download_right = st.columns(3)

with download_left:
    st.download_button(
        "Download all results as CSV",
        data=results_csv_bytes(results),
        file_name="dga_detection_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_middle:
    st.download_button(
        "Download HTML report",
        data=report_html(
            results,
            used_model,
            used_threshold,
            rejected=rejected,
            filename=used_filename,
        ),
        file_name="dga_detection_report.html",
        mime="text/html",
        use_container_width=True,
    )

with download_right:
    if not rejected.empty:
        st.download_button(
            "Download rejected rows as CSV",
            data=rejected_csv_bytes(rejected),
            file_name="dga_rejected_rows.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button(
            "No rejected rows",
            disabled=True,
            use_container_width=True,
        )


# ============================================================
# 19. REJECTED INPUT ROWS
# ============================================================

if not rejected.empty:
    with st.expander(f"Skipped invalid rows ({len(rejected):,})"):
        st.caption(
            "These rows were excluded because they did not pass domain validation."
        )
        st.dataframe(
            rejected,
            use_container_width=True,
            hide_index=True,
            height=300,
        )


# ============================================================
# 20. SAVED MODEL EVALUATION METRICS
# ============================================================

with st.expander("Model evaluation from the training notebook"):
    metric_rows = []

    for name, values in bundle.get("evaluation_metrics", {}).items():
        metric_rows.append({"Model": name, **values})

    ngram_metrics = bundle.get("ngram_metrics", {})
    if ngram_metrics:
        metric_rows.append({
            "Model": NGRAM_MODEL_NAME,
            **ngram_metrics,
        })

    if metric_rows:
        st.dataframe(
            pd.DataFrame(metric_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No saved evaluation metrics were found.")


# ============================================================
# 21. FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        <strong>DOMAIN SENTINEL</strong>
        <br>
        Supervised lexical DGA detection
        <br>
        The current inference pipeline uses supervised classifiers only.
        The bundled Isolation Forest artifact is not loaded or used.
        <br>
        Model probability and model agreement are indicators, not proof
        that a domain is malicious.
        <br>
        Severity bands are provisional and should not be treated as
        calibrated operational risk ratings.
    </div>
    """,
    unsafe_allow_html=True,
)
