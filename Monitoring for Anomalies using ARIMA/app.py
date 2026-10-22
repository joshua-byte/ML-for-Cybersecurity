
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from capture import capture_traffic, list_interfaces
from report import generate_html_report


# ============================================================
# Configuration
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "arima_model.pkl"

MODEL_TARGET = "packet_count"
MODEL_INTERVAL_SECONDS = 60
REPORT_DIR = APP_DIR / "reports"


# ============================================================
# Page configuration and styling
# ============================================================

st.set_page_config(
    page_title="Network Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128,128,128,.22);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: .13em;
            font-size: .72rem;
            font-weight: 700;
            opacity: .68;
            margin-bottom: .3rem;
        }

        .page-title {
            font-size: 2.1rem;
            line-height: 1.15;
            font-weight: 720;
            margin-bottom: .4rem;
        }

        .page-subtitle {
            opacity: .72;
            margin-bottom: 1.4rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.22);
            padding: .9rem 1rem;
            border-radius: .8rem;
            background: rgba(128,128,128,.035);
        }

        .status-pill {
            display: inline-block;
            border: 1px solid rgba(128,128,128,.35);
            border-radius: 999px;
            padding: .25rem .65rem;
            font-size: .78rem;
            font-weight: 600;
            margin-right: .3rem;
        }

        .muted {
            opacity: .7;
            font-size: .88rem;
        }

        hr {
            margin: 1.25rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model loading
# ============================================================

@st.cache_resource(show_spinner="Loading forecasting model...")
def load_model_artifact(path: str) -> dict[str, Any]:
    """
    Load the saved model artifact.

    Only load trusted pickle files. Pickle can execute code when
    deserialized, so never load an artifact from an untrusted source.
    """
    with open(path, "rb") as model_file:
        artifact = pickle.load(model_file)

    required = {
        "model",
        "order",
        "target",
        "threshold_95",
        "threshold_99",
        "training_observations",
    }

    missing = required.difference(artifact.keys())

    if missing:
        raise ValueError(
            "Model artifact is missing required fields: "
            + ", ".join(sorted(missing))
        )

    for key in ("threshold_95", "threshold_99"):
        value = float(artifact[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid model threshold: {key}")

    return artifact


# ============================================================
# Forecasting and anomaly scoring
# ============================================================

def score_series(
    values: np.ndarray,
    threshold: float,
    initial_model: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling one-step forecasting.

    Each observation is forecast before it is incorporated into the
    model state. Model parameters remain fixed; refitting is disabled.
    """
    if not np.isfinite(values).all():
        raise ValueError("The series contains non-finite values.")

    model = initial_model
    predictions = []
    residuals = []
    alerts = []

    for value in values:
        prediction = float(
            np.asarray(model.forecast(steps=1)).reshape(-1)[0]
        )

        residual = float(value) - prediction

        predictions.append(prediction)
        residuals.append(residual)
        alerts.append(abs(residual) > threshold)

        model = model.append([float(value)], refit=False)

    return (
        np.asarray(predictions, dtype=float),
        np.asarray(residuals, dtype=float),
        np.asarray(alerts, dtype=bool),
    )


def prepare_scored_results(
    frame: pd.DataFrame,
    value_column: str,
    threshold: float,
    initial_model: Any,
) -> pd.DataFrame:
    """Validate and score one ordered traffic time series."""
    if frame.empty:
        raise ValueError("The dataset contains no rows.")

    if value_column not in frame.columns:
        raise ValueError(f"Column '{value_column}' does not exist.")

    values = pd.to_numeric(frame[value_column], errors="coerce")

    if values.isna().any():
        raise ValueError(
            f"Column '{value_column}' contains missing or non-numeric values. "
            "Clean the data before scoring."
        )

    numeric_values = values.to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("The selected feature contains infinite values.")

    if (numeric_values < 0).any():
        raise ValueError("Traffic counts must not be negative.")

    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("The selected threshold is invalid.")

    output = frame.copy().reset_index(drop=True)

    predictions, residuals, alerts = score_series(
        numeric_values,
        threshold,
        initial_model,
    )

    output["model_prediction"] = predictions
    output["residual"] = residuals
    output["absolute_error"] = np.abs(residuals)
    output["anomaly_alert"] = alerts

    output["alert_direction"] = np.select(
        [
            residuals > threshold,
            residuals < -threshold,
        ],
        [
            "higher-than-expected",
            "lower-than-expected",
        ],
        default="normal",
    )

    return output


# ============================================================
# Visualization
# ============================================================

def make_traffic_chart(
    results: pd.DataFrame,
    observed_column: str,
) -> plt.Figure:
    """Create the observed-versus-forecast traffic chart."""
    timestamp_column = next(
        (
            column
            for column in (
                "interval_start_utc",
                "timestamp_utc",
                "timestamp",
                "datetime",
                "time",
            )
            if column in results.columns
        ),
        None,
    )

    if timestamp_column is not None:
        timestamps = pd.to_datetime(
            results[timestamp_column],
            errors="coerce",
            utc=True,
        )

        if timestamps.notna().all():
            x_values = timestamps
            x_label = "Time (UTC)"
        else:
            x_values = np.arange(len(results))
            x_label = "Interval"
    else:
        x_values = np.arange(len(results))
        x_label = "Interval"

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(
        x_values,
        results[observed_column].to_numpy(),
        label="Observed traffic",
        linewidth=1.7,
    )

    ax.plot(
        x_values,
        results["model_prediction"].to_numpy(),
        label="One-step forecast",
        linestyle="--",
        linewidth=1.5,
    )

    flagged = results["anomaly_alert"].to_numpy(dtype=bool)

    if flagged.any():
        ax.scatter(
            np.asarray(x_values)[flagged],
            results.loc[flagged, observed_column],
            label="Anomaly alert",
            s=28,
            zorder=3,
        )

    ax.set_title(
        "Observed Traffic vs. Forecast",
        loc="left",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(observed_column.replace("_", " ").title())
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    return fig


# ============================================================
# Report generation and downloads
# ============================================================

def build_report_metadata(
    source: str,
    threshold: float,
    artifact: dict[str, Any],
    interface: str | None = None,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Build metadata shared with report.py."""
    return {
        "source": source,
        "interface": interface or "Not applicable",
        "scenario": scenario or artifact.get("scenario", "Not recorded"),
        "model_name": "ARIMA",
        "model_order": str(tuple(artifact["order"])),
        "model_version": artifact.get("model_version", "Baseline"),
        "threshold": threshold,
        "threshold_method": artifact.get(
            "calibration_method",
            "Absolute forecast residual percentile",
        ),
        "calibration_dataset": artifact.get(
            "scenario",
            "CESNET-CC25 baseline scenario",
        ),
        "interval_seconds": MODEL_INTERVAL_SECONDS,
    }


def render_exports(
    results: pd.DataFrame,
    observed_column: str,
    metadata: dict[str, Any],
) -> None:
    """Render downloadable HTML report and CSV exports."""
    st.subheader("Export report")

    try:
        html_report = generate_html_report(
            results=results,
            target_column=observed_column,
            metadata=metadata,
        )
    except Exception as exc:
        st.error(f"Could not generate the HTML report: {exc}")
        html_report = None

    alerts = results.loc[results["anomaly_alert"]].copy()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.download_button(
            "Download scored data",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="network_monitor_scored.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_b:
        st.download_button(
            "Download alerts",
            data=alerts.to_csv(index=False).encode("utf-8"),
            file_name="network_monitor_alerts.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_c:
        if html_report is not None:
            st.download_button(
                "Download detailed HTML report",
                data=html_report.encode("utf-8"),
                file_name="network_monitor_report.html",
                mime="text/html",
                use_container_width=True,
            )
        else:
            st.button(
                "HTML report unavailable",
                disabled=True,
                use_container_width=True,
            )


# ============================================================
# Results dashboard
# ============================================================

def render_results(
    results: pd.DataFrame,
    observed_column: str,
    metadata: dict[str, Any],
) -> None:
    """Render traffic metrics, charts, alerts, and report downloads."""
    row_count = len(results)
    alert_frame = results.loc[results["anomaly_alert"]].copy()
    alert_count = len(alert_frame)

    observed = results[observed_column].astype(float)
    residuals = results["residual"].astype(float)

    alert_rate = alert_count / max(row_count, 1) * 100
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))

    st.markdown("### Analysis summary")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Intervals analyzed", f"{row_count:,}")
    m2.metric("Anomaly alerts", f"{alert_count:,}")
    m3.metric("Alert rate", f"{alert_rate:.2f}%")
    m4.metric("Forecast MAE", f"{mae:.2f}")
    m5.metric("Peak traffic", f"{observed.max():,.0f}")

    st.pyplot(
        make_traffic_chart(results, observed_column),
        use_container_width=True,
    )

    st.caption(
        f"Forecast RMSE: {rmse:.2f}. These metrics describe forecasting "
        "errors, not confirmed attack-detection performance."
    )

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("### Alert review")

        if alert_frame.empty:
            st.success("No intervals exceeded the selected threshold.")
        else:
            st.dataframe(
                alert_frame,
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.markdown("### Alert distribution")

        if alert_frame.empty:
            st.write("No alerts to summarize.")
        else:
            counts = (
                alert_frame["alert_direction"]
                .value_counts()
                .rename_axis("Direction")
                .reset_index(name="Intervals")
            )

            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True,
            )

    st.warning(
        "An anomaly is a statistical deviation from expected traffic. "
        "It is not, by itself, evidence of C2 traffic or another attack."
    )

    st.divider()
    render_exports(results, observed_column, metadata)


# ============================================================
# Sidebar and model initialization
# ============================================================

with st.sidebar:
    st.markdown("## NETWORK MONITOR")
    st.caption("Traffic analytics and anomaly monitoring")
    st.divider()

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Analyze dataset",
            "Live capture",
            "Model information",
        ],
        key="workspace",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### Detection sensitivity")

    threshold_label = st.selectbox(
        "Residual threshold",
        ["95th percentile", "99th percentile"],
        index=1,
        help="The 95th-percentile threshold generally produces more alerts.",
    )

try:
    artifact = load_model_artifact(str(MODEL_PATH))

    threshold = float(
        artifact["threshold_95"]
        if threshold_label.startswith("95th")
        else artifact["threshold_99"]
    )

except Exception as exc:
    st.error(f"Unable to load the model artifact: {exc}")
    st.info(
        "Ensure cesnet_arima_model.pkl is in the same directory as app.py "
        "and was generated by a trusted training process."
    )
    st.stop()

with st.sidebar:
    st.metric("Active threshold", f"{threshold:.2f}")
    st.caption(f"Model: ARIMA{tuple(artifact['order'])}")
    st.caption(
        f"Training observations: "
        f"{int(artifact['training_observations']):,}"
    )
    st.divider()
    st.caption("Experimental monitoring. Alerts require contextual review.")


# ============================================================
# Main header
# ============================================================

st.markdown(
    '<div class="eyebrow">Network operations</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-title">Network Monitor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-subtitle">'
    "Monitor traffic patterns, compare observations with forecasts, "
    "and generate detailed analysis reports."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Overview
# ============================================================

if page == "Overview":
    st.subheader("Monitoring overview")

    left, right = st.columns([1.6, 1])

    with left:
        st.markdown("#### Network traffic analytics")

        st.write(
            "Network Monitor uses an ARIMA time-series model to forecast "
            "traffic volume one interval ahead. Large forecast residuals "
            "are flagged for investigation."
        )

        st.markdown(
            '<span class="status-pill">Model loaded</span>'
            '<span class="status-pill">Forecasting enabled</span>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Active model")

        st.write(f"**Model:** ARIMA{tuple(artifact['order'])}")
        st.write(
            f"**Feature:** `{artifact.get('target', MODEL_TARGET)}`"
        )
        st.write(
            f"**Training observations:** "
            f"{int(artifact['training_observations']):,}"
        )
        st.write("**Interval:** 60 seconds")

    st.divider()
    st.subheader("Workspaces")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Dataset analysis")
        st.write(
            "Upload a CSV of pre-aggregated traffic and review "
            "forecasts and anomaly alerts."
        )

    with c2:
        st.markdown("#### Live capture")
        st.write(
            "Capture traffic from an authorized network interface "
            "on the machine running this application."
        )

    with c3:
        st.markdown("#### Detailed reports")
        st.write(
            "Generate a standalone HTML report with traffic statistics, "
            "forecast charts, alert records, and methodology."
        )

    st.divider()
    st.subheader("Current model limitations")

    st.warning(
        "The saved model was trained on one-minute packet counts from "
        "a single CESNET-CC25 scenario. Results on other datasets or "
        "networks must be validated before drawing operational conclusions."
    )


# ============================================================
# CSV analysis
# ============================================================

elif page == "Analyze dataset":
    st.subheader("Analyze a traffic dataset")

    st.write(
        "Upload a single, chronologically ordered capture. The current "
        "model expects one-minute packet_count observations. Keep separate "
        "captures and scenarios in separate files."
    )

    uploaded_file = st.file_uploader(
        "Upload pre-aggregated traffic CSV",
        type=["csv"],
        help="The current artifact is trained for packet_count.",
    )

    if uploaded_file is not None:
        try:
            frame = pd.read_csv(uploaded_file)

            if frame.empty:
                st.error("The uploaded CSV contains no rows.")
            else:
                a, b, c = st.columns(3)
                a.metric("Rows", f"{len(frame):,}")
                b.metric("Columns", f"{len(frame.columns):,}")

                timestamp_present = any(
                    column in frame.columns
                    for column in (
                        "interval_start_utc",
                        "timestamp",
                        "timestamp_utc",
                        "datetime",
                        "time",
                    )
                )

                c.metric(
                    "Timestamp column",
                    "Detected" if timestamp_present else "Not detected",
                )

                st.markdown("#### Dataset preview")
                st.dataframe(
                    frame.head(12),
                    use_container_width=True,
                    hide_index=True,
                )

                target_column = artifact.get("target", MODEL_TARGET)

                if target_column not in frame.columns:
                    st.error(
                        f"The model requires a `{target_column}` column. "
                        "This model cannot directly score a different feature."
                    )
                else:
                    if st.button(
                        "Run analysis",
                        type="primary",
                        use_container_width=True,
                    ):
                        # Parse timestamps and check chronological order.
                        timestamp_column = next(
                            (
                                column
                                for column in (
                                    "interval_start_utc",
                                    "timestamp_utc",
                                    "timestamp",
                                    "datetime",
                                    "time",
                                )
                                if column in frame.columns
                            ),
                            None,
                        )

                        if timestamp_column:
                            parsed_times = pd.to_datetime(
                                frame[timestamp_column],
                                errors="coerce",
                                utc=True,
                            )

                            if parsed_times.isna().any():
                                st.error(
                                    f"Invalid timestamps found in "
                                    f"`{timestamp_column}`."
                                )
                                st.stop()

                            if not parsed_times.is_monotonic_increasing:
                                st.error(
                                    "Timestamps are not in chronological order. "
                                    "Sort the dataset before scoring."
                                )
                                st.stop()

                            if parsed_times.duplicated().any():
                                st.error(
                                    "Duplicate timestamps were found. "
                                    "Resolve duplicate intervals before scoring."
                                )
                                st.stop()

                        with st.spinner("Scoring traffic intervals..."):
                            scored = prepare_scored_results(
                                frame=frame,
                                value_column=target_column,
                                threshold=threshold,
                                initial_model=artifact["model"],
                            )

                        metadata = build_report_metadata(
                            source=uploaded_file.name,
                            threshold=threshold,
                            artifact=artifact,
                            scenario=uploaded_file.name,
                        )

                        st.session_state["scored_results"] = scored
                        st.session_state["scored_target"] = target_column
                        st.session_state["scored_metadata"] = metadata
                        st.success("Analysis completed.")

                if "scored_results" in st.session_state:
                    render_results(
                        st.session_state["scored_results"],
                        st.session_state["scored_target"],
                        st.session_state["scored_metadata"],
                    )

        except Exception as exc:
            st.error(f"Could not process the uploaded dataset: {exc}")

    else:
        st.info(
            "Upload a CSV containing one row per fixed interval and a "
            "numeric packet_count column. Missing values and invalid "
            "timestamps should be resolved before scoring."
        )


# ============================================================
# Live capture
# ============================================================

elif page == "Live capture":
    st.subheader("Live traffic capture")

    st.warning(
        "Capture runs on the machine hosting this application, not in "
        "the viewer's browser. Only monitor networks and interfaces "
        "you are authorized to observe."
    )

    try:
        interfaces = list_interfaces()
    except Exception as exc:
        interfaces = []
        st.error(f"Could not list network interfaces: {exc}")

    duration = st.number_input(
        "Capture duration (seconds)",
        min_value=10,
        max_value=600,
        value=60,
        step=10,
    )

    if interfaces:
        interface_options = ["Default interface"] + interfaces
        selected_interface = st.selectbox(
            "Network interface",
            interface_options,
        )
        interface = (
            None
            if selected_interface == "Default interface"
            else selected_interface
        )
    else:
        interface = None
        st.text_input(
            "Network interface",
            value="Default interface",
            disabled=True,
        )

    st.caption(
        "Aggregation is fixed at 60 seconds to match the current model. "
        "Training a compatible model is required before using other intervals."
    )

    if st.button(
        "Start authorized capture",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                f"Capturing traffic for {int(duration)} seconds..."
            ):
                traffic = capture_traffic(
                    duration_seconds=int(duration),
                    interface=interface,
                    interval_seconds=MODEL_INTERVAL_SECONDS,
                )

            if traffic.empty:
                st.warning(
                    "No packets were captured. Check interface selection "
                    "and packet-capture permissions."
                )
            else:
                st.success(
                    f"Capture completed. Collected "
                    f"{int(traffic['packet_count'].sum()):,} packets."
                )

                st.markdown("#### Captured traffic")
                st.dataframe(
                    traffic,
                    use_container_width=True,
                    hide_index=True,
                )

                scored = prepare_scored_results(
                    frame=traffic,
                    value_column=MODEL_TARGET,
                    threshold=threshold,
                    initial_model=artifact["model"],
                )

                metadata = build_report_metadata(
                    source="Live capture",
                    threshold=threshold,
                    artifact=artifact,
                    interface=interface or "Default interface",
                )

                st.session_state["scored_results"] = scored
                st.session_state["scored_target"] = MODEL_TARGET
                st.session_state["scored_metadata"] = metadata

                render_results(
                    scored,
                    MODEL_TARGET,
                    metadata,
                )

                st.download_button(
                    "Download raw capture aggregation",
                    data=traffic.to_csv(index=False).encode("utf-8"),
                    file_name="network_monitor_capture.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except PermissionError as exc:
            st.error(
                f"Permission denied: {exc} Check operating-system "
                "capture permissions and the selected interface."
            )
        except Exception as exc:
            st.error(f"Capture failed: {exc}")


# ============================================================
# Model information
# ============================================================

elif page == "Model information":
    st.subheader("Model and calibration")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Model artifact")

        details = {
            "Model": f"ARIMA{tuple(artifact['order'])}",
            "Target feature": artifact.get("target", MODEL_TARGET),
            "Scenario": artifact.get("scenario", "Not recorded"),
            "Training observations": artifact.get(
                "training_observations", "Unknown"
            ),
            "Calibration observations": artifact.get(
                "calibration_observations", "Unknown"
            ),
            "Calibration method": artifact.get(
                "calibration_method", "Not recorded"
            ),
        }

        st.dataframe(
            pd.DataFrame(
                {
                    "Property": list(details.keys()),
                    "Value": list(details.values()),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("#### Calibrated thresholds")

        st.metric(
            "95th-percentile threshold",
            f"{float(artifact['threshold_95']):.2f}",
        )

        st.metric(
            "99th-percentile threshold",
            f"{float(artifact['threshold_99']):.2f}",
        )

        st.write(
            "The detector compares absolute one-step forecast residuals "
            "with the selected threshold. Both high and low deviations "
            "can trigger alerts."
        )

    st.divider()
    st.markdown("#### Scoring methodology")

    st.write(
        "The model predicts the next traffic observation, compares the "
        "prediction with the observed value, and then appends the observed "
        "value to the model state. Model parameters remain fixed during "
        "scoring."
    )

    st.markdown("#### Interpretation and limitations")

    st.warning(
        "An anomaly is not a confirmed attack. The current model has not "
        "been established as a general-purpose C2 detector."
    )

    st.markdown(
        """
        - The model was trained on a specific traffic feature and interval.
        - Thresholds were calibrated on a limited dataset segment.
        - Other networks may have different traffic distributions.
        - MAE and RMSE measure forecasting error, not security detection accuracy.
        - Use separate training, calibration, and test data when evaluating improvements.
        - Validate alert behavior against suitable benign and labeled traffic.
        """
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Network Monitor | Experimental network traffic analytics | "
    "Use only on networks you are authorized to monitor."
)
