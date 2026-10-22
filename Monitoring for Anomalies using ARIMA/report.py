
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPORT_TITLE = "Network Monitoring Report"
REPORT_VERSION = "1.0"
DEFAULT_TIME_COLUMN = "interval_start_utc"
DEFAULT_TARGET_COLUMN = "packet_count"

ALERT_COLUMNS = [
    "interval_start_utc",
    "timestamp",
    "timestamp_utc",
    "time",
    "model_prediction",
    "arima_prediction",
    "residual",
    "absolute_error",
    "anomaly_alert",
    "alert_direction",
]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Convert a value to a finite float, or None if unavailable."""
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _format_number(value: Any, decimals: int = 2) -> str:
    """Format numeric values consistently for human-readable reports."""
    number = _safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:,.{decimals}f}"


def _format_timestamp(value: Any) -> str:
    """Format a timestamp in UTC where parsing is possible."""
    if value is None:
        return "N/A"

    try:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return str(value)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return str(value)


def _escape(value: Any) -> str:
    """Escape values before placing them in HTML."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and not np.isfinite(value):
        return "N/A"
    return html.escape(str(value))


def _json_safe(value: Any) -> Any:
    """Convert common Python and NumPy values into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return str(value)


def _first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first candidate column present in a DataFrame."""
    return next((column for column in candidates if column in frame.columns), None)


def _get_time_series(
    frame: pd.DataFrame,
) -> tuple[pd.Series | None, str | None]:
    """Find and parse a timestamp column, if available."""
    column = _first_existing_column(
        frame,
        [
            "interval_start_utc",
            "timestamp_utc",
            "timestamp",
            "datetime",
            "time",
        ],
    )

    if column is None:
        return None, None

    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if parsed.notna().all():
        return parsed, column

    return None, column


# ---------------------------------------------------------------------------
# Data validation and report analysis
# ---------------------------------------------------------------------------

def validate_report_data(
    results: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> None:
    """
    Validate the minimum information required to generate a report.

    Expected input is the scored DataFrame produced by the monitoring app.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")

    if results.empty:
        raise ValueError("Cannot generate a report from an empty dataset.")

    if target_column not in results.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing from the results."
        )

    if "anomaly_alert" not in results.columns:
        raise ValueError(
            "The scored results must contain an 'anomaly_alert' column."
        )

    target = pd.to_numeric(results[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing or "
            "non-numeric values."
        )

    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(
            f"Target column '{target_column}' contains infinite values."
        )


def calculate_summary(
    results: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate the report summary.

    Forecast metrics are calculated from the supplied scored observations.
    They describe forecasting errors on this dataset, not attack-detection
    accuracy.
    """
    validate_report_data(results, target_column)
    metadata = dict(metadata or {})

    observed = pd.to_numeric(results[target_column], errors="raise").astype(float)
    alerts = results["anomaly_alert"].fillna(False).astype(bool)
    alert_frame = results.loc[alerts].copy()

    prediction_column = _first_existing_column(
        results,
        ["model_prediction", "arima_prediction"],
    )

    residual_column = "residual" if "residual" in results.columns else None

    if residual_column:
        residuals = pd.to_numeric(
            results[residual_column], errors="coerce"
        )
    elif prediction_column:
        predictions = pd.to_numeric(
            results[prediction_column], errors="coerce"
        )
        residuals = observed - predictions
    else:
        residuals = pd.Series(np.nan, index=results.index, dtype=float)

    valid_residuals = residuals.dropna()
    valid_predictions = (
        pd.to_numeric(results[prediction_column], errors="coerce").dropna()
        if prediction_column
        else pd.Series(dtype=float)
    )

    alert_count = int(alerts.sum())
    total_intervals = len(results)

    time_values, time_column = _get_time_series(results)

    if time_values is not None:
        start_time = time_values.min()
        end_time = time_values.max()
        elapsed_seconds = max(
            0.0,
            (end_time - start_time).total_seconds(),
        )
        start_display = _format_timestamp(start_time)
        end_display = _format_timestamp(end_time)
    else:
        start_display = str(metadata.get("start_time", "Not available"))
        end_display = str(metadata.get("end_time", "Not available"))
        elapsed_seconds = _safe_float(metadata.get("duration_seconds"))

    interval_seconds = _safe_float(metadata.get("interval_seconds"))
    if interval_seconds is None:
        interval_seconds = _safe_float(metadata.get("aggregation_interval_seconds"))

    total_bytes = None
    if "total_bytes" in results.columns:
        byte_values = pd.to_numeric(results["total_bytes"], errors="coerce")
        if byte_values.notna().all():
            total_bytes = float(byte_values.sum())

    # If a total-byte field is absent, do not infer it from packet counts.
    summary = {
        "report_title": metadata.get("report_title", REPORT_TITLE),
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "source": metadata.get("source", "Not specified"),
        "interface": metadata.get("interface", "Not specified"),
        "scenario": metadata.get("scenario", "Not specified"),
        "target_column": target_column,
        "model_name": metadata.get("model_name", "Not specified"),
        "model_version": metadata.get("model_version", "Not recorded"),
        "model_order": metadata.get("model_order", "Not recorded"),
        "threshold": metadata.get("threshold", "Not recorded"),
        "threshold_method": metadata.get(
            "threshold_method", "Not recorded"
        ),
        "calibration_dataset": metadata.get(
            "calibration_dataset", "Not recorded"
        ),
        "interval_seconds": interval_seconds,
        "time_column": time_column,
        "start_time": start_display,
        "end_time": end_display,
        "duration_seconds": elapsed_seconds,
        "total_intervals": int(total_intervals),
        "alert_count": alert_count,
        "alert_rate_percent": alert_count / total_intervals * 100,
        "normal_interval_count": int(total_intervals - alert_count),
        "mean_observed": float(observed.mean()),
        "median_observed": float(observed.median()),
        "std_observed": float(observed.std(ddof=0)),
        "minimum_observed": float(observed.min()),
        "maximum_observed": float(observed.max()),
        "total_observed": float(observed.sum()),
        "total_bytes": total_bytes,
        "prediction_column": prediction_column,
        "mae": (
            float(np.mean(np.abs(valid_residuals)))
            if not valid_residuals.empty else None
        ),
        "rmse": (
            float(np.sqrt(np.mean(np.square(valid_residuals))))
            if not valid_residuals.empty else None
        ),
        "mean_residual": (
            float(valid_residuals.mean())
            if not valid_residuals.empty else None
        ),
        "forecast_coverage_percent": (
            len(valid_predictions) / total_intervals * 100
            if prediction_column else 0.0
        ),
        "positive_alerts": 0,
        "negative_alerts": 0,
        "unknown_direction_alerts": 0,
        "alert_records": int(len(alert_frame)),
        "metadata": _json_safe(metadata),
    }

    if "alert_direction" in alert_frame.columns:
        directions = alert_frame["alert_direction"].astype(str)
        summary["positive_alerts"] = int(
            directions.eq("higher-than-expected").sum()
        )
        summary["negative_alerts"] = int(
            directions.eq("lower-than-expected").sum()
        )
        summary["unknown_direction_alerts"] = int(
            alert_count
            - summary["positive_alerts"]
            - summary["negative_alerts"]
        )
    else:
        summary["unknown_direction_alerts"] = alert_count

    return summary


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _chart_to_base64(fig: plt.Figure) -> str:
    """Render a Matplotlib figure to a base64 PNG for standalone HTML."""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _make_traffic_chart(
    results: pd.DataFrame,
    target_column: str,
) -> str | None:
    """Create an observed-versus-forecast chart."""
    prediction_column = _first_existing_column(
        results,
        ["model_prediction", "arima_prediction"],
    )

    if prediction_column is None:
        return None

    observed = pd.to_numeric(results[target_column], errors="coerce")
    predicted = pd.to_numeric(results[prediction_column], errors="coerce")
    valid = observed.notna() & predicted.notna()

    if not valid.any():
        return None

    time_values, _ = _get_time_series(results)
    if time_values is not None:
        x_values = time_values[valid]
        x_label = "Time (UTC)"
    else:
        x_values = np.arange(len(results))[valid.to_numpy()]
        x_label = "Interval index"

    fig, ax = plt.subplots(figsize=(11, 4.1))
    ax.plot(
        x_values,
        observed.loc[valid].to_numpy(),
        label="Observed traffic",
        linewidth=1.7,
    )
    ax.plot(
        x_values,
        predicted.loc[valid].to_numpy(),
        label="Forecast",
        linewidth=1.5,
        linestyle="--",
    )

    if "anomaly_alert" in results.columns:
        flagged = results["anomaly_alert"].fillna(False).astype(bool) & valid
        if flagged.any():
            flagged_x = (
                time_values[flagged]
                if time_values is not None
                else np.flatnonzero(flagged.to_numpy())
            )
            ax.scatter(
                flagged_x,
                observed.loc[flagged].to_numpy(),
                label="Flagged interval",
                s=28,
                zorder=3,
            )

    ax.set_title("Observed Traffic vs. Forecast", loc="left", fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(target_column.replace("_", " ").title())
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    return _chart_to_base64(fig)


def _make_alert_chart(summary: Mapping[str, Any]) -> str | None:
    """Create an alert-direction chart when direction information exists."""
    higher = int(summary.get("positive_alerts", 0))
    lower = int(summary.get("negative_alerts", 0))
    unknown = int(summary.get("unknown_direction_alerts", 0))

    if higher + lower + unknown == 0:
        return None

    labels = ["Higher than expected", "Lower than expected"]
    values = [higher, lower]

    if unknown:
        labels.append("Unclassified direction")
        values.append(unknown)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(labels, values)
    ax.set_title("Alert Direction", loc="left", fontsize=12)
    ax.set_xlabel("Flagged intervals")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value}",
            va="center",
        )

    fig.tight_layout()
    return _chart_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{_escape(label)}</div>
        <div class="metric-value">{_escape(value)}</div>
        <div class="metric-note">{_escape(note)}</div>
    </div>
    """


def _html_table(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int = 500,
) -> str:
    """Render a bounded HTML table with escaped cell values."""
    if frame.empty:
        return '<p class="muted">No records to display.</p>'

    selected_columns = [
        column for column in (columns or list(frame.columns))
        if column in frame.columns
    ]
    display = frame[selected_columns].head(max_rows).copy()

    # Escape every cell explicitly. Do not render user-supplied HTML.
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: _escape(value)
        )

    table_html = display.to_html(
        index=False,
        escape=False,
        border=0,
        classes="data-table",
    )

    if len(frame) > max_rows:
        table_html += (
            f'<p class="muted">Showing {max_rows:,} of {len(frame):,} records. '
            "Download the complete CSV for all rows.</p>"
        )

    return table_html


def _format_duration(seconds: Any) -> str:
    number = _safe_float(seconds)
    if number is None:
        return "Not available"

    total_seconds = int(max(0, number))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_left = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes} min {seconds_left} sec"
    if minutes:
        return f"{minutes} min {seconds_left} sec"
    return f"{seconds_left} sec"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_html_report(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """
    Generate a detailed, standalone HTML monitoring report.

    Args:
        results:
            Scored DataFrame containing the observed target and anomaly_alert.
        output_path:
            Optional path where the HTML report will be written.
        target_column:
            Name of the observed traffic feature.
        metadata:
            Optional information about the capture, dataset, model and threshold.

    Returns:
        Complete HTML document as a string.
    """
    summary = calculate_summary(
        results=results,
        target_column=target_column,
        metadata=metadata,
    )

    traffic_chart = _make_traffic_chart(results, target_column)
    alert_chart = _make_alert_chart(summary)

    alert_frame = results.loc[
        results["anomaly_alert"].fillna(False).astype(bool)
    ].copy()

    alert_display_columns = [
        column for column in ALERT_COLUMNS if column in alert_frame.columns
    ]

    if not alert_display_columns:
        alert_display_columns = list(alert_frame.columns)

    # Build key metrics.
    metrics_html = "".join(
        [
            _metric_card(
                "Intervals analyzed",
                f"{summary['total_intervals']:,}",
                "Rows scored by the detector",
            ),
            _metric_card(
                "Anomaly alerts",
                f"{summary['alert_count']:,}",
                "Intervals exceeding the selected threshold",
            ),
            _metric_card(
                "Alert rate",
                f"{summary['alert_rate_percent']:.2f}%",
                "Share of analyzed intervals flagged",
            ),
            _metric_card(
                "Mean absolute error",
                _format_number(summary["mae"]),
                "Forecasting metric, not detection accuracy",
            ),
            _metric_card(
                "Peak observed value",
                _format_number(summary["maximum_observed"]),
                target_column,
            ),
        ]
    )

    traffic_chart_html = (
        f'<img class="chart" alt="Observed traffic compared with forecast" '
        f'src="data:image/png;base64,{traffic_chart}">'
        if traffic_chart
        else '<p class="muted">Forecast data was unavailable; no traffic chart was generated.</p>'
    )

    alert_chart_html = (
        f'<img class="chart chart-narrow" alt="Alert direction distribution" '
        f'src="data:image/png;base64,{alert_chart}">'
        if alert_chart
        else '<p class="muted">No anomaly alerts were recorded.</p>'
    )

    threshold_value = summary.get("threshold", "Not recorded")
    threshold_method = summary.get("threshold_method", "Not recorded")

    if summary["alert_count"] == 0:
        finding_text = (
            "No intervals exceeded the selected threshold in this dataset. "
            "This does not establish that the network was free from malicious activity."
        )
    else:
        finding_text = (
            f"{summary['alert_count']:,} interval(s) exceeded the selected threshold. "
            "These are statistical deviations requiring contextual review; they are "
            "not confirmed security incidents."
        )

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(summary['report_title'])}</title>
<style>
    :root {{
        color-scheme: light;
        --ink: #172033;
        --muted: #627087;
        --line: #dce2eb;
        --surface: #f4f6fa;
        --panel: #ffffff;
        --accent: #254a77;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0;
        background: var(--surface);
        color: var(--ink);
        font-family: Inter, "Segoe UI", Arial, sans-serif;
        font-size: 14px;
        line-height: 1.55;
    }}
    .report {{
        max-width: 1180px;
        margin: 28px auto;
        padding: 0 24px 36px;
    }}
    .cover {{
        background: #172b46;
        color: white;
        border-radius: 14px;
        padding: 34px 36px;
        margin-bottom: 20px;
    }}
    .eyebrow {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        opacity: .78;
    }}
    h1 {{
        margin: 10px 0;
        font-size: 31px;
        line-height: 1.2;
    }}
    .cover p {{
        max-width: 760px;
        margin-bottom: 0;
        color: #d8e2f0;
    }}
    .cover-meta {{
        margin-top: 23px;
        padding-top: 15px;
        border-top: 1px solid rgba(255,255,255,.22);
        font-size: 12px;
        color: #d8e2f0;
    }}
    .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 23px 25px;
        margin-bottom: 18px;
        page-break-inside: avoid;
    }}
    h2 {{
        font-size: 18px;
        margin: 0 0 15px;
        letter-spacing: -.01em;
    }}
    h3 {{
        font-size: 14px;
        margin: 19px 0 8px;
    }}
    p {{ margin: 7px 0 12px; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .metrics {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 18px;
    }}
    .metric-card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 15px;
        min-width: 0;
    }}
    .metric-label {{
        color: var(--muted);
        font-size: 11px;
        font-weight: 650;
    }}
    .metric-value {{
        font-size: 23px;
        font-weight: 750;
        margin: 8px 0 5px;
        overflow-wrap: anywhere;
    }}
    .metric-note {{ color: var(--muted); font-size: 10px; }}
    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
    }}
    .facts {{
        width: 100%;
        border-collapse: collapse;
    }}
    .facts td {{
        padding: 9px 0;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
    }}
    .facts td:first-child {{
        width: 44%;
        color: var(--muted);
        padding-right: 12px;
    }}
    .facts tr:last-child td {{ border-bottom: 0; }}
    .chart {{
        display: block;
        max-width: 100%;
        height: auto;
        margin: 8px auto 0;
    }}
    .chart-narrow {{ max-width: 780px; }}
    .data-table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 11px;
        overflow-wrap: anywhere;
    }}
    .data-table th {{
        text-align: left;
        background: #edf1f7;
        padding: 9px 8px;
        border-bottom: 1px solid var(--line);
    }}
    .data-table td {{
        padding: 8px;
        border-bottom: 1px solid var(--line);
    }}
    .data-table tr:nth-child(even) td {{ background: #fafbfd; }}
    .notice {{
        border-left: 4px solid var(--accent);
        background: #f0f4f9;
        padding: 13px 16px;
        border-radius: 0 8px 8px 0;
    }}
    .notice.warning {{
        border-left-color: #a66b12;
        background: #fff7e8;
    }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 5px 0; }}
    .footer {{
        color: var(--muted);
        font-size: 11px;
        text-align: center;
        padding: 18px 0 0;
    }}
    @media (max-width: 900px) {{
        .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .two-col {{ grid-template-columns: 1fr; }}
        .report {{ padding: 0 12px 24px; }}
        .cover {{ padding: 25px 22px; }}
        h1 {{ font-size: 25px; }}
    }}
    @media print {{
        @page {{ size: A4 landscape; margin: 12mm; }}
        body {{ background: white; font-size: 10px; }}
        .report {{ max-width: none; margin: 0; padding: 0; }}
        .cover {{ border-radius: 0; print-color-adjust: exact; }}
        .panel, .metric-card {{ box-shadow: none; }}
        .panel {{ padding: 14px; margin-bottom: 10px; }}
        .metrics {{ grid-template-columns: repeat(5, 1fr); }}
        .metric-value {{ font-size: 18px; }}
        .data-table {{ font-size: 9px; }}
        .no-print {{ display: none !important; }}
        a {{ color: inherit; text-decoration: none; }}
    }}
</style>
</head>
<body>
<main class="report">
    <header class="cover">
        <div class="eyebrow">Network Operations / Analytical Report</div>
        <h1>{_escape(summary['report_title'])}</h1>
        <p>
            A structured review of observed traffic, forecast deviations,
            anomaly alerts, and model context for the supplied monitoring session.
        </p>
        <div class="cover-meta">
            Report version {_escape(summary['report_version'])}
            &nbsp; | &nbsp; Generated {_escape(summary['generated_at'])}
            &nbsp; | &nbsp; Source {_escape(summary['source'])}
        </div>
    </header>

    <section class="metrics">
        {metrics_html}
    </section>

    <section class="panel">
        <h2>1. Executive Summary</h2>
        <div class="notice">
            {_escape(finding_text)}
        </div>
        <p>
            The analysis covers <strong>{summary['total_intervals']:,}</strong>
            interval(s), of which <strong>{summary['alert_count']:,}</strong>
            were flagged. The resulting alert rate is
            <strong>{summary['alert_rate_percent']:.2f}%</strong>.
            The selected feature is
            <code>{_escape(target_column)}</code>.
        </p>
        <p class="muted">
            This report describes statistical monitoring output. It does not
            independently confirm malicious activity, identify an attacker,
            or establish command-and-control communication.
        </p>
    </section>

    <section class="panel">
        <h2>2. Monitoring Scope and Data Context</h2>
        <div class="two-col">
            <div>
                <h3>Dataset and capture</h3>
                <table class="facts">
                    <tr><td>Data source</td><td>{_escape(summary['source'])}</td></tr>
                    <tr><td>Scenario / capture</td><td>{_escape(summary['scenario'])}</td></tr>
                    <tr><td>Interface</td><td>{_escape(summary['interface'])}</td></tr>
                    <tr><td>Feature analyzed</td><td>{_escape(target_column)}</td></tr>
                    <tr><td>Aggregation interval</td><td>{_escape(summary['interval_seconds'])} seconds</td></tr>
                </table>
            </div>
            <div>
                <h3>Time coverage</h3>
                <table class="facts">
                    <tr><td>First interval</td><td>{_escape(summary['start_time'])}</td></tr>
                    <tr><td>Last interval</td><td>{_escape(summary['end_time'])}</td></tr>
                    <tr><td>Elapsed span</td><td>{_escape(_format_duration(summary['duration_seconds']))}</td></tr>
                    <tr><td>Intervals analyzed</td><td>{summary['total_intervals']:,}</td></tr>
                    <tr><td>Normal intervals</td><td>{summary['normal_interval_count']:,}</td></tr>
                </table>
            </div>
        </div>
        <p class="muted">
            Time span is calculated from the first and last valid timestamps when
            available. It is not necessarily the same as the actual capture duration.
        </p>
    </section>

    <section class="panel">
        <h2>3. Traffic Statistics</h2>
        <div class="two-col">
            <div>
                <table class="facts">
                    <tr><td>Mean per interval</td><td>{_format_number(summary['mean_observed'])}</td></tr>
                    <tr><td>Median per interval</td><td>{_format_number(summary['median_observed'])}</td></tr>
                    <tr><td>Standard deviation</td><td>{_format_number(summary['std_observed'])}</td></tr>
                    <tr><td>Minimum observed</td><td>{_format_number(summary['minimum_observed'])}</td></tr>
                    <tr><td>Maximum observed</td><td>{_format_number(summary['maximum_observed'])}</td></tr>
                    <tr><td>Sum of observations</td><td>{_format_number(summary['total_observed'], 0)}</td></tr>
                    <tr><td>Total bytes, if supplied</td><td>{_format_number(summary['total_bytes'], 0)}</td></tr>
                </table>
            </div>
            <div>
                <p>
                    These statistics summarize the supplied feature over the
                    analyzed intervals. For packet counts, the sum represents
                    the total number of packets counted in the supplied rows.
                </p>
                <p class="muted">
                    Totals and averages are only meaningful when the input uses
                    consistent aggregation intervals and a consistent feature
                    definition. Empty intervals should be represented explicitly
                    when zero traffic is known to have occurred.
                </p>
            </div>
        </div>
    </section>

    <section class="panel">
        <h2>4. Observed Traffic and Forecast</h2>
        {traffic_chart_html}
        <p class="muted">
            Forecasts are one-step-ahead predictions from the supplied scoring
            results. Highlighted points indicate intervals flagged by the detector.
            The chart is descriptive and does not by itself establish the cause
            of a traffic deviation.
        </p>
    </section>

    <section class="panel">
        <h2>5. Anomaly Findings</h2>
        <div class="two-col">
            <div>
                <table class="facts">
                    <tr><td>Flagged intervals</td><td>{summary['alert_count']:,}</td></tr>
                    <tr><td>Alert rate</td><td>{summary['alert_rate_percent']:.2f}%</td></tr>
                    <tr><td>Higher-than-expected</td><td>{summary['positive_alerts']:,}</td></tr>
                    <tr><td>Lower-than-expected</td><td>{summary['negative_alerts']:,}</td></tr>
                    <tr><td>Other / unclassified direction</td><td>{summary['unknown_direction_alerts']:,}</td></tr>
                    <tr><td>Selected threshold</td><td>{_escape(threshold_value)}</td></tr>
                    <tr><td>Threshold method</td><td>{_escape(threshold_method)}</td></tr>
                </table>
            </div>
            <div>
                {alert_chart_html}
            </div>
        </div>
        <h3>Alert records</h3>
        {_html_table(alert_frame, alert_display_columns)}
        <p class="muted">
            An alert indicates that an observation crossed the configured
            statistical threshold. A high deviation can reflect a burst of
            legitimate activity; a low deviation can reflect a drop in activity,
            an idle period, or data-collection behavior. Review relevant context
            before treating an alert as a security incident.
        </p>
    </section>

    <section class="panel">
        <h2>6. Forecasting and Model Information</h2>
        <div class="two-col">
            <div>
                <table class="facts">
                    <tr><td>Model</td><td>{_escape(summary['model_name'])}</td></tr>
                    <tr><td>Model version</td><td>{_escape(summary['model_version'])}</td></tr>
                    <tr><td>Model order</td><td>{_escape(summary['model_order'])}</td></tr>
                    <tr><td>Target feature</td><td>{_escape(target_column)}</td></tr>
                    <tr><td>Calibration dataset</td><td>{_escape(summary['calibration_dataset'])}</td></tr>
                    <tr><td>Calibration method</td><td>{_escape(threshold_method)}</td></tr>
                </table>
            </div>
            <div>
                <table class="facts">
                    <tr><td>Mean absolute error (MAE)</td><td>{_format_number(summary['mae'])}</td></tr>
                    <tr><td>Root mean squared error (RMSE)</td><td>{_format_number(summary['rmse'])}</td></tr>
                    <tr><td>Mean residual</td><td>{_format_number(summary['mean_residual'])}</td></tr>
                    <tr><td>Forecast coverage</td><td>{summary['forecast_coverage_percent']:.2f}%</td></tr>
                </table>
                <p class="muted">
                    Forecast errors are calculated from the supplied predictions
                    and observations. They are not precision, recall, F1, or
                    false-positive rates for malicious-traffic detection.
                </p>
            </div>
        </div>
        <div class="notice warning">
            The model's performance on this dataset does not establish
            generalization to other networks, traffic features, or aggregation
            intervals. Detection quality requires suitable labeled evaluation
            data and a clearly defined anomaly or incident criterion.
        </div>
    </section>

    <section class="panel">
        <h2>7. Interpretation and Recommended Follow-up</h2>
        <h3>Interpretation</h3>
        <ul>
            <li>Review flagged intervals alongside surrounding traffic and known network activity.</li>
            <li>Distinguish unusually high traffic from unusually low traffic before investigating.</li>
            <li>Confirm that timestamps, feature definitions, and aggregation intervals are consistent.</li>
            <li>Do not label statistical deviations as confirmed attacks without corroborating evidence.</li>
        </ul>
        <h3>Recommended follow-up</h3>
        <ul>
            <li>Compare results with a simple forecasting baseline on the same held-out observations.</li>
            <li>Evaluate thresholds on separate calibration data rather than the model-fitting segment.</li>
            <li>Test the model on additional captures and benign traffic to measure alert burden.</li>
            <li>Record dataset provenance, model version, configuration, and evaluation results.</li>
        </ul>
    </section>

    <section class="panel">
        <h2>8. Methodology, Assumptions, and Limitations</h2>
        <ul>
            <li>The report summarizes the scored DataFrame provided by the application; it does not independently rerun the model.</li>
            <li>The alert rate is the percentage of supplied intervals marked as alerts.</li>
            <li>MAE and RMSE describe forecast residuals only when compatible prediction and observation values are available.</li>
            <li>Thresholds are reported from supplied metadata and are not assumed to be calibrated false-positive rates.</li>
            <li>Missing timestamps, irregular intervals, or concatenated captures can undermine interpretation and should be resolved before scoring.</li>
            <li>This report does not perform packet attribution, payload inspection, or confirmation of command-and-control activity.</li>
        </ul>
        <p class="muted">
            Report generated by Network Monitor, version
            {_escape(REPORT_VERSION)}. Generated at
            {_escape(summary['generated_at'])}.
        </p>
    </section>

    <footer class="footer">
        Network Monitor &middot; Traffic analytics report &middot;
        For authorized monitoring and analytical review
    </footer>
</main>
</body>
</html>
"""

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html_document, encoding="utf-8")

    return html_document


# ---------------------------------------------------------------------------
# CSV exports and report bundles
# ---------------------------------------------------------------------------

def export_scored_csv(
    results: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write all scored observations to CSV."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)
    return destination


def export_alerts_csv(
    results: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write only flagged intervals to CSV."""
    if "anomaly_alert" not in results.columns:
        raise ValueError("Results must contain an 'anomaly_alert' column.")

    alert_mask = results["anomaly_alert"].fillna(False).astype(bool)
    alerts = results.loc[alert_mask].copy()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(destination, index=False)
    return destination


def export_summary_json(
    results: pd.DataFrame,
    output_path: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export the calculated summary as JSON."""
    summary = calculate_summary(
        results=results,
        target_column=target_column,
        metadata=metadata,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def generate_report_bundle(
    results: pd.DataFrame,
    output_directory: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    metadata: Mapping[str, Any] | None = None,
    filename_prefix: str = "network_monitor_report",
) -> dict[str, Path]:
    """
    Write a complete report bundle:
        - Detailed HTML report
        - All scored observations CSV
        - Alerts-only CSV
        - Summary JSON

    Returns a mapping of artifact names to their filesystem paths.
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"{filename_prefix}.html"
    scored_path = output_dir / f"{filename_prefix}_scored.csv"
    alerts_path = output_dir / f"{filename_prefix}_alerts.csv"
    summary_path = output_dir / f"{filename_prefix}_summary.json"

    generate_html_report(
        results=results,
        output_path=html_path,
        target_column=target_column,
        metadata=metadata,
    )
    export_scored_csv(results, scored_path)
    export_alerts_csv(results, alerts_path)
    export_summary_json(
        results=results,
        output_path=summary_path,
        target_column=target_column,
        metadata=metadata,
    )

    return {
        "html_report": html_path,
        "scored_csv": scored_path,
        "alerts_csv": alerts_path,
        "summary_json": summary_path,
    }
