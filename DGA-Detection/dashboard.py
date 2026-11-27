
"""Plotly visualisations for the Streamlit DGA detection dashboard."""

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLASSIFICATION_COLORS = {
    "Likely benign": "#6B9B8A",
    "Potential DGA": "#C58B75",
}

EXPECTED_CLASSIFICATIONS = [
    "Likely benign",
    "Potential DGA",
]

DEFAULT_TOP_LIMIT = 15


# DOMAIN SENTINEL — MUTED SLATE + TEAL THEME

CHART_BACKGROUND = "#1E293B"
PAPER_BACKGROUND = "#0F172A"

TEXT_COLOR = "#E2E8F0"
MUTED_TEXT_COLOR = "#94A3B8"

GRID_COLOR = "#334155"
BORDER_COLOR = "#475569"

ACCENT_COLOR = "#8B9BB4"


# ============================================================
# 2. SHARED CHART STYLING
# ============================================================

def apply_chart_theme(fig, height=350):
    """Apply consistent DOMAIN SENTINEL styling to a Plotly figure."""

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,

        font=dict(
            family="Inter, Segoe UI, sans-serif",
            size=12,
            color=TEXT_COLOR,
        ),

        margin=dict(
            l=15,
            r=15,
            t=30,
            b=15,
        ),

        height=height,

        legend=dict(
            title_text="",
            font=dict(
                size=11,
                color=MUTED_TEXT_COLOR,
            ),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),

        hoverlabel=dict(
            bgcolor="#172033",
            bordercolor=BORDER_COLOR,
            font=dict(
                family="Inter, Segoe UI, sans-serif",
                size=12,
                color=TEXT_COLOR,
            ),
        ),
    )

    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        linecolor=BORDER_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(
            color=MUTED_TEXT_COLOR,
            size=10,
        ),
        title_font=dict(
            color=TEXT_COLOR,
            size=12,
        ),
    )

    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        linecolor=BORDER_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(
            color=MUTED_TEXT_COLOR,
            size=10,
        ),
        title_font=dict(
            color=TEXT_COLOR,
            size=12,
        ),
    )

    return fig


# ============================================================
# 3. VALIDATION
# ============================================================

def validate_results(results):
    """
    Validate the results dataframe before visualisation.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "Expected results to be a pandas DataFrame."
        )

    required_columns = {
        "domain",
        "dga_probability",
        "prediction",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Results dataframe is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if results.empty:
        return

    probabilities = pd.to_numeric(
        results["dga_probability"],
        errors="coerce",
    )

    if probabilities.isna().any():
        raise ValueError(
            "DGA probabilities contain missing or non-numeric values."
        )

    if not probabilities.map(math.isfinite).all():
        raise ValueError(
            "DGA probabilities contain non-finite values."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "DGA probabilities must be between 0 and 1."
        )

    if results["prediction"].isna().any():
        raise ValueError(
            "Results contain missing classification labels."
        )

    unexpected_labels = (
        set(results["prediction"].unique())
        - set(EXPECTED_CLASSIFICATIONS)
    )

    if unexpected_labels:
        raise ValueError(
            "Results contain unexpected classification labels: "
            f"{sorted(unexpected_labels)}"
        )


# ============================================================
# 4. EMPTY FIGURE
# ============================================================

def empty_figure(message="No data available"):
    """Return a consistently styled empty-state chart."""

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=14,
            color=MUTED_TEXT_COLOR,
        ),
    )

    fig.update_layout(
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        height=350,

        margin=dict(
            l=15,
            r=15,
            t=20,
            b=15,
        ),

        xaxis=dict(visible=False),
        yaxis=dict(visible=False),

        font=dict(
            family="Inter, Segoe UI, sans-serif",
            color=TEXT_COLOR,
        ),
    )

    return fig


# ============================================================
# 5. PREDICTION DISTRIBUTION
# ============================================================

def prediction_distribution(results):
    """
    Display the distribution of predicted benign and DGA domains.

    Uses the complete classified dataset.
    """

    validate_results(results)

    if results.empty:
        return empty_figure(
            "No classification results available"
        )

    counts = (
        results["prediction"]
        .value_counts()
        .reindex(
            EXPECTED_CLASSIFICATIONS,
            fill_value=0,
        )
        .rename_axis("Classification")
        .reset_index(name="Domains")
    )

    total_domains = int(counts["Domains"].sum())

    fig = px.pie(
        counts,
        names="Classification",
        values="Domains",
        hole=0.68,
        color="Classification",
        color_discrete_map=CLASSIFICATION_COLORS,
        category_orders={
            "Classification": EXPECTED_CLASSIFICATIONS
        },
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        textfont=dict(
            size=12,
            color="#FFFFFF",
        ),

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Domains: %{value:,}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        ),

        marker=dict(
            line=dict(
                color=CHART_BACKGROUND,
                width=3,
            )
        ),

        sort=False,
    )

    fig.update_layout(
        showlegend=True,

        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
        ),

        margin=dict(
            l=10,
            r=100,
            t=25,
            b=10,
        ),

        height=350,

        annotations=[
            dict(
                text=(
                    f"<b>{total_domains:,}</b>"
                    "<br><span style='font-size:11px'>Domains</span>"
                ),
                x=0.5,
                y=0.5,
                font=dict(
                    size=19,
                    color=TEXT_COLOR,
                ),
                showarrow=False,
            )
        ],
    )

    return fig


# ============================================================
# 6. PROBABILITY DISTRIBUTION
# ============================================================

def probability_distribution(results, threshold=0.50):
    """
    Display the distribution of predicted DGA probabilities,
    separated by predicted classification.

    A vertical line indicates the decision threshold.

    Uses all classified domains.
    """

    validate_results(results)

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
    ):
        raise ValueError(
            "Threshold must be numeric."
        )

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if results.empty:
        return empty_figure(
            "No probability data available"
        )

    chart_data = results[
        ["dga_probability", "prediction"]
    ].copy()

    chart_data["dga_probability"] = pd.to_numeric(
        chart_data["dga_probability"],
        errors="raise",
    )

    fig = px.histogram(
        chart_data,
        x="dga_probability",
        color="prediction",
        nbins=30,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map=CLASSIFICATION_COLORS,
        category_orders={
            "prediction": EXPECTED_CLASSIFICATIONS
        },
        labels={
            "dga_probability": "Predicted DGA probability",
            "count": "Number of domains",
            "prediction": "Classification",
        },
    )

    fig.update_traces(
        marker_line_width=0,
        hovertemplate=(
            "Probability: %{x:.1%}<br>"
            "Number of domains: %{y:,}"
            "<extra></extra>"
        ),
    )

    apply_chart_theme(fig, height=350)

    fig.update_layout(
        barmode="overlay",
        bargap=0.06,

        legend=dict(
            orientation="h",
            x=0,
            y=1.13,
            xanchor="left",
            yanchor="bottom",
        ),

        xaxis_title="Predicted DGA probability",
        yaxis_title="Number of domains",
    )

    fig.update_xaxes(
        range=[0, 1],
        tickformat=".0%",
        dtick=0.1,
    )

    fig.update_yaxes(
        rangemode="tozero",
        separatethousands=True,
    )

    # Decision threshold reference line

    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color=ACCENT_COLOR,
        annotation_text=f"Threshold: {threshold:.0%}",
        annotation_position="top",
        annotation_font=dict(
            color=ACCENT_COLOR,
            size=10,
        ),
    )

    return fig


# ============================================================
# 7. TOP SUSPICIOUS DOMAINS
# ============================================================

def top_suspicious_domains(
    results,
    limit=DEFAULT_TOP_LIMIT,
):
    """
    Display the highest-scoring domains.

    Long domain names are shortened on the chart axis.
    The complete domain remains available in the hover tooltip.
    """

    validate_results(results)

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):
        raise ValueError(
            "The chart limit must be a positive integer."
        )

    if results.empty:
        return empty_figure(
            "No domains available for ranking"
        )

    chart_data = results[
        ["domain", "dga_probability"]
    ].copy()

    chart_data["dga_probability"] = pd.to_numeric(
        chart_data["dga_probability"],
        errors="raise",
    )

    chart_data["domain"] = (
        chart_data["domain"].astype(str)
    )

    # Stable sorting for consistent ranking.

    top = (
        chart_data
        .sort_values(
            "dga_probability",
            ascending=False,
            kind="mergesort",
        )
        .head(limit)
        .copy()
    )

    top["display_domain"] = top["domain"].apply(
        lambda domain: (
            domain
            if len(domain) <= 38
            else domain[:17] + "..." + domain[-17:]
        )
    )

    # Highest probability appears at the top.

    top = top.sort_values(
        "dga_probability",
        ascending=True,
        kind="mergesort",
    )

    fig = px.bar(
        top,
        x="dga_probability",
        y="display_domain",
        orientation="h",
        color="dga_probability",
        custom_data=["domain"],

        color_continuous_scale=[
            "#6B9B8A",
            "#8B9BB4",
            "#C58B75",
        ],

        range_color=[0, 1],

        labels={
            "dga_probability": "DGA probability",
            "display_domain": "Domain",
        },
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "DGA probability: %{x:.2%}"
            "<extra></extra>"
        ),
        marker_line_width=0,
    )

    apply_chart_theme(
        fig,
        height=max(350, 32 * len(top)),
    )

    fig.update_layout(
        coloraxis_showscale=False,
        showlegend=False,
        xaxis_title="Predicted DGA probability",
        yaxis_title="",

        margin=dict(
            l=15,
            r=20,
            t=25,
            b=15,
        ),
    )

    fig.update_xaxes(
        range=[0, 1],
        tickformat=".0%",
        dtick=0.1,
    )

    fig.update_yaxes(
        automargin=True,
        tickfont=dict(
            family="monospace",
            size=10,
            color=MUTED_TEXT_COLOR,
        ),
    )

    return fig
