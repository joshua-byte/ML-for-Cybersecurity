
"""Result summaries and downloadable report generation."""

from datetime import datetime, timezone
from html import escape
import math

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

REPORT_TOP_N = 15

BENIGN_COLOR = "#20A58A"
DGA_COLOR = "#E69A42"

BACKGROUND_COLOR = "#101827"
CARD_COLOR = "#1A293B"
BORDER_COLOR = "#304258"
TEXT_COLOR = "#E8EEF5"
MUTED_COLOR = "#A8B8C9"


# ============================================================
# 2. SUMMARY METRICS
# ============================================================

def build_summary(
    results: pd.DataFrame,
    rejected: pd.DataFrame | None = None,
) -> dict:
    """
    Calculate summary statistics across the complete
    classified dataset, not just the top 15.
    """

    rejected_count = (
        0 if rejected is None else len(rejected)
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
        return {
            "total_processed": 0,
            "potential_dga": 0,
            "likely_benign": 0,
            "mean_dga_probability": 0.0,
            "median_dga_probability": 0.0,
            "max_dga_probability": 0.0,
            "min_dga_probability": 0.0,
            "rejected": rejected_count,
        }

    probabilities = pd.to_numeric(
        results["dga_probability"],
        errors="raise",
    )

    if probabilities.isna().any():
        raise ValueError(
            "Results contain missing DGA probabilities."
        )

    if not probabilities.map(math.isfinite).all():
        raise ValueError(
            "Results contain non-finite DGA probabilities."
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
        - {"Likely benign", "Potential DGA"}
    )

    if unexpected_labels:
        raise ValueError(
            "Results contain unexpected classification labels: "
            f"{sorted(unexpected_labels)}"
        )

    return {
        "total_processed": int(len(results)),

        "potential_dga": int(
            (results["prediction"] == "Potential DGA").sum()
        ),

        "likely_benign": int(
            (results["prediction"] == "Likely benign").sum()
        ),

        "mean_dga_probability": float(
            probabilities.mean()
        ),

        "median_dga_probability": float(
            probabilities.median()
        ),

        "max_dga_probability": float(
            probabilities.max()
        ),

        "min_dga_probability": float(
            probabilities.min()
        ),

        "rejected": rejected_count,
    }


# ============================================================
# 3. CSV EXPORT
# ============================================================

def results_csv_bytes(
    results: pd.DataFrame,
) -> bytes:
    """
    Export every classified domain.

    The CSV is intentionally not restricted to the top 15.
    """

    return results.to_csv(
        index=False,
    ).encode("utf-8-sig")


def rejected_csv_bytes(
    rejected: pd.DataFrame,
) -> bytes:
    """Export rejected input rows separately."""

    return rejected.to_csv(
        index=False,
    ).encode("utf-8-sig")


# ============================================================
# 4. HTML REPORT GENERATION
# ============================================================

def report_html(
    results: pd.DataFrame,
    model_name: str,
    threshold: float,
    rejected: pd.DataFrame | None = None,
    filename: str | None = None,
) -> bytes:
    """
    Generate a professionally styled HTML DGA detection report.

    Summary statistics:
        Calculated using all successfully classified domains.

    Domain results table:
        Restricted to the 15 highest DGA probabilities.

    Rejected rows:
        Counted in the report but exported separately as CSV.
    """

    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    try:
        threshold = float(threshold)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Threshold must be a valid number."
        ) from exc

    if (
        not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Generate report metadata
    # --------------------------------------------------------

    summary = build_summary(
        results,
        rejected,
    )

    generated = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    safe_model_name = escape(str(model_name))

    safe_filename = escape(
        str(filename) if filename else "Not provided"
    )

    # --------------------------------------------------------
    # Calculate classification percentages
    # --------------------------------------------------------

    total = summary["total_processed"]

    if total:

        dga_percentage = (
            summary["potential_dga"] / total
        ) * 100

        benign_percentage = (
            summary["likely_benign"] / total
        ) * 100

    else:

        dga_percentage = 0.0
        benign_percentage = 0.0

    # --------------------------------------------------------
    # Build top 15 domain-level results table
    # --------------------------------------------------------

    if not results.empty:

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

        sorted_results = results[
            ["domain", "dga_probability", "prediction"]
        ].copy()

        sorted_results["dga_probability"] = pd.to_numeric(
            sorted_results["dga_probability"],
            errors="raise",
        )

        probabilities = sorted_results["dga_probability"]

        if probabilities.isna().any():
            raise ValueError(
                "Results contain missing DGA probabilities."
            )

        if not probabilities.map(math.isfinite).all():
            raise ValueError(
                "Results contain non-finite DGA probabilities."
            )

        if not probabilities.between(0, 1).all():
            raise ValueError(
                "DGA probabilities must be between 0 and 1."
            )

        if sorted_results["prediction"].isna().any():
            raise ValueError(
                "Results contain missing classification labels."
            )

        # Sort the complete dataset and select only 15 rows.

        top_domains = (
            sorted_results
            .sort_values(
                "dga_probability",
                ascending=False,
                kind="mergesort",
            )
            .head(REPORT_TOP_N)
            .copy()
        )

        # Construct escaped HTML table rows.

        table_rows = []

        for rank, (_, row) in enumerate(
            top_domains.iterrows(),
            start=1,
        ):

            domain = escape(str(row["domain"]))

            probability = float(row["dga_probability"])

            classification = str(row["prediction"])

            if classification == "Potential DGA":

                badge_class = "dga-badge"

            else:

                badge_class = "benign-badge"

            table_rows.append(
                f"""
                <tr>
                    <td class="rank-cell">{rank}</td>

                    <td class="domain-cell">
                        {domain}
                    </td>

                    <td class="probability-cell">
                        <div class="probability-wrapper">

                            <span class="probability-value">
                                {probability:.2%}
                            </span>

                            <div class="probability-track">
                                <div
                                    class="probability-fill"
                                    style="width:{probability * 100:.2f}%;">
                                </div>
                            </div>

                        </div>
                    </td>

                    <td>
                        <span class="{badge_class}">
                            {escape(classification)}
                        </span>
                    </td>
                </tr>
                """
            )

        table = "\n".join(table_rows)

        displayed_count = len(top_domains)

    else:

        table = """
        <tr>
            <td colspan="4" class="empty-table">
                No valid domains were available for classification.
            </td>
        </tr>
        """

        displayed_count = 0

    # --------------------------------------------------------
    # Build HTML document
    # --------------------------------------------------------

    report_document = f"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>DOMAIN SENTINEL | Detection Report</title>

<style>

/* ==========================================================
   GLOBAL STYLING
   ========================================================== */

* {{
    box-sizing: border-box;
}}

:root {{
    --background: {BACKGROUND_COLOR};
    --surface: {CARD_COLOR};
    --border: {BORDER_COLOR};
    --text: {TEXT_COLOR};
    --muted: {MUTED_COLOR};
    --teal: {BENIGN_COLOR};
    --amber: {DGA_COLOR};
    --cyan: #35B8D4;
}}

body {{
    margin: 0;
    padding: 35px 24px;
    background: var(--background);
    color: var(--text);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    line-height: 1.6;
}}

.container {{
    max-width: 1150px;
    margin: 0 auto;
}}

/* ==========================================================
   HEADER
   ========================================================== */

.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    padding: 24px 26px;
    background: linear-gradient(
        120deg,
        #18283B,
        #142235
    );
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 25px;
}}

.brand-title {{
    font-size: 25px;
    font-weight: 750;
    letter-spacing: 1.2px;
    margin: 0;
}}

.brand-subtitle {{
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-top: 5px;
}}

.report-label {{
    display: inline-block;
    padding: 7px 12px;
    border-radius: 20px;
    background: #123B35;
    border: 1px solid #226B5B;
    color: #65D6B5;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    white-space: nowrap;
}}

.generated {{
    color: var(--muted);
    font-size: 12px;
    margin-top: 18px;
}}

/* ==========================================================
   METADATA
   ========================================================== */

.metadata {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 15px;
    margin-top: 22px;
}}

.metadata-item {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 15px 17px;
    min-width: 0;
}}

.metadata-label {{
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 7px;
}}

.metadata-value {{
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
    overflow-wrap: anywhere;
}}

/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section {{
    margin-top: 34px;
}}

.section-heading {{
    font-size: 18px;
    font-weight: 650;
    margin: 0 0 5px;
    color: var(--text);
}}

.section-description {{
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 17px;
}}

/* ==========================================================
   SUMMARY CARDS
   ========================================================== */

.cards {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 20px 0;
}}

.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 18px;
    min-width: 0;
}}

.card-title {{
    color: var(--muted);
    font-size: 11px;
    font-weight: 550;
    margin-bottom: 12px;
}}

.value {{
    color: var(--text);
    font-size: 25px;
    font-weight: 700;
    letter-spacing: -0.5px;
    overflow-wrap: anywhere;
}}

.card-accent {{
    height: 3px;
    width: 32px;
    border-radius: 5px;
    margin-top: 13px;
}}

/* ==========================================================
   CLASSIFICATION DISTRIBUTION
   ========================================================== */

.distribution-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 20px;
}}

.distribution {{
    display: flex;
    width: 100%;
    height: 14px;
    overflow: hidden;
    border-radius: 8px;
    margin: 17px 0;
    background: #26384D;
}}

.dga-bar {{
    background: var(--amber);
    width: {dga_percentage:.4f}%;
}}

.benign-bar {{
    background: var(--teal);
    width: {benign_percentage:.4f}%;
}}

.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    color: var(--muted);
    font-size: 12px;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
}}

.legend-dot {{
    height: 9px;
    width: 9px;
    border-radius: 3px;
    display: inline-block;
}}

.distribution-stat {{
    color: var(--text);
    font-weight: 650;
}}

/* ==========================================================
   TOP DOMAINS TABLE
   ========================================================== */

.table-wrapper {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
}}

table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
}}

th {{
    background: #22344A;
    color: #B7C7D8;
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    padding: 14px;
    text-align: left;
    white-space: nowrap;
}}

td {{
    border-top: 1px solid var(--border);
    padding: 13px 14px;
    color: var(--text);
    vertical-align: middle;
}}

tr:hover {{
    background: #203249;
}}

.rank-cell {{
    color: var(--muted);
    width: 55px;
}}

.domain-cell {{
    font-family: Consolas, "Courier New", monospace;
    overflow-wrap: anywhere;
    min-width: 280px;
}}

.probability-cell {{
    min-width: 160px;
}}

.probability-wrapper {{
    display: flex;
    flex-direction: column;
    gap: 6px;
}}

.probability-value {{
    color: var(--text);
    font-weight: 650;
}}

.probability-track {{
    height: 5px;
    width: 100%;
    border-radius: 5px;
    background: #33465B;
    overflow: hidden;
}}

.probability-fill {{
    height: 100%;
    background: var(--amber);
    border-radius: 5px;
}}

.dga-badge,
.benign-badge {{
    display: inline-block;
    padding: 5px 9px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 650;
    white-space: nowrap;
}}

.dga-badge {{
    background: #4B351F;
    color: #F0B66F;
    border: 1px solid #76522C;
}}

.benign-badge {{
    background: #153B35;
    color: #65D6B5;
    border: 1px solid #286B5A;
}}

.empty-table {{
    text-align: center;
    color: var(--muted);
    padding: 25px;
}}

/* ==========================================================
   INTERPRETATION AND LIMITATIONS
   ========================================================== */

.note {{
    margin-top: 28px;
    padding: 20px;
    background: #182638;
    border: 1px solid var(--border);
    border-left: 4px solid var(--cyan);
    border-radius: 9px;
}}

.note-heading {{
    font-weight: 700;
    margin-bottom: 10px;
}}

.note p {{
    color: var(--muted);
    font-size: 12px;
    margin: 8px 0;
}}

/* ==========================================================
   FOOTER
   ========================================================== */

.footer {{
    margin-top: 35px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    color: #8294A9;
    font-size: 11px;
    text-align: center;
    line-height: 1.9;
}}

/* ==========================================================
   RESPONSIVE LAYOUT
   ========================================================== */

@media (max-width: 800px) {{

    body {{
        padding: 18px 12px;
    }}

    .cards {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .metadata {{
        grid-template-columns: 1fr;
    }}

    .header {{
        align-items: flex-start;
        flex-direction: column;
    }}
}}

@media (max-width: 450px) {{

    .cards {{
        grid-template-columns: 1fr;
    }}

    .brand-title {{
        font-size: 21px;
    }}
}}

/* ==========================================================
   PRINT STYLING
   ========================================================== */

@media print {{

    body {{
        background: white;
        color: #17202A;
        padding: 10px;
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }}

    .container {{
        max-width: 100%;
    }}

    .card,
    .metadata-item,
    .distribution-card,
    .table-wrapper,
    .note,
    .header {{
        break-inside: avoid;
    }}

    .section {{
        break-inside: avoid;
    }}

    table {{
        font-size: 10px;
    }}

}}

</style>

</head>

<body>

<div class="container">

    <!-- HEADER -->

    <div class="header">

        <div>
            <div class="brand-title">
                ◈ DOMAIN SENTINEL
            </div>

            <div class="brand-subtitle">
                SUPERVISED LEXICAL DGA DETECTION
            </div>
        </div>

        <div class="report-label">
            ANALYSIS REPORT
        </div>

    </div>

    <div class="generated">
        Report generated: {generated}
    </div>


    <!-- REPORT METADATA -->

    <div class="metadata">

        <div class="metadata-item">

            <div class="metadata-label">
                Input dataset
            </div>

            <div class="metadata-value">
                {safe_filename}
            </div>

        </div>

        <div class="metadata-item">

            <div class="metadata-label">
                Classifier
            </div>

            <div class="metadata-value">
                {safe_model_name}
            </div>

        </div>

        <div class="metadata-item">

            <div class="metadata-label">
                Decision threshold
            </div>

            <div class="metadata-value">
                {threshold:.0%}
            </div>

        </div>

    </div>


    <!-- ANALYSIS SUMMARY -->

    <div class="section">

        <h2 class="section-heading">
            Detection overview
        </h2>

        <div class="section-description">
            Summary of supervised classification results
            across the uploaded dataset.
        </div>

        <div class="cards">

            <div class="card">

                <div class="card-title">
                    Domains analysed
                </div>

                <div class="value">
                    {summary['total_processed']:,}
                </div>

                <div class="card-accent"
                     style="background:#35B8D4;">
                </div>

            </div>


            <div class="card">

                <div class="card-title">
                    Potential DGA
                </div>

                <div class="value">
                    {summary['potential_dga']:,}
                </div>

                <div class="card-accent"
                     style="background:{DGA_COLOR};">
                </div>

            </div>


            <div class="card">

                <div class="card-title">
                    Likely benign
                </div>

                <div class="value">
                    {summary['likely_benign']:,}
                </div>

                <div class="card-accent"
                     style="background:{BENIGN_COLOR};">
                </div>

            </div>


            <div class="card">

                <div class="card-title">
                    Rejected inputs
                </div>

                <div class="value">
                    {summary['rejected']:,}
                </div>

                <div class="card-accent"
                     style="background:#8299E8;">
                </div>

            </div>

        </div>

    </div>


    <!-- CLASSIFICATION DISTRIBUTION -->

    <div class="section">

        <h2 class="section-heading">
            Classification distribution
        </h2>

        <div class="section-description">
            Proportion of successfully classified domains
            by predicted class.
        </div>

        <div class="distribution-card">

            <div class="distribution">

                <div class="dga-bar"></div>

                <div class="benign-bar"></div>

            </div>

            <div class="legend">

                <div class="legend-item">

                    <span class="legend-dot"
                          style="background:{DGA_COLOR};">
                    </span>

                    <span>
                        Potential DGA:
                        <span class="distribution-stat">
                            {dga_percentage:.2f}%
                        </span>
                    </span>

                </div>


                <div class="legend-item">

                    <span class="legend-dot"
                          style="background:{BENIGN_COLOR};">
                    </span>

                    <span>
                        Likely benign:
                        <span class="distribution-stat">
                            {benign_percentage:.2f}%
                        </span>
                    </span>

                </div>

            </div>

        </div>

    </div>


    <!-- PROBABILITY STATISTICS -->

    <div class="section">

        <h2 class="section-heading">
            Probability statistics
        </h2>

        <div class="section-description">
            Statistics calculated from the complete set
            of classified domain probabilities.
        </div>

        <div class="cards">

            <div class="card">

                <div class="card-title">
                    Mean DGA probability
                </div>

                <div class="value">
                    {summary['mean_dga_probability']:.2%}
                </div>

                <div class="card-accent"
                     style="background:#35B8D4;">
                </div>

            </div>


            <div class="card">

                <div class="card-title">
                    Median DGA probability
                </div>

                <div class="value">
                    {summary['median_dga_probability']:.2%}
                </div>

                <div class="card-accent"
                     style="background:#8299E8;">
                </div>

            </div>


            <div class="card">

                <div class="card-title">
                    Maximum DGA probability
                </div>

                <div class="value">
                    {summary['max_dga_probability']:.2%}
                </div>

                <div class="card-accent"
                     style="background:{DGA_COLOR};">
                </div>

            </div>


            <div class="card">

                <div class="card-title">
                    Minimum DGA probability
                </div>

                <div class="value">
                    {summary['min_dga_probability']:.2%}
                </div>

                <div class="card-accent"
                     style="background:{BENIGN_COLOR};">
                </div>

            </div>

        </div>

    </div>


    <!-- TOP 15 DOMAINS -->

    <div class="section">

        <h2 class="section-heading">
            Highest-scoring domains
        </h2>

        <div class="section-description">
            Displaying {displayed_count} of
            {summary['total_processed']:,} successfully
            classified domains, ranked by descending
            predicted DGA probability.
        </div>

        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>
                        <th>Rank</th>
                        <th>Domain</th>
                        <th>DGA probability</th>
                        <th>Classification</th>
                    </tr>

                </thead>

                <tbody>

                    {table}

                </tbody>

            </table>

        </div>

    </div>


    <!-- REJECTED INPUTS -->

    <div class="section">

        <h2 class="section-heading">
            Rejected input rows
        </h2>

        <div class="section-description">
            {summary['rejected']:,} invalid input rows
            were excluded from classification.
            Rejected rows can be downloaded separately
            as a CSV file from the application.
        </div>

    </div>


    <!-- INTERPRETATION AND LIMITATIONS -->

    <div class="note">

        <div class="note-heading">
            Interpretation and limitations
        </div>

        <p>
            Results are supervised machine-learning predictions
            based on lexical characteristics of domain names.
        </p>

        <p>
            A predicted DGA classification is not proof that
            a domain is malicious. Similarly, a benign
            classification does not guarantee that a domain
            is safe.
        </p>

        <p>
            This application does not perform DNS lookups,
            reputation checks or live network monitoring.
        </p>

        <p>
            Probability values represent model outputs and
            should not automatically be interpreted as
            calibrated probabilities.
        </p>

    </div>


    <!-- FOOTER -->

    <div class="footer">

        <strong>DOMAIN SENTINEL</strong>

        <br>

        Supervised Lexical DGA Domain Detection

        <br>

        Generated {generated}

    </div>

</div>

</body>

</html>
"""

    return report_document.encode("utf-8")
