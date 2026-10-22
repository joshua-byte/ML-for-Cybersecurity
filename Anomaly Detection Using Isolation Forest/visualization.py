import pandas as pd
import plotly.express as px


# =========================
# DATAFRAME CREATION
# =========================

def create_dataframe(X, feature_names, results):
    """
    Convert model output into a structured dataframe
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = results
    return df


# =========================
# 1. FLOW DURATION
# =========================

def plot_duration(df):
    fig = px.histogram(
        df,
        x="duration",
        color="label",
        title="Flow Duration Distribution",
        nbins=30
    )
    fig.update_layout(template="plotly_dark")
    return fig


# =========================
# 2. PACKET RATE (TRAFFIC INTENSITY)
# =========================

def plot_packets_rate(df):
    fig = px.scatter(
        df,
        x="total_packets",
        y="packets_per_sec",
        color="label",
        title="Traffic Intensity (Packets vs Rate)",
        hover_data=["total_bytes"]
    )
    fig.update_layout(template="plotly_dark")
    return fig


# =========================
# 3. ANOMALY DISTRIBUTION
# =========================

def plot_anomaly_pie(df):
    fig = px.pie(
        df,
        names="label",
        title="Anomaly vs Normal Traffic"
    )
    fig.update_layout(template="plotly_dark")
    return fig


# =========================
# 4. OUTLIER VISUALIZATION
# =========================

def plot_bytes_vs_packets(df):
    fig = px.scatter(
        df,
        x="total_bytes",
        y="total_packets",
        color="label",
        title="Bytes vs Packets (Outlier Detection)",
        hover_data=["duration"]
    )
    fig.update_layout(template="plotly_dark")
    return fig


# =========================
# 5. FEATURE CORRELATION (ADVANCED 🔥)
# =========================

def plot_correlation(df):
    numeric_df = df.drop(columns=["label"])

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Feature Correlation Matrix"
    )
    fig.update_layout(template="plotly_dark")
    return fig


# =========================
# 6. TOP ANOMALIES TABLE
# =========================

def get_top_anomalies(df, n=10):
    """
    Return most suspicious flows (based on packets/sec)
    """
    anomalies = df[df["label"] == "ANOMALY"]

    if len(anomalies) == 0:
        return pd.DataFrame()

    return anomalies.sort_values(
        by="packets_per_sec",
        ascending=False
    ).head(n)