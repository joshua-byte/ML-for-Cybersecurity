import streamlit as st
import pandas as pd
import time

from capture import start_capture, stop_capture, get_packet_count
from model_utils import prepare_dataset, train_model, predict

from visualization import (
    create_dataframe,
    plot_duration,
    plot_packets_rate,
    plot_anomaly_pie,
    plot_bytes_vs_packets,
    plot_correlation,
    get_top_anomalies
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.title("🚨 Intrusion Detection System")
st.write("Flow-Based Anomaly Detection using Isolation Forest")

# =========================
# SESSION STATE
# =========================

if "capturing" not in st.session_state:
    st.session_state.capturing = False

if "packets" not in st.session_state:
    st.session_state.packets = []

if "results" not in st.session_state:
    st.session_state.results = None

if "df" not in st.session_state:
    st.session_state.df = None

# =========================
# CONTROL PANEL
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Capture"):
        start_capture()
        st.session_state.capturing = True
        st.session_state.results = None
        st.session_state.df = None

with col2:
    if st.button("⏹ Stop Capture"):
        packets = stop_capture()
        st.session_state.packets = packets
        st.session_state.capturing = False

with col3:
    if st.button("🔍 Run Analysis"):
        if len(st.session_state.packets) == 0:
            st.warning("No packets captured yet!")
        else:
            with st.spinner("Analyzing traffic..."):
                X, feature_names, _ = prepare_dataset(st.session_state.packets)

                if len(X) == 0:
                    st.error("No valid flows detected!")
                else:
                    model, scaler = train_model(X)
                    results = predict(model, scaler, X)

                    df = create_dataframe(X, feature_names, results)

                    st.session_state.results = results
                    st.session_state.df = df

# =========================
# LIVE CAPTURE (SAFE REFRESH)
# =========================

st.subheader("📡 Live Capture")

if st.session_state.capturing:
    count = get_packet_count()
    st.metric("Packets Captured", count)

    # 🔥 controlled refresh (prevents infinite loop crash)
    time.sleep(1)
    st.rerun()

else:
    st.info("Capture is not running")

# =========================
# RESULTS
# =========================

if st.session_state.results:

    df = st.session_state.df
    results = st.session_state.results

    total = len(results)
    anomalies = results.count("ANOMALY")

    st.subheader("📊 Analysis Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", total)
    col2.metric("Anomalies", anomalies)
    col3.metric("Anomaly %", f"{(anomalies / total) * 100:.2f}%")

    st.markdown("---")

    # =========================
    # 📈 GRAPH DASHBOARD
    # =========================

    st.subheader("📈 Traffic Behavior Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_duration(df), use_container_width=True)
        st.plotly_chart(plot_anomaly_pie(df), use_container_width=True)

    with col2:
        st.plotly_chart(plot_packets_rate(df), use_container_width=True)
        st.plotly_chart(plot_bytes_vs_packets(df), use_container_width=True)

    # 🔥 Advanced graph
    st.plotly_chart(plot_correlation(df), use_container_width=True)

    st.markdown("---")

    # =========================
    # 🚨 TOP ANOMALIES
    # =========================

    st.subheader("🚨 Top Suspicious Flows")

    top_anomalies = get_top_anomalies(df)

    if not top_anomalies.empty:
        st.dataframe(top_anomalies)
    else:
        st.success("No significant anomalies detected")

    st.markdown("---")

    # =========================
    # 🔎 FLOW RESULTS
    # =========================

    st.subheader("🔎 Flow Classification")

    for i, row in df.head(50).iterrows():
        if row["label"] == "ANOMALY":
            st.error(f"Flow {i} → ANOMALY")
        else:
            st.success(f"Flow {i} → NORMAL")

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("Flow-Based IDS • Isolation Forest • Real-Time Monitoring")