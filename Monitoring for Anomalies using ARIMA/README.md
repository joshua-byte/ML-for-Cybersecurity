# Network Anomaly Monitoring Using ARIMA

A Python-based network monitoring application that uses ARIMA time-series forecasting to detect unusual patterns in network traffic. It supports CSV-based analysis, authorized live packet capture, anomaly visualization, and HTML report generation through a Streamlit interface.

## Features
- ARIMA(2, 0, 1) time-series forecasting.
- Residual-based anomaly detection.
- CSV traffic analysis and live packet capture using Scapy.
- Traffic visualization and HTML report generation.

## Project Files
- `app.py` — Streamlit application.
- `arima_model.pkl` — Trained ARIMA model artifact.
- `capture.py` — Live packet capture and traffic aggregation.
- `report.py` — HTML report generation.
- `requirements.txt` — Python dependencies.
- `/training` - the dataset and training of the model.

## Installation & Usage

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:

   ```bash
   streamlit run app.py
   ```

## Input Requirements

- Upload a chronologically ordered CSV containing one observation per fixed time interval.
- The default feature is `packet_count`.
- The model was trained on one-minute packet counts from the CESNET-CC25 dataset, scenario `7-u6_3`. Matching the feature and interval is recommended for meaningful results.

## Live Monitoring

Live packet capture runs on the machine hosting Streamlit and requires a suitable network interface and authorized capture privileges. Only monitor networks you are authorized to access.

**Note:** Anomaly alerts indicate deviations from predicted traffic patterns and do not independently confirm malicious activity. Results on different traffic distributions may require model recalibration.
