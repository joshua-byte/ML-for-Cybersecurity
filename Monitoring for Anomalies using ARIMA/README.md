# Network Anomaly Monitoring using ARIMA

## Files
- `app.py`: Streamlit interface for pre-aggregated CSV scoring and authorized live packet capture.
- `cesnet_arima_model.pkl`: exported statsmodels ARIMA(2,0,1) artifact and experimental thresholds.
- `requirements.txt`: Python dependencies.

## Run
1. Put all three files in the same folder.
2. Create/activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## CSV input
Upload one chronologically ordered row per fixed interval. The default feature is `packet_count`. The artifact was trained on one-minute packet counts from CESNET-CC25 scenario `7-u6_3`; use the same feature and interval semantics for meaningful results. Score one capture/scenario at a time.

## Live capture
Scapy capture runs on the machine hosting Streamlit, not on the user's browser machine. It requires a suitable network interface and authorized packet-capture privileges. The artifact was trained on one-minute bins; other bin sizes are for exploration only and require recalibration before meaningful use.

.
