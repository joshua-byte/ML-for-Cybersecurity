import numpy as np
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =========================
# FLOW CREATION
# =========================

def create_flows(packets, time_window=5):
    """
    Convert packets into flows based on 5-tuple + time window
    """

    flows = defaultdict(list)

    for pkt in packets:
        try:
            key = (
                pkt['src_ip'],
                pkt['dst_ip'],
                pkt['src_port'],
                pkt['dst_port'],
                pkt['protocol']
            )
            flows[key].append(pkt)
        except KeyError:
            continue  # skip malformed packets

    flow_list = []

    for key, pkts in flows.items():
        if not pkts:
            continue

        pkts = sorted(pkts, key=lambda x: x['timestamp'])

        start_time = pkts[0]['timestamp']
        current_window = []

        for pkt in pkts:
            if pkt['timestamp'] - start_time <= time_window:
                current_window.append(pkt)
            else:
                if current_window:
                    flow_list.append(current_window)
                current_window = [pkt]
                start_time = pkt['timestamp']

        if current_window:
            flow_list.append(current_window)

    return flow_list


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(flow):
    """
    Extract meaningful statistical + behavioral features
    """

    timestamps = [pkt['timestamp'] for pkt in flow]
    sizes = [pkt['length'] for pkt in flow]

    duration = max(timestamps) - min(timestamps)
    duration = max(duration, 1e-6)  # avoid division by zero

    total_packets = len(flow)
    total_bytes = sum(sizes)

    packets_per_sec = total_packets / duration
    bytes_per_sec = total_bytes / duration

    avg_packet_size = np.mean(sizes)
    std_packet_size = np.std(sizes)

    # 🔥 FIX: flags may contain multiple values like "SA", "FA"
    syn_count = sum(1 for pkt in flow if 'S' in pkt.get('flags', ''))
    ack_count = sum(1 for pkt in flow if 'A' in pkt.get('flags', ''))
    fin_count = sum(1 for pkt in flow if 'F' in pkt.get('flags', ''))

    return [
        duration,
        total_packets,
        total_bytes,
        packets_per_sec,
        bytes_per_sec,
        avg_packet_size,
        std_packet_size,
        syn_count,
        ack_count,
        fin_count
    ]


# =========================
# DATASET PREPARATION
# =========================

def prepare_dataset(packets):
    flows = create_flows(packets)

    if len(flows) == 0:
        return np.array([]), [], []

    feature_list = []
    flow_metadata = []

    for flow in flows:
        try:
            feature_list.append(extract_features(flow))
            flow_metadata.append(flow)
        except Exception:
            continue

    feature_names = [
        "duration",
        "total_packets",
        "total_bytes",
        "packets_per_sec",
        "bytes_per_sec",
        "avg_packet_size",
        "std_packet_size",
        "syn_count",
        "ack_count",
        "fin_count"
    ]

    return np.array(feature_list), feature_names, flow_metadata


# =========================
# MODEL TRAINING
# =========================

def train_model(X):
    """
    Train Isolation Forest
    """

    if len(X) == 0:
        raise ValueError("Empty dataset! No flows generated.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,      # 🔥 slightly stronger
        contamination=0.02,
        random_state=42
    )

    model.fit(X_scaled)

    return model, scaler


# =========================
# PREDICTION
# =========================

def predict(model, scaler, X):
    """
    Predict anomalies
    """

    if len(X) == 0:
        return []

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    return ["ANOMALY" if p == -1 else "NORMAL" for p in preds]


# =========================
# ANALYSIS REPORT
# =========================

def analyze_results(results):
    total = len(results)
    anomalies = results.count("ANOMALY")

    print("\n=== IDS ANALYSIS REPORT ===")
    print(f"Total Flows: {total}")
    print(f"Anomalies Detected: {anomalies}")

    if total > 0:
        print(f"Anomaly Percentage: {(anomalies / total) * 100:.2f}%")

    if anomalies > 0:
        print("\n⚠️ Suspicious activity detected!")
    else:
        print("\n✅ No major anomalies detected.")