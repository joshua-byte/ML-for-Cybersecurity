# AI-Assisted Flow-Based Intrusion Detection System (IDS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![Scapy](https://img.shields.io/badge/Scapy-Networking-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Isolation%20Forest-orange?style=for-the-badge)
![CICIDS2017](https://img.shields.io/badge/Dataset-CICIDS2017-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

</div>

A real-time flow-based Intrusion Detection System (IDS) designed to analyze
network traffic and identify suspicious behavioral patterns using
machine-learning-driven anomaly detection.

The system captures network packets, aggregates them into communication flows,
extracts statistical traffic features, and applies an **Isolation Forest**
model to identify unusual network behavior such as abnormal packet bursts,
high-throughput communication, and scanning activity.

The anomaly-detection model was **trained and evaluated using the
CICIDS2017 benchmark dataset**, while the current application extends the
pipeline toward live network traffic analysis using Scapy.

Built with a detection-engineering mindset, the project explores lightweight
SOC-oriented workflows by combining behavioral traffic analytics, anomaly
scoring, and interactive visualization.

---

#  Detection Pipeline

```text
                    TRAINING / EVALUATION
                           │
                           ▼
                      CICIDS2017
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                    Isolation Forest
                           │
                           ▼
                    Trained Model
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Benchmark Traffic           Live Traffic
                                   Capture
              │                         │
              │                         ▼
              │                  Flow Aggregation
              │                         │
              │                         ▼
              │                  Feature Extraction
              │                         │
              └──────────────┬──────────┘
                             │
                             ▼
                     Anomaly Detection
                             │
                             ▼
                    Anomaly Scoring
                             │
                             ▼
                  Streamlit Visualization
                             │
                             ▼
                    Suspicious Flow
                       Inspection
```

---

#  Features

- Real-time packet capture using **Scapy**
- Flow-based network traffic analysis
- Machine-learning anomaly detection using **Isolation Forest**
- Model training/evaluation using **CICIDS2017**
- Behavioral traffic analytics
- Statistical network-flow feature extraction
- Interactive **Streamlit + Plotly** dashboard
- Suspicious-flow inspection
- Anomaly scoring and visualization
- Detection of abnormal traffic spikes and burst behavior
- Lightweight SOC-style traffic analysis workflow

---

#  Why Flow-Based Detection?

Traditional packet-level inspection can become noisy and computationally
expensive when analyzing large volumes of network traffic.

This project therefore focuses on **flow-based behavioral analysis**.

Instead of treating every packet as an isolated observation, packets are
aggregated into communication flows and analyzed using statistical properties
of the resulting traffic behavior.

This allows the system to investigate:

- unusual communication patterns,
- abnormal traffic intensity,
- packet and byte-rate deviations,
- burst behavior,
- irregular TCP activity,
- and other statistical characteristics of network flows.

The goal is to explore anomaly-oriented detection that does not depend
exclusively on predefined attack signatures.

---

#  System Architecture

The IDS operates through a behavioral flow-analysis pipeline.

## 1. Packet Capture

Network packets are captured in real time using **Scapy** from an active
network interface.

## 2. Flow Aggregation

Captured packets are grouped into logical communication flows using
attributes such as:

- Source IP
- Destination IP
- Source Port
- Destination Port
- Protocol
- Session timing

## 3. Feature Engineering

The system extracts behavioral traffic statistics including:

- Flow duration
- Total packets
- Total bytes
- Packets per second
- Bytes per second
- Average packet size
- Packet-size variance
- TCP SYN activity
- TCP ACK activity
- TCP FIN activity

## 4. Anomaly Detection

An **Isolation Forest** model analyzes network-flow behavior and identifies
statistical outliers that may represent suspicious or abnormal traffic.

The model is trained/evaluated using the **CICIDS2017** benchmark dataset.

## 5. Visualization & Triage

Detected anomalies are presented through an interactive dashboard to support
traffic inspection and lightweight anomaly-triage workflows.

---

# CICIDS2017

The project uses **CICIDS2017** as the benchmark dataset for model
training and evaluation.

The dataset provides labeled network traffic containing both benign and
attack-oriented traffic patterns.

The project uses the dataset to develop and evaluate the anomaly-detection
pipeline before applying the resulting approach to live traffic analysis.

The benchmark experiments include traffic associated with abnormal behaviors
such as:

- DDoS traffic
- Port scanning
- High-volume traffic
- Other anomalous network activity represented in the dataset

---

#  Dashboard Preview

## Detection Analytics Dashboard

<img width="1913" height="889" alt="dashboard" src="https://github.com/user-attachments/assets/0a3b4da0-fa7e-4db9-a093-609bb9a01350" />

The dashboard provides interactive behavioral traffic analytics for
anomaly-oriented monitoring.

Visualizations include:

- Flow-duration distributions
- Traffic-intensity analysis
- Anomaly ratios
- Byte-versus-packet analysis
- Suspicious-flow inspection

The interface is designed to provide a lightweight environment for
investigating anomalous network behavior.

---

# Suspicious Flow Analysis

## Top Suspicious Flows

<img width="1863" height="351" alt="suspicious" src="https://github.com/user-attachments/assets/b178f9fc-b5e3-4e39-a845-bc9080fddaa5" />

The suspicious-flow analysis module surfaces statistically anomalous
network flows using behavioral traffic indicators such as:

- Abnormal packet throughput
- Unusually large byte-transfer volumes
- High packets-per-second activity
- Sustained burst communication
- Irregular TCP communication patterns

This provides an analyst-oriented view of potentially abnormal network
behavior.

---

#  Example Detection Output

```text
[ALERT] Potential Anomalous Flow Detected

Source IP: 192.168.1.15
Destination IP: 192.168.1.1
Flow Duration: 4.99s
Packets/sec: 2020.83
Bytes/sec: 4262951.38
Anomaly Score: -0.42
Classification: Suspicious Burst Activity
```

The anomaly score is produced by the Isolation Forest model and is used as
an indicator of how strongly a flow deviates from the learned behavioral
distribution.

---

#  Simulated Attack Scenarios

The live IDS was tested against simulated abnormal traffic conditions
including:

- ICMP flood traffic
- Port-scanning behavior
- High-frequency packet bursts
- Abnormal connection spikes
- Rapid traffic generation
- Sustained high-throughput communication

These scenarios were used to investigate how anomalous traffic behavior
appears within the flow-level feature space and how effectively the system
surfaces suspicious activity.

---

# Security Relevance

This project explores practical concepts used in:

- Detection engineering
- SOC monitoring
- Anomaly-based intrusion detection
- Behavioral traffic analytics
- Network security
- AI-assisted security analytics

The system explores a defensive approach where statistical behavioral
deviations can be used to identify potentially suspicious traffic without
relying exclusively on static signatures.

---

# Detection Engineering Focus

The project focuses on:

```text
Behavioral Traffic Analysis
        +
Anomaly-Based Detection
        +
Flow-Level Analytics
        +
Machine Learning
        +
Analyst-Oriented Visualization
        ↓
Network Security Detection
```

The platform is intended as an educational and research-oriented environment
for experimenting with network detection engineering concepts.

---

# Tech Stack

## Languages & Frameworks

- Python

## Networking & Traffic Analysis

- Scapy

## Machine Learning

- Scikit-learn
- Isolation Forest

## Visualization

- Streamlit
- Plotly

## Data Processing

- Pandas
- NumPy

---

# Installation

## Clone Repository

```bash
git clone https://github.com/joshua-byte/Anomaly-Detection-IDS.git
cd Anomaly-Detection-IDS
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run IDS Dashboard

```bash
sudo streamlit run app.py
```

> Administrative privileges may be required for live packet capture depending
> on the operating system and network-interface configuration.

---

# Project Structure

```text
Anomaly-Detection-IDS/
│
├── app.py
├── capture.py
├── features.py
├── model_utils.py
├── visualization.py
├── logs.csv
├── requirements.txt
└── README.md
```

---

# Future Improvements

The project can be extended with:

- Real-time alert streaming
- Dedicated threat-classification models
- SIEM integration
- Multi-model anomaly detection
- Adaptive threshold tuning
- Distributed traffic monitoring
- ATT&CK-aligned event categorization
- Detection-rule correlation
- Automated reporting pipelines
- Real-time network alert notifications
- Signature-based detection integration
- Hybrid behavioral + signature-based detection
