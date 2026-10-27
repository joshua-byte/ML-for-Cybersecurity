# Investigating the DARPA Problem

## Overview

This project investigates the generalization limitations of supervised machine-learning models in network intrusion detection.

## Objective

To investigate the reliability of supervised intrusion detection models when evaluated on attack traffic different from the traffic used during training.

## Dataset

The project uses the CIC-IDS2017 dataset, specifically:

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

Dataset source: [CIC-IDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

## Methodology

1. Preprocessed the network traffic data and handled missing and infinite values.
2. Trained two supervised machine-learning models:
   - Random Forest
   - Logistic Regression
3. Evaluated the models on DDoS traffic and exported the trained models using joblib.
4. Tested the exported models on a separate dataset containing benign and PortScan traffic.

## Results

Both models performed extremely well on the held-out DDoS test set but performed poorly when evaluated on PortScan traffic.

| Model | PortScan Samples Detected | PortScan Recall |
|---|---:|---:|
| Random Forest | 56 / 158,930 | 0.0352% |
| Logistic Regression | 1 / 158,930 | 0.0006% |

The results indicate that the models failed to identify most PortScan traffic, classifying nearly all samples as benign.

## Conclusion

The results do not establish that supervised intrusion detection systems universally fail to generalize. Dataset differences and distribution shift may also contribute to the observed performance decline.

This project is an exploratory investigation inspired by the broader challenges associated with the DARPA intrusion detection problem.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook
