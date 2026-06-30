# 🛡️ SentinelAI — Anomaly Detection System

An unsupervised machine learning project that detects fraudulent credit card transactions using Isolation Forest and DBSCAN, without relying on labeled fraud examples during training.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)

---

## 🎯 Overview

This project builds a complete anomaly detection pipeline that:

1. **Loads** the Credit Card Fraud dataset (anonymized transaction features)
2. **Analyzes** severe class imbalance (fraud cases are <1% of data)
3. **Scales** features to normalize transaction amount and time
4. **Applies** Isolation Forest to isolate anomalous transactions
5. **Applies** DBSCAN to find density-based outliers for comparison
6. **Tunes** contamination/threshold parameters against known fraud labels
7. **Evaluates** using precision-recall, since accuracy is misleading on imbalanced data
8. **Visualizes** anomaly scores and decision boundaries for explainability

## 🧠 Why Unsupervised Learning

Fraud detection in production rarely has complete labels for new fraud patterns. Unsupervised methods like Isolation Forest learn what "normal" transactions look like and flag deviations, making them effective at catching novel fraud types that supervised models trained only on past labeled fraud would miss.

## 📊 Algorithms Used

| Algorithm | Type | Why Used |
|-----------|------|----------|
| Isolation Forest | Tree-based | Efficiently isolates anomalies with fewer splits than normal points |
| DBSCAN | Density-based | Flags points in low-density regions as outliers |

## 🏗️ Pipeline

```
Raw Transactions → Feature Scaling → Isolation Forest / DBSCAN
                                              ↓
                                    Anomaly Scoring
                                              ↓
                              Precision-Recall Evaluation → Flagging Logic
```

## 📁 Project Structure

```
sentinelai/
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── SentinelAI_Analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── detector.py
│   ├── evaluation.py
│   └── visualize.py
├── outputs/
│   ├── anomaly_scores.png
│   ├── precision_recall_curve.png
│   └── flagged_transactions.csv
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Pavan67u/anomaly-detection.git
cd anomaly-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook notebooks/SentinelAI_Analysis.ipynb
```

### 3. Run as Script

```bash
python src/detector.py --data data/creditcard.csv --algorithm isolation_forest
python src/detector.py --data data/creditcard.csv --algorithm dbscan
```

## 📈 Methodology

### Handling Class Imbalance
Dataset has roughly 492 fraud cases out of 284,807 transactions. Contamination parameter for Isolation Forest set based on this known ratio.

### Feature Scaling
Transaction Amount and Time scaled using StandardScaler, while PCA-derived features (V1-V28) are already normalized in the source dataset.

### Threshold Tuning
Anomaly score threshold tuned by scanning across precision-recall tradeoff curve to balance catching fraud against false alarms.

### Evaluation Strategy
Precision, recall, and F1-score used instead of accuracy, since accuracy is misleading when 99%+ of transactions are legitimate.

## 🛠️ Tech Stack

- **Data:** pandas, numpy
- **ML:** scikit-learn (IsolationForest, DBSCAN, StandardScaler)
- **Visualization:** matplotlib, seaborn
- **Evaluation:** precision_recall_curve, classification_report

## ⚠️ Disclaimer

This project is for educational and portfolio purposes, demonstrating anomaly detection techniques on a public dataset.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
