# Customer Behaviour Analytics & Risk Prediction System

## Project Overview
An end-to-end machine learning system to identify three business-critical customer outcomes from transactional retail data:
- Churn risk
- High-value customer potential
- Future high-risk cancellation behavior
Built on ~540k transactions across ~4,400 customers, the system integrates unsupervised segmentation, NLP-based product embeddings, and supervised classification into a single modular pipeline, with a multi-page Streamlit dashboard for business-facing decision support.

---

## Dataset

- Source: [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- Scale: 541,909 transactions, 4,372 unique customers
- Fields: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- Nature: Event-level purchase and cancellation logs requiring customer-level aggregation

---

## Key Features
- End-to-end ML workflow: 
```
Raw Transactions
  -> Data Cleaning & Normalisation
  -> Temporal Split (reference-date framework)
  -> Customer-Level Feature Engineering (23 behavioural + 4 NLP features)
  -> Unsupervised Layer (KMeans segmentation + Isolation Forest anomaly detection)
  -> Supervised Layer (churn / high-value / high-risk classifiers)
  -> Threshold Tuning & Tier Assignment
  -> Customer-level predictions
  -> Streamlit Dashboard + Live Inference Endpoint
```
- Leakage-aware temporal framing with strict pre/post reference-date split
- Behavioral feature engineering (RFM-style, cancellation behavior, recency/frequency dynamics)
- NLP-enhanced customer representation via product description embeddings (MiniLM + UMAP + KMeans)
- Hybrid modeling strategy: clustering and anomaly flags used alongside supervised classifiers
- Class-imbalance aware evaluation and decision threshold tuning for F1 optimization
- Tier-based segmentation from predicted probabilities for action prioritization
- Streamlit dashboard for predictions, metrics, and customer segment analysis

---

## Tech Stack

| Category | Libraries |
|---|---|
| Language | Python |
| Data | pandas, numpy, openpyxl |
| ML | scikit-learn, xgboost, lightgbm, catboost |
| NLP | sentence-transformers (all-MiniLM-L6-v2) |
| Dimensionality Reduction | UMAP |
| Unsupervised | KMeans, Isolation Forest |
| Visualization | plotly, matplotlib, seaborn |
| Frontend | Streamlit |
| Model Persistence | joblib |

---

## Feature Engineering

**Behavioural & RFM features (23 total):**
- Purchase volume: total spend, order count, item count, avg/max/min/std order value
- Recency: days since last purchase, days since first purchase, purchase span
- Frequency dynamics: avg days between orders, activity gap flag, per-day spend
- Cancellation behaviour: cancellation count, cancellation amount, cancellation rate, return-purchase ratio, days since last cancellation, order completion rate
- Product breadth: unique product count, product diversity ratio

**NLP-based product features (4 total):**
- Product descriptions encoded with `all-MiniLM-L6-v2` (Sentence Transformers)
- Dimensionality reduced via UMAP (384 -> 8 dimensions)
- Cluster assignment via KMeans (k=14 product clusters)
- Customer-level: `product_cluster_diversity`, `primary_product_cluster`, `product_cluster_entropy`, `product_cluster_diversity_ratio`

---

## Modeling Approach
### Unsupervised Layer
**KMeans segmentation (k=3):**

| Cluster | Name | Behavioural Profile |
|---|---|---|
| 0 | One-Time Churners | Single purchase, high recency gap, low spend |
| 1 | Engaged Regulars | High frequency, low recency gap, broad product spread |
| 2 | At-Risk Irregulars | Moderate purchase history, early activity then dormant |

**Isolation Forest anomaly detection (contamination=0.05):**
- Flags customers whose behaviour deviates significantly from the population
- Anomaly scores and binary labels passed as features into supervised models

**Note on LOF:** Local Outlier Factor was implemented and evaluated but excluded from the final pipeline as legitimate high-spend VIP customers were consistently flagged as anomalies due to their density separation from the general population — a known limitation of density-based methods when genuine outlier classes exist. Isolation Forest, being tree-based, is more robust to this and was retained.

**Signal from unsupervised labels:** Cluster membership encodes behavioural archetype (e.g. one-time vs. engaged) which the supervised models cannot directly infer from raw numeric features alone. Anomaly flags capture global behavioural deviation. These are passed as one-hot encoded features alongside the core feature set. A systematic ablation study (see `outputs/`) compared baseline vs. unsupervised-augmented feature sets across all three tasks — results showed task-specific signal with marginal aggregate gains, validating that inclusion was evidence-based rather than arbitrary.

### Supervised Layer
Model families explored and/or configured in the project include:
- Logistic Regression
- KNN
- Naive Bayes
- SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- LGBM

#### Model Selection Process(supervised)

For each prediction task, 10 model families were evaluated at baseline (no hyperparameter tuning) with both baseline and unsupervised-augmented feature sets. Top candidates were then hyperparameter tuned via GridSearchCV with StratifiedKFold (k=5). Final models were selected based on tuned CV F1 and test set generalization.

#### Baseline Performance (pre-tuning, best model per task)

| Task | Best Model | CV F1 | Test F1 | ROC-AUC |
|---|---|---|---|---|
| Churn | SVM (baseline features) | 0.660 | 0.670 | 0.731 |
| High-Value | XGBoost (baseline features) | 0.593 | 0.568 | 0.833 |
| High-Risk | Naive Bayes (baseline features) | 0.193 | 0.224 | 0.826 |

#### Final Performance (post-tuning, ensemble/final model)

| Task | Final Model | CV F1 | Test F1 | ROC-AUC |
|---|---|---|---|---|
| Churn | Soft-vote ensemble (NB + SVM + RF + XGB) | 0.687 | 0.687 | 0.735 |
| High-Value | XGBoost | 0.658 | 0.615 | 0.854 |
| High-Risk | XGBoost (baseline features) | 0.231 | 0.313 | 0.791 |

**Churn improvement over best baseline:** CV F1 +0.027, Test F1 +0.017 — achieved through ensemble combination and threshold tuning (threshold=0.38 vs. default 0.50).

**High-value:** Hyperparameter tuning added ~0.07 F1 over the untuned XGBoost baseline(0.619 -> 0.689). ROC-AUC of 0.890 indicates strong customer ranking quality even where F1 is moderate.

**High-risk:** The low F1 (0.31) reflects extreme class imbalance rather than poor discrimination — ROC-AUC of 0.791 confirms the model ranks cancellation-prone customers correctly. Conservative thresholds (Watch List: >0.30, Urgent Attention: >0.60) are intentional to minimize false-positive intervention costs.

#### Churn Ensemble

The final churn model is a soft-voting ensemble combining four model families:

- Gaussian Naive Bayes (linear feature set, scaled)
- SVM with RBF kernel (linear feature set, scaled)
- Random Forest (tree feature set, unscaled)
- XGBoost (tree feature set, unscaled)

Average predicted probabilities are thresholded at 0.38 (CV-tuned). Feature importance is derived via manual permutation importance against ROC-AUC on a held-out sample.

#### Threshold Tuning

All models underwent F1 threshold tuning via StratifiedKFold cross-validation over thresholds in [0.10, 0.90]. The threshold maximising mean CV F1 was applied at test time.

| Task | Default F1 | Tuned F1 | Optimal Threshold |
|---|---|---|---|
| Churn (ensemble) | 0.679 | 0.687 | 0.38 |
| High-Value (XGB) | 0.615 | 0.689 | 0.53 |
| High-Risk (XGB) | 0.313 | 0.313 | 0.50 (no change) |

High-risk threshold tuning did not improve test F1 — this is expected behaviour under extreme class imbalance where CV-optimised thresholds overfit to training fold distributions.

### Segmentation Layer
Predicted probabilities are mapped to business-action tiers:

| Model | Low Tier | Mid Tier | High Tier |
|---|---|---|---|
| Churn | < 0.40 — Low Risk | 0.40–0.70 — Medium Risk | > 0.70 — High Risk |
| High-Value | < 0.40 — Standard | 0.40–0.70 — Growing Potential | > 0.70 — VIP |
| High-Risk | < 0.30 — Normal | 0.30–0.60 — Watch List | > 0.60 — Urgent Attention |

---
## Evaluation Metrics
F1-score is prioritised over accuracy due to significant class imbalance across all three targets. ROC-AUC is used as a threshold-independent measure of model discrimination. Where F1 is low (high-risk), ROC-AUC is the primary signal of usefulness — a model with ROC-AUC 0.79 is generating actionable risk rankings even if the binary classification boundary is imprecise.

---

## Results and Interpretation
### Final Performance (Approximate)
- Churn: F1 ~ 0.687, ROC-AUC ~ 0.735
- High Value: F1 ~ 0.667, ROC-AUC ~ 0.890
- High Risk: F1 ~ 0.312, ROC_AUC ~ 0.791 (highly imbalanced target)

### Interpretation
- Churn model shows dependable signal quality for proactive retention workflows.
- High-value model demonstrates strong ranking power (high ROC-AUC), making it effective for prioritization and upsell targeting.
- High-risk cancellation prediction remains difficult due to extreme imbalance; lower F1 is expected and still useful for watchlist-style triage when paired with probability tiers.


---

## Project Structure

```
customer behavioural analysis/
|- README.md
|- requirements.txt
|- TECHNICAL_DOCUMENTATION.md
|- data/
|  |- customer_predictions.csv
|  |- processed/
|  |  |- customer_features_with_labels.csv
|  |  |- customer_features.csv
|  |  |- customer_nlp_features_with_labels.csv
|  |  |- customer_nlp_features.csv
|  |  |- nlp_features.csv
|  |  |- product_clusters.csv
|  |- raw/
|- outputs/                        # Supervised model training result CSVs (baseline, unsup+NLP, post hyperparam tuning)
|- src/
|  |- config.py
|  |- data_preprocessing/
|  |  |- clean_data.py
|  |  |- feature_eng.py
|  |  |- load_data.py
|  |- models/
|  |  |- anomaly_detection.py      # Final IF model
|  |  |- anomaly_detection_exp.py  # IF vs LOF evaluation
|  |  |- clustering.py             # Final KMeans
|  |  |- clustering_exp.py         # KMeans / hierarchical / DBSCAN 
|  |  |- churn.py                  # Ensemble model + helper functions
|  |  |- high_risk_customer.py
|  |  |- high_val_customer.py
|  |  |- supervised_exp.py         # Baseline model comparisons
|  |  |- supervised_exp_tuning.py  # Model hyperparam tuning using GridSearchCV
|  |  |- f1_tuning_exp.py          # Threshold tuning experiments to improve F1 score
|  |  |- save_all.py
|  |  |- util.py                   #helper functions
|  |- pipelines/
|  |  |- train_pipeline.py
|  |  |- inference_pipeline.py
|  |- visualization/
|  |  |- anomaly_plots.py
|  |  |- behav_plots.py
|  |  |- cluster_plot.py
|  |  |- data_overview.py
|  |  |- feature_importance.py
|  |  |- tier_plots.py
|  |  |- visual.py
|- frontend/
|  |- app.py
|  |- pages/
|- stuff/                          # All model artifacts (~9MB)
|  |- nlp/
|  |  |- product_kmeans.pkl
|  |  |- umap_reducer.pkl
|  |- supervised/
|  |  |- churn_model.pkl           # Soft-vote ensemble dict
|  |  |- high_value_model.pkl      # XGBoost
|  |  |- high_risk_model.pkl       # XGBoost
|  |  |- scaler.pkl
|  |  |- results.json
|  |- unsupervised/
|  |  |- scaler.pkl
|  |  |- pca.pkl
|  |  |- cluster_model.pkl
|  |  |- isolation_forest.pkl
|- visuals/
|- notebooks/
|- testing_results/
|  |- v1_dataset_validation_report.md
|  |- v2_dataset_validation_report.md
```
*** Note: *** The model files and testing dataset(s) have been uploaded on google drive(link : https://drive.google.com/drive/folders/1WH3oCimeBjwfeCiXZ0_IrlanAPnCg-CG?usp=sharing)

---

## How to Run

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipelines.train_pipeline
streamlit run frontend/app.py

# Linux / Mac
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.pipelines.train_pipeline
streamlit run frontend/app.py
```

Place the raw dataset at `data/raw/online_retail.xlsx` (or `.csv`) before running the training pipeline. Pre-trained model artifacts are included in `stuff/` — the Streamlit app can be launched directly without retraining.

---

## Business Applicability

- **Retention teams:** target high churn-risk customers with early personalised interventions
- **Growth teams:** focus upsell and loyalty campaigns on VIP and growing-potential customers
- **Risk and operations:** monitor watch-list customers for cancellation-heavy patterns
- **Analytics:** tier-based outputs convert raw probabilities into action-ready business segments


