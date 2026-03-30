# Technical Documentation
## Customer Behaviour Analytics & Risk Prediction System

## 1. Problem Definition

### 1.1 Objective
Develop a multi-task machine learning system that predicts customer-level outcomes from transactional retail data:
- Churn propensity
- High-value customer propensity
- Future high-risk cancellation propensity

### 1.2 Business Relevance
The system is designed to support three operational workflows:
- Retention intervention (churn prevention)
- Revenue growth prioritization (high-value targeting)
- Risk monitoring (future cancellation management)

### 1.3 Prediction Scope
Input data is event-level (transactions), but outputs are customer-level probabilities, class labels, and decision tiers.

---

## 2. Dataset Description

### 2.1 Source and Scale
- Source type: Online retail transaction records
- Approximate scale (raw): ~500k transactions
- Working cleaned scale (as reported in EDA): 406,789 transactions
- Customer entities: ~4k unique customers

### 2.2 Core Fields
- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

### 2.3 Data Characteristics
- Mixed purchase and return/cancellation behavior
- Heavy right-skew in quantity and monetary distributions
- Outlier-rich retail distribution (bulk vs standard buyers)
- Strongly imbalanced downstream labels for high-risk class

---

## 3. Data Cleaning Steps (with Reasoning)

Implemented in preprocessing modules:
- src/data_preprocessing/load_data.py
- src/data_preprocessing/clean_data.py

### 3.1 Column normalization
- Convert names to lowercase
- Remove bracketed text and non-alphanumeric characters
- Standardize to snake_case

Reasoning:
Ensures deterministic schema handling across train/inference pipelines.

### 3.2 Missing customer ID removal
- Drop rows where customerid is null

Reasoning:
Customer-level aggregation and labeling require stable customer identity; null IDs cannot be reliably assigned to entities.

### 3.3 Duplicate row removal
- Remove exact duplicates

Reasoning:
Prevents artificial inflation of frequency and monetary features.

### 3.4 Invalid price filtering
- Retain only rows with unitprice > 0

Reasoning:
Non-positive prices can represent non-standard accounting events and distort monetary statistics used for modeling.

### 3.5 Transaction-type decomposition
- Identify cancellation invoices: InvoiceNo starts with 'C' OR Quantity < 0
- Separate purchase and cancellation behavior:
  - If cancellation: cancel_qty = |Quantity|, purchase_qty = 0
  - If purchase: purchase_qty = Quantity, cancel_qty = 0
  - cancel_amnt = cancel_qty * UnitPrice
  - purchase_amnt = purchase_qty * UnitPrice
- Create is_cancellation boolean flag

This helps to preserve cancellation behavior as informative signal rather than discarding negative quantities.

### 3.6 Temporal parsing
- Parse invoicedate and derive month period columns

Reasoning:
Enables robust time-based feature engineering and temporal split logic.

---

## 4. Feature Engineering (Detailed)

Implemented mainly in:
- src/data_preprocessing/feature_eng.py
- src/models/util.py

### 4.1 Reference Date Framework
A reference date t_ref is used to separate historical feature windows from future label windows.

Default implementation:
- t_ref = max(invoicedate) - 3 months

Data partitioning:
- D_before = {rows where invoicedate <= t_ref}
- D_after  = {rows where invoicedate > t_ref}

Purpose:
- Simulate realistic forecasting where only historical data is available at prediction time.
- Reduce target leakage risk by strict temporal causality.

### 4.2 Purchase Features (Customer-level)
For customer i using D_before:
- Total purchase:
  - T_i = sum of purchase_amnt
- Order count:
  - O_i = number of unique purchase invoices
- Total purchased items:
  - Q_i = sum of purchase_qty
- Product diversity:
  - U_i = number of unique stockcode
- Average order value:
  - AOV_i = T_i / O_i
- Average items per order:
  - IPO_i = Q_i / O_i
- Product diversity ratio:
  - PDR_i = U_i / Q_i

Invoice-level order total variance descriptors:
- max_order_val, min_order_val, std_order_val

Recency and activity features:
- days_since_last_purchase = (t_ref - last_purchase_date_i).days
- days_since_first_purchase = (t_ref - first_purchase_date_i).days
- purchase_span = (last_purchase_date_i - first_purchase_date_i).days
- avg_days_between_orders = purchase_span / (O_i - 1), if O_i > 1 else 0

### 4.3 Cancellation Features
From cancellation events in D_before:
- total_cancellation_count
- total_cancellation_amnt
- total_cancelled_qty
- days_since_last_cancellation = (t_ref - last_cancel_date_i).days

Imputation logic:
- Missing cancellation values are filled with zeros
- days_since_last_cancellation fallback uses days_since_first_purchase when no cancellation exists

### 4.4 Derived Behavioral Features
For each customer i:
- cancellation_rate:
  - cancellation_count / (order_count + cancellation_count)
- order_completion_rate:
  - order_count / (order_count + cancellation_count)
- return_purchase_ratio:
  - total_cancelled_qty / tot_items (safe-guarded for zero division)
- per_day_purchase_amnt:
  - total_purchase / days_since_first_purchase
- activity_gap (binary):
  - 1 if days_since_last_purchase > 2 * avg_days_between_orders for multi-order customers, else 0

Skew handling for selected variables (log transform):
- x_transformed = log(1 + x)

### 4.5 Label Engineering (Future Window)
Using D_after:
- Churn label:
  - churn_i = 1 if customer has no non-cancellation purchase in D_after, else 0
- High-value label:
  - Compute future_purchase_amnt_i in D_after
  - threshold = 80th percentile of future_purchase_amnt
  - high_value_customer_i = 1 if future_purchase_amnt_i > threshold else 0
- High-future-cancellation label:
  - high_future_cancellation_i = 1 if
    - total_future_cancel_i >= 3
    - OR future_cancel_ratio_i >= 0.20
- Implementation note:
  - All three labels are binary (0/1) classification targets
  - Labels are created at customer level from aggregated post-reference behavior
  - Customers with no post-reference transactions:
    - churn = 1 (by definition)
    - high_value_customer = 0 (no future value demonstrated)
    - high_future_cancellation = 0 (no future cancellation behavior)

### 4.6 Unsupervised/NLP Feature Augmentation
Additional features include cluster and anomaly labels/scores, and product-cluster behavioral NLP features (detailed in Sections 9 and 10).

### 4.7 Leakage Control Note
The intended design enforces D_before-only feature creation and D_after-only label creation. A dedicated leakage analysis report is present at insights/NLP_Feature_Leakage_Report.md and should be considered in model governance.

---

## 5. Model Architectures Used

### 5.1 Supervised Model Families Evaluated
Implemented in src/models/supervised_exp.py and src/models/supervised_exp_tuning.py:
- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine (RBF)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- LightGBM

### 5.2 Final Pipeline Models
Implemented in training/inference pipeline:
- Churn: Gaussian Naive Bayes
- High-value: XGBoost classifier
- High-risk: XGBoost classifier

### 5.3 Unsupervised Components
- KMeans clustering (k=3)
- Isolation Forest (contamination=0.05)
- Local Outlier Factor (n_neighbors=20, contamination=0.05, novelty=True)

### 5.4 NLP Representation Components
- SentenceTransformer: all-MiniLM-L6-v2 (384-dimensional embeddings)
  - Not saved; reloaded from HuggingFace in inference
- UMAP reducer: 384 → 8 dimensions (saved as umap_reducer.pkl)
- Product KMeans: k=14 clusters (saved as product_kmeans.pkl)

Note: In inference, product descriptions are re-encoded with the same SentenceTransformer model, then transformed using saved UMAP and clustered using saved KMeans. This ensures consistent product cluster assignments.

---

## 6. Training Procedure

Primary orchestration in src/pipelines/train_pipeline.py.

### 6.1 Data Flow
1. Load raw transaction dataset
2. Clean data
3. Generate customer features + labels
4. Build unsupervised customer representations (scaled -> PCA -> cluster/anomaly)
5. Merge NLP product-cluster features
6. Train task-specific supervised models
7. Evaluate and persist model artifacts and summary metrics
8. Generate customer-level predictions and tier labels

### 6.2 Train/Test Split
For each supervised target:
- Stratified split
- test_size = 0.20
- random_state = 42

### 6.3 Preprocessing by Model Type
- Linear/probabilistic path:
  - skew correction + scaling
- Tree path:
  - no mandatory scaling

### 6.4 Artifact Persistence
Saved with joblib/json under stuff/:

Unsupervised (stuff/unsupervised/):
- scaler.pkl: StandardScaler fit on behavioral features
- pca.pkl: PCA reducer (95% variance)
- cluster_model.pkl: K-Means (k=3)
- isolation_forest.pkl: Anomaly detector (global)
- lof_novelty.pkl: Local outlier factor detector

NLP (stuff/nlp/):
- umap_reducer.pkl: UMAP dimensionality reducer
- product_kmeans.pkl: Product cluster model (k=14)

Supervised (stuff/supervised/):
- churn_model.pkl: Naive Bayes classifier
- high_value_model.pkl: XGBoost classifier
- high_risk_model.pkl: XGBoost classifier
- scaler.pkl: StandardScaler for linear model features (separate from unsupervised scaler)
- results.json: Performance metrics (F1, ROC-AUC, accuracy per target)

Note: Two different scalers are saved:
- stuff/unsupervised/scaler.pkl: For unsupervised feature transformation
- stuff/supervised/scaler.pkl: For churn model (linear preprocessing)

---

## 7. Evaluation Strategy

### 7.1 Primary Metrics
- F1-score
- ROC-AUC
- Accuracy (reported, but secondary under imbalance)

### 7.2 Cross-validation
- 5-fold StratifiedKFold used in supervised experiments
- CV mean and standard deviation captured for F1 and ROC-AUC

### 7.3 Why These Metrics
- F1 balances precision-recall for minority class behavior
- ROC-AUC evaluates discrimination quality independent of specific threshold

### 7.4 Task Difficulty
- Churn: moderate class balance, stable performance
- High-value: moderate imbalance, strong ranking performance
- High-risk: severe imbalance, lower F1 expected due to minority scarcity

---

## 8. Threshold Optimization

Implemented in src/models/f1_tuning_exp.py.

### 8.1 Motivation
Default threshold p >= 0.5 may be suboptimal under class imbalance and asymmetric business costs.

### 8.2 Procedure
For a candidate model:
1. Obtain probability scores on train and test
2. Sweep thresholds from 0.10 to 0.89 (step 0.01)
3. For each threshold, compute CV F1 on train folds
4. Select threshold maximizing mean CV F1
5. Evaluate tuned threshold on test set

### 8.3 Outcome
Produces:
- best_threshold
- best_cv_f1
- test F1 at default threshold
- test F1 at tuned threshold
- absolute improvement

---

## 9. Unsupervised Learning Approach

### 9.1 Clustering Experiments
Implemented in src/models/clustering_exp.py and documented in insights/Clustering_results.md.

Algorithms explored:
- KMeans (k=2..12)
- Agglomerative clustering (ward/complete/average)
- DBSCAN

Selection rationale:
- KMeans k=3 chosen for practical balance of separation and actionable segment sizes
- PCA-based feature space marginally improved silhouette over raw scaled space

### 9.2 Anomaly Detection
Implemented in src/models/anomaly_detection.py and anomaly_detection_exp.py:
- Isolation Forest
- Local Outlier Factor

Output integration:
- Binary anomaly labels and anomaly scores are appended to customer features
- These features are then used by supervised models as additional behavioral risk signals

---

## 10. NLP Feature Integration

### 10.1 Product Text Embedding
- Product descriptions are encoded using all-MiniLM-L6-v2 sentence embeddings

### 10.2 Dimensionality Reduction and Product Clustering
- UMAP reduces embedding dimensionality
- KMeans assigns product clusters

### 10.3 Customer-level NLP Features
Aggregated from customer-product interactions:
- product_cluster_diversity (number of unique clusters interacted with)
- primary_product_cluster (mode of cluster usage)
- product_cluster_entropy (distribution entropy of cluster interactions)

Entropy formulation:
- H_i = -sum_k p_ik * log(p_ik + epsilon)
where p_ik is cluster proportion for customer i in cluster k.

### 10.4 Integration Point
NLP-derived features are merged into customer tabular features before supervised training and inference.

---

## 11. Pipeline Design (Training vs Inference)

### 11.1 Training Pipeline
Entry point:
- src/pipelines/train_pipeline.py

Responsibilities:
- Build all preprocessors/models from data
- Create labels (training-only)
- Serialize full artifact stack
- Export predictions and visuals

### 11.2 Inference Pipeline
Entry point:
- src/pipelines/inference_pipeline.py (called by Streamlit app)

Key function: predict_all_customers(raw_transaction_df)

Responsibilities:
- Validate input schema (required columns check)
- Normalize column names to lowercase
- Clean data using same pipeline as training
- Compute reference date dynamically: max(InvoiceDate) from uploaded data
- Engineer behavioral features using dynamic reference date
- Transform features using saved scaler and PCA (no fitting)
- Predict cluster and anomaly labels using saved models
- Generate NLP features using saved UMAP and K-means
- Score customers with saved supervised models
- Convert probabilities to tier labels
- Return: predictions_df, customer_features_df, metrics, warnings, upload_status

Critical difference from training:
- Reference date is dynamic (max of uploaded data), not fixed
- All transformers/models use .transform() or .predict() only, never .fit()
- No label generation (labels not needed for prediction)

### 11.3 UI Integration
Streamlit app:
- frontend/app.py and frontend/pages
- Supports default repository data and user-uploaded transaction CSV
- Displays metrics, predictions, and segment analytics

---

## 12. Inference Pipeline Technical Details

### 12.1 Reference Date Strategy
Training: Fixed date (2011-09-09)
Inference: Dynamic date computed as max(InvoiceDate) from uploaded data

Rationale: User uploads can contain data from any time period. Using the maximum date ensures recency features are computed relative to the latest activity in the dataset.

### 12.2 Model Loading and Caching
All models are loaded once and cached globally using Python's module-level caching:
- First call to load_model() loads all 10 model files
- Subsequent calls return cached models
- Models persist across multiple inference requests in the same session
- Streamlit uses @st.cache_resource decorator for additional caching

### 12.3 Feature Engineering Consistency
Training and inference use the same core feature calculation functions but with different orchestration:
- Training: Calls wrapper function feature_eng() with fixed reference date
- Inference: Calls individual functions (purchase_features, cancellation_features, derive_features) with dynamic reference date

This separation prevents accidental model refitting in production while maintaining feature calculation consistency.

### 12.4 NLP Pipeline in Inference
1. Load SentenceTransformer('all-MiniLM-L6-v2') - not saved, reloaded each time
2. Encode product descriptions to 384-dimensional embeddings
3. Transform embeddings using saved UMAP reducer (no fitting)
4. Assign product clusters using saved K-means (predict only, no fitting)
5. Aggregate customer-level NLP features (diversity, entropy, primary cluster)

### 12.5 Scaling Strategy
Only the churn model requires scaled features (Naive Bayes is sensitive to feature magnitude).
- Training: Fit StandardScaler on churn features, save to stuff/supervised/scaler.pkl
- Inference: Load scaler, transform churn features only
- High-value and high-risk models (XGBoost) receive raw features (no scaling)

Note: Separate from the scaler used for unsupervised learning (stuff/scaler.pkl).

### 12.6 Output Format
predict_all_customers() returns a dictionary:
{
  'predictions_df': DataFrame with customerid, probabilities, predictions, tiers for all 3 targets
  'customer_features_df': DataFrame with all 29 engineered features + cluster/anomaly labels
  'metrics': Dictionary with model performance from training (F1, ROC-AUC per target)
  'warnings': List of data quality warnings (e.g., high row removal rate during cleaning)
  'upload_status': Dictionary with row counts (uploaded, cleaned, customers, features_created)
}

### 12.7 Error Handling
- Missing required columns: Raises ValueError with list of missing columns
- Invalid data types: datetime parsing errors are caught and flagged in warnings
- Empty results after cleaning: Raises RuntimeError
- Model loading failures: Raises FileNotFoundError with missing model path

## 13. Limitations

1. Class imbalance severity for high_future_cancellation limits F1 performance.
2. Fixed heuristic thresholds for tiering may not be optimal across business contexts.
3. Single split evaluation in final pipeline can underrepresent variance compared to repeated validation.
4. Potential leakage risk in NLP feature construction requires strict governance, as identified in dedicated leakage analysis.
5. Batch-oriented design; no native online feature store or streaming inference path.
6. Limited explicit probability calibration analysis (e.g., Platt/Isotonic).
7. Single scaler complexity: Two separate scalers (unsupervised and supervised) increases artifact management burden; consolidation to single scaler recommended for future versions.

---
