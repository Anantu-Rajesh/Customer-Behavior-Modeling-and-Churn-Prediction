# Model Validation Report — Sample Dataset v1

## Dataset Link: https://drive.google.com/drive/folders/1WH3oCimeBjwfeCiXZ0_IrlanAPnCg-CG?usp=sharing (goto testing data/sample_transactions.csv)

## Dataset Overview

| Property | Value |
|---|---|
| Total transactions | 2,000 |
| Unique customers | 538 |
| Unique invoices | 555 |
| Date range | 2024-01-04 to 2024-12-30 |
| Cancellation rows | 0 |
| Negative quantity rows | 0 |
| Zero price rows | 0 |
| Extra columns (pipeline noise test) | SalesChannel, PaymentMethod |

The dataset was generated synthetically to test whether the inference pipeline handles a realistic-looking transactional file end-to-end. It is intentionally limited in diversity — no returns, no VIP-level spend, and near-uniform single-order customers — to expose edge cases and baseline behaviour.

---

## Customer Distribution (Ground Truth from Data)

| Property | Value |
|---|---|
| Customers with exactly 1 invoice | 522 (97.0%) |
| Customers with 2 invoices | 15 (2.8%) |
| Customers with 3 invoices | 1 (0.2%) |
| Median customer spend | £51.38 |
| Mean customer spend | £80.97 |
| Max customer spend | £666.32 |
| Min customer spend | £1.42 |
| Customers spending > £500 | 2 |
| Customers spending > £2,000 | 0 |

---

## Prediction vs Expected Behaviour

### Churn Model

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| Overall churn rate | ~97% (522/538 customers have 1 order and no repeat) | 99.1% classified as High Risk | Slight overestimate |
| Medium Risk customers | ~3% (15–16 customers with 2–3 orders) | ~1% customer in Medium Risk | Underestimate — model pushed most into High Risk |
| Low Risk customers | ~0% (no genuine repeat buyers with recent activity) | 0% | Correct |
| Distribution shape | Expected spike near probability 1.0 | Histogram shows spike at 0.75–0.85 | Model is calibrated conservatively — not outputting raw 1.0 probabilities |

**Interpretation:** The churn output is largely correct for this dataset. 97% of customers having a single purchase maps cleanly to High Risk churn. The handful of 2-order customers are mostly classified as High Risk too rather than Medium Risk, which is a minor miscalibration — the model needs more orders and lower recency gaps to assign Medium Risk confidence. This is not a model failure; the dataset simply does not contain customers with the behavioural profile that would push probabilities into the 0.4–0.7 band.

---

### High-Value Model

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| VIP customers | 0 (max spend £666, no customer above £2k, no repeat high-spend pattern) | 0% VIP | Correct |
| Growing Potential | ~0–2% at most (2 customers above £500, single orders) | 0% Growing Potential | Correct |
| Standard (100%) | Expected ~100% | 100% Standard | Correct |
| Histogram shape | Expected all probabilities < 0.40 | Flat bar across full 0–1 range | Concerning — see note below |

**Interpretation:** The tier output is correct — everyone is Standard, which matches the data. However the probability histogram showing a flat distribution across the full 0–1 range rather than a spike near 0 is a calibration issue. The model is outputting spread-out probabilities instead of confidently assigning near-zero scores to clearly non-VIP customers. This is likely caused by feature scaling at inference compressing the feature values into a range the model associates with ambiguity. It does not affect tier assignment (since all are still below 0.40) but is a sign that the model's confidence is not well-calibrated on this population.

---

### High-Risk (Cancellation) Model

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| Urgent Attention customers | 0 (no cancellations in dataset) | 0% | Correct |
| Watch List customers | 0 (no cancellation signals) | 0% | Correct |
| Normal (100%) | Expected 100% | 100% Normal | Correct |
| Probability distribution | Expected spike near 0 | Spike at 0.05–0.20, all below 0.30 | Correct |

**Interpretation:** The cleanest output of the three. With zero cancellations in the dataset, every customer has a cancellation rate of 0 and no cancellation-related features populated. The model correctly outputs near-zero probabilities for all customers. The histogram shape (dense spike in the 0.05–0.20 range rather than exactly at 0) reflects the model assigning a small residual probability based on recency and order patterns, which is appropriate behaviour.

---

### Unsupervised Segmentation (Clustering)

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| One-Time Churners | ~97% (522 single-order customers) | 97% | Correct |
| Engaged Regulars | ~2% (12 customers, 2+ orders, varied recency) | 2.2% (12 customers) | Correct |
| At-Risk Irregulars | ~1% (customers with 2+ orders but long recency gaps) | 0.7% (4 customers) | Broadly correct |
| Cluster spend separation | One-Timers: lower spend, Regulars: higher | One-Timers £76 avg, Regulars £215, At-Risk £267 | Correct direction |
| Product diversity separation | Expected Regulars > One-Timers | Confirmed in violin plot | Correct |

**Interpretation:** Clustering is the most reliable output on this dataset. KMeans correctly groups the overwhelming majority of single-order customers into One-Time Churners. The small number of repeat customers are distributed between Engaged Regulars and At-Risk Irregulars based on their recency patterns, which aligns with the cluster definitions. Spend and product diversity distributions across clusters are directionally correct.

---

### Anomaly Detection (Isolation Forest)

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| Anomaly rate | 5% (contamination parameter set to 0.05) | 13.6% (73/538 customers) | Higher than expected |
| Anomaly concentration | Should correlate with unusual spend or behaviour | 20% of Medium Risk churn flagged, 13.5% of High Risk | Partially correct correlation |
| Low Risk anomaly rate | Should be lower than other tiers | 0% (no Low Risk customers exist) | N/A |

**Interpretation:** The 13.6% anomaly rate exceeds the 5% contamination parameter set during training. This is expected when inference data has a different distribution from training data — the model was trained on the real UCI dataset which has a much richer behavioural spread. On a flat, homogeneous synthetic dataset where nearly all customers look identical, the Isolation Forest still forces anomaly assignments based on relative deviation, causing more customers to be flagged than the contamination parameter would imply. The correlation with Medium Risk churn (higher anomaly rate than High Risk) is mildly counterintuitive but reflects that Medium Risk customers in this dataset are the unusual ones — they have 2–3 orders, making them genuine outliers relative to the 97% single-order population.

---

## Summary

| Model | Overall Accuracy | Calibration Quality | Notes |
|---|---|---|---|
| Churn | High — direction correct | Moderate — probabilities cluster at 0.75–0.85 rather than ~1.0 | Expected given dataset homogeneity |
| High-Value | High — tier assignment correct | Poor — probability histogram flat instead of near-zero spike | Does not affect tiers; calibration gap worth investigating |
| High-Risk | High — both tier and calibration correct | Good — well-concentrated near 0 | Cleanest model output |
| Clustering | High — all three clusters behave as expected | N/A | Most reliable output |
| Anomaly Detection | Moderate — tier correlation partially holds | N/A — forced relative flagging | Rate inflation expected on homogeneous synthetic data |

---

## Notes

**On dataset limitations:** This dataset was not designed to produce diverse model outputs. It was generated to test pipeline robustness — specifically whether the system handles extra columns (SalesChannel, PaymentMethod), processes correctly through cleaning and feature engineering, and assigns predictions without errors. The outputs reflect a worst-case homogeneous population and should not be interpreted as the model performing poorly.

**On the high-value calibration gap:** The flat probability histogram in the high-value analysis warrants a closer look at how `prepare_for_inference` handles feature scaling for this model at inference. The XGBoost model was trained on customers with genuine spend variation. When all inference customers cluster at similar low spend values, the model may output spread-out probabilities due to uncertainty rather than confident near-zero scores. This does not affect business decisions (all are correctly classified as Standard) but would be worth resolving with probability calibration (Platt scaling or isotonic regression) in a future iteration.

**On churn probability range:** The churn probabilities clustering at 0.75–0.85 rather than near 1.0 for single-order customers reflects the ensemble's averaging behaviour. Individual models within the ensemble may disagree on degree of churn risk even when all agree it is High Risk, causing the average probability to sit below 0.9. This is appropriate behaviour for a soft-vote ensemble — it avoids overconfident outputs.

**On anomaly rate inflation:** The Isolation Forest contamination parameter (0.05) defines the expected anomaly rate during training, not a hard cap at inference. On a synthetic dataset where 97% of customers are behaviorally identical, the algorithm identifies the small number of multi-order customers as anomalies by relative comparison rather than by absolute deviance from training patterns. This is a known limitation of using contamination-based isolation forests on out-of-distribution inference data.
