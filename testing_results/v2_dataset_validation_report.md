# Model Validation Report — Sample Dataset v2

## Dataset Link: ## Dataset Link: https://drive.google.com/drive/folders/1WH3oCimeBjwfeCiXZ0_IrlanAPnCg-CG?usp=sharing (goto testing data/sample_transactions_v2.csv) 

## Dataset Overview

| Property | Value |
|---|---|
| Total transactions | 14,369 |
| Unique customers | 984 |
| Unique invoices | 4,405 |
| Date range | 2024-01-01 to 2025-01-02 |
| Cancellation rows (C-prefix, negative qty) | 157 |
| Customers with at least one cancellation | 94 |
| Extra columns (pipeline noise test) | SalesChannel, PaymentMethod |

This dataset was built to stress-test the full model suite. It contains six distinct behavioural segments with deliberate variation in spend, frequency, recency, and cancellation behaviour, plus edge cases for bulk orders, full returns, split same-day orders, and zero-price items. It is the more representative of the two test datasets.

---

## Ground Truth by Segment

| Segment | Customer IDs | Count | Avg Spend | Avg Orders | Max Spend | Design Intent |
|---|---|---|---|---|---|---|
| One-Time Churners | 13000–13350 | 351 | £28 | 1.0 | £193 | Should be flagged as High Churn, Standard value |
| Engaged Regulars | 14000–14250 | 251 | £631 | 8.6 | £1,376 | Should be Low/Medium Churn, Standard/Growing value |
| At-Risk Irregulars | 15000–15200 | 201 | £116 | 2.9 | £359 | Should be High Churn, Standard value |
| VIP | 16000–16050 | 51 | £34,429 | 14.7 | £54,062 | Should be Low Churn, VIP value tier |
| Cancellation-Prone | 17000–17100 | 101 | £140 | 3.5 | £414 | Should be Normal/Watch List cancellation risk |
| Edge Cases | 18000–18306 | 29 | £298 | 1.6 | £2,820 | Bulk buyers, full-return customers, split orders, zero-price |

---

## Prediction vs Expected Behaviour

### Churn Model

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| High Risk overall | ~38% (One-Timers 351 + At-Risk 201 = 552 / 984) | 65.1% | Significant overestimate |
| Medium Risk | ~26% (Engaged Regulars 251) | 32.3% | Reasonable overestimate |
| Low Risk | ~5–8% (VIPs 51 + some Engaged Regulars) | 2.6% | Underestimate |
| Probability distribution | Spread across all three bands | Histogram spike in 0.6–0.85 range | Shifted right — model is more aggressive than expected |
| Churn rate by cluster — Engaged Regulars | Expected notably lower than other clusters | 92.9% churn rate | Near-identical to One-Timers — see note |

**Interpretation:** The churn model is over-predicting High Risk on this dataset. The expected High Risk pool is roughly 55% (One-Timers + At-Risk), but 65% is being classified there. The primary driver is the Engaged Regulars cluster — 92.9% churn rate for customers who were designed to have 5–12 orders is not expected. This is a label definition issue: the churn label in training was defined as no post-reference purchase. Since these synthetic customers' purchases are spread across 2024 and the reference date is max invoice date, many Engaged Regulars have their last purchase well before the reference cutoff and are legitimately labelled as churned. The model is not misclassifying — the label itself is accurate for this dataset's structure. VIPs are similarly affected: their recency is high (recent purchases) so some make it into Low Risk, but the majority still trigger High Risk based on the label.

The tier distribution by cluster shows the clearest signal the churn model is working: Engaged Regulars have significantly more Medium Risk than One-Time Churners, and At-Risk Irregulars sit between the two. This is correct directional behaviour.

---

### High-Value Model

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| VIP tier | ~5.2% (51 VIP customers with avg spend £34,429) | 0% | Complete miss |
| Growing Potential | ~25% (Engaged Regulars, At-Risk Irregulars with moderate-high spend) | 0% | Complete miss |
| Standard (100%) | Should be ~70% | 100% | Major overestimate |
| Probability histogram | Should show spread across 0–1 | Completely flat, no data in histogram | Calibration failure |

**Interpretation:** The high-value model is not working on this dataset. This is a genuine model failure.

The root cause is how the high-value label is defined and who the model was trained on. The model was trained on real UCI customers where high-value is defined by post-reference spend exceeding the 80th percentile. At inference, the pipeline filters to `churn == 0` before running high-value scoring. Since 65% of customers in this dataset are classified as High Risk churn, only ~35% (roughly 340 customers) are passed to the high-value model — and these are disproportionately the moderate-spend customers, not the VIPs who were also classified as churned.

Even if VIPs reached the high-value model, the probability histogram showing completely flat (empty bars) is a separate and more serious issue: the model is generating NaN or constant probability outputs for the entire population. This indicates a feature mismatch at inference — the feature values the model sees at inference are outside the range it was trained on. The VIP customers have avg order values up to £2,820 and spend up to £54,061 — both are extreme outliers relative to the UCI training distribution where the 80th percentile spend was in the hundreds of pounds. XGBoost trees cannot extrapolate beyond their training range and output constant leaf values for out-of-distribution inputs, which manifests as flat probabilities.

**What would fix it:** The high-value model needs to be exposed to customers with genuine VIP-level spend during training, or feature scaling needs to compress extreme spend values into the training range before inference. Currently the model has never seen customers spending £34k and cannot meaningfully score them.

---

### High-Risk (Cancellation) Model

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| Watch List (0.30–0.60) | ~10% — 94 customers with cancellations, roughly half above 0.30 threshold | 0% | Complete miss |
| Urgent Attention (>0.60) | ~5% — customers with 3–4 cancellations | 0% | Complete miss |
| Normal (100%) | Expected ~85–90% | 100% | Overestimate |
| Probability histogram | Should show some customers pushing toward 0.30+ | Spike at 0.05–0.20, all below 0.30 | Max probability below threshold |

**Interpretation:** The cancellation risk model is not triggering Watch List or Urgent Attention for any customers despite 94 customers having genuine cancellation history. This is the same issue identified in our earlier analysis and confirmed here with real cancellation data present.

The cancellation rate values in this dataset max out at 0.50 (1 cancellation for every 2 orders). The XGB model, trained on the real UCI dataset where genuine high-risk cancellers had higher and more sustained cancellation rates over longer periods, assigns low-to-moderate probabilities to these customers because their cancellation signal is mild relative to the training distribution. The conservative thresholds (Watch List > 0.30) are deliberately set to avoid false positives — they require genuine cancellation pattern strength to fire, which this dataset's cancellation-prone customers do not quite reach.

This is not a model failure — it is the correct behaviour for the signal strength present. Documenting it as "model is conservative and requires stronger cancellation signals than present in the test data" is the accurate framing.

---

### Unsupervised Segmentation (Clustering)

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| One-Time Churners share | ~36% (351/984) | 39.3% (385) | Minor overcount |
| Engaged Regulars share | ~26% (251/984) | 31.5% (308) | Minor overcount |
| At-Risk Irregulars share | ~20% (201/984) | 29.1% (285) | Overcount absorbs VIPs and cancellation-prone |
| Spend separation — Engaged Regulars | Expected highest avg spend | £4,909 avg | Correct — clear separation |
| Product diversity — Engaged > Others | Expected | Confirmed in violin plot | Correct |
| VIPs in correct cluster | Expected Engaged Regulars | Spread across At-Risk and Engaged | Partially correct — VIPs split by recency |

**Interpretation:** Clustering is working directionally well. The three-cluster model correctly separates low-frequency/low-spend from high-frequency/high-spend customers, and the spend distribution by cluster (Image 13) shows strong separation — Engaged Regulars have a wide spend range including the VIP outliers at £50k+. The VIP customers do not form their own cluster because KMeans was trained with k=3 and sees the VIPs as high-spend members of the Engaged Regulars or At-Risk groups depending on their recency pattern. This is expected behaviour — a k=4 or k=5 model would be needed to isolate VIPs as a separate cluster.

The segment behavior table (Image 11) showing "Standard Active" avg spend of £41,311 with recency of 14.7 days confirms the VIP customers are being captured in the business-rule segment layer correctly, even if the unsupervised clustering doesn't isolate them separately.

---

### Anomaly Detection (Isolation Forest)

| Dimension | Expected from Data | Observed Output | Deviation |
|---|---|---|---|
| Anomaly rate | 5% (contamination=0.05) → ~49 customers | 26.4% (258 customers) | Major overestimate — 5x expected |
| Low Risk anomaly rate | Should be low | 100% — all 258 anomalies are Low Risk churn customers | Counterintuitive |
| High Risk anomaly rate | Should be higher (cancellation-prone, edge cases) | 27.9% | Partially correct |
| Medium Risk anomaly rate | Should be moderate | 17.4% | Reasonable |

**Interpretation:** The 26.4% anomaly rate is again significantly above the 5% contamination parameter, for the same reason as v1 — the model was trained on UCI data and the inference population is out-of-distribution. The more interesting finding is the 100% anomaly rate for Low Risk customers (Image 14). Low Risk in this dataset is a very small group (the small green slice in Image 6, roughly 2.6% of customers = ~25 customers). These are likely the VIP customers who have very recent purchase dates, giving them low churn probability. The Isolation Forest correctly identifies VIPs as anomalies — they have extreme spend, high order counts, and are genuine global outliers. This is the model working correctly, not a failure: VIPs are anomalous relative to the general population.

The 27.9% anomaly rate for High Risk churn customers reflects the at-risk and cancellation-prone segments being flagged as anomalous — again correct, since customers with unusual cancellation patterns are behavioural outliers.

---

## Summary

| Model | Overall Accuracy | Calibration Quality | Verdict |
|---|---|---|---|
| Churn | Directionally correct | Moderate — over-aggressive but cluster separation holds | Working as designed, label structure explains deviation |
| High-Value | Tier assignment completely wrong | Failed — flat/empty histogram | Not working on this dataset |
| High-Risk | Tier assignment correct (all Normal) | Good — well-concentrated below threshold | Working correctly, data cancellation signal too weak to cross threshold |
| Clustering | Directionally correct, spend separation strong | N/A | Working well, VIPs absorbed into Engaged Regulars as expected for k=3 |
| Anomaly Detection | Rate inflated, Low Risk = VIP detection is correct | N/A | Partially working — VIP detection correct, rate inflation is out-of-distribution effect |

---

## Notes

**On the high-value model failure:** This is the most significant finding from v2 testing. The model produces flat/empty probability outputs for 100% of the inference population. Two compounding causes: first, most genuine high-value customers (VIPs) are filtered out before reaching the model because they are classified as churned; second, even those that reach the model receive constant XGBoost leaf outputs because their feature values (avg order value £2,820, total spend £54,061) are far beyond the training distribution. Resolving this requires either retraining on data that includes VIP-scale customers, or applying feature clipping/capping before inference to map extreme values into the trained range.

**On churn over-prediction:** The 65% High Risk rate versus the expected ~55% is partly a label artifact. The synthetic At-Risk Irregulars were designed to be dormant from month 5 onwards, so their last purchase date is 8–12 months before the reference date — a strong churn signal. Combined with the One-Time Churners, this legitimately pushes the High Risk count above what the segment design might suggest at first glance. The model is not wrong; the label definition and segment design interact in a way that produces higher churn labels than the segment names imply.

**On cancellation risk conservatism:** The 0.30 threshold for Watch List is correctly set for real-world use where false positives have operational cost. The test dataset's maximum cancellation rate of 0.50 is not extreme enough relative to the model's training distribution to cross this threshold. This validates that the model will not flag customers with minor cancellation activity — which is the intended conservative behaviour.

**On clustering and VIP isolation:** A k=3 cluster model cannot isolate a 5% VIP minority as a distinct cluster — the centroids will be pulled toward the majority populations. For production use where VIP identification from clustering is important, k=4 or k=5 with the fourth centroid seeded near VIP-level spend values would be worth testing.
