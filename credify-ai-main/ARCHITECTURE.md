# CredifyAI — Architecture, Context and Decision Records

**Project:** CredifyAI — Explainable AI-Powered Credit Risk Assessment
**Document status:** Living design document (mid-term snapshot, November 2026)
**Scope of this file:** system context, component architecture, data flow, and the key design
decisions with their rationale and trade-offs.

---

## 1. Context

### 1.1 Problem

Credit scoring in practice is often a binary approve/reject produced by a model that the
applicant, and sometimes even the loan officer, cannot inspect. Two problems follow from
that:

1. **Opacity.** A single score does not say *which* financial behaviours drove the outcome,
   so applicants cannot act on the result and reviewers cannot audit it.
2. **Brittleness.** Traditional scorecards rely on hard thresholds (for example, "reject if
   DTI > 43%"). Real risk is continuous and multi-dimensional; a threshold throws away the
   magnitude of the signal right at the boundary where it matters most.

CredifyAI addresses both: it predicts a **probability** of high risk rather than a verdict,
and it attaches an **explanation** to every prediction.

### 1.2 System context

```
                        ┌──────────────────────────┐
                        │  Applicant / Loan Reviewer │
                        └────────────┬─────────────┘
                                     │  6 financial parameters
                                     ▼
   ┌──────────────────┐   REST/JSON   ┌──────────────────────────┐
   │  Web dashboard   │ ────────────► │  FastAPI prediction      │
   │  (static HTML/   │ ◄──────────── │  service                 │
   │   CSS/JS)        │  probability, │  (api/main.py)           │
   └──────────────────┘  band, reasons└────────────┬─────────────┘
                                                   │
                     ┌─────────────────────────────┼──────────────────────────┐
                     ▼                             ▼                          ▼
          ┌────────────────────┐       ┌────────────────────┐     ┌────────────────────┐
          │ Preprocessing +    │       │ Trained XGBoost    │     │ Rule-based explainer│
          │ feature schema     │       │ binary model       │     │ (reasons, insights, │
          │ (src/preprocess.py,│       │ (models/*.joblib)  │     │  what-if text)      │
          │  src/features.py)  │       └────────────────────┘     │ (src/risk_reasons.py│
          └────────────────────┘                                  └────────────────────┘
```

Outside the code boundary sit three external dependencies: the **LendingClub-style loan
dataset** (offline, not committed to the repository), **Hugging Face Spaces / Docker** for
serving, and **SHAP** for offline interpretability analysis.

### 1.3 Users and their needs

| User | Needs |
|------|-------|
| Applicant | A risk band, plain-language reasons, and a "what could improve my score" view |
| Loan reviewer | Fast, reproducible scoring plus the dominant risk drivers for a manual file |
| Developer | A pipeline where every stage can be verified independently (see §4) |

---

## 2. Component architecture

The repository is deliberately layered so that training and serving never share mutable
state.

```
credify-ai/
├── api/main.py            HTTP layer: Pydantic request validation, CORS, static hosting
├── src/                   library layer (no I/O side effects beyond documented file access)
│   ├── preprocess.py      load CSV → whitelist columns → clean → create risk label
│   ├── features.py        6 interpretable ratio features + target
│   ├── train_model.py     stratified split, class weights, XGBClassifier (multiclass)
│   ├── evaluate.py        classification report + confusion matrix
│   ├── explain.py         SHAP Explainer over model.predict_proba
│   ├── model_io.py        joblib save/load of {model, feature_names}
│   ├── inference.py       schema check → predict_proba → risk band → payload
│   ├── risk_reasons.py    rule-based reasons, pros/cons, what-if simulations
│   └── field_explanations.py  plain-English metadata per input field
├── scripts/               executable phase gates (verify_*.py, save_model.py, test_inference.py)
├── static/                single-page dashboard (index.html, script.js, styles.css)
├── notebooks/             EDA and SHAP exploration (01, 02)
└── visuals/               generated figures (SHAP summary plot)
```

### 2.1 Request/response flow

1. The dashboard collects six values (DTI, credit utilization, EMI ratio, loan-to-income,
   active loan count, delinquency count) and `POST`s them to `/predict`.
2. `PredictionRequest` (Pydantic) rejects missing or mistyped fields with a `422` before any
   model code runs.
3. `predict_risk()` loads the joblib artifact, checks that every feature expected by the
   model is present, and reorders the dictionary to the training column order. Column order
   mismatch is a classic silent failure in tree models, so it is enforced explicitly.
4. `predict_proba(...)[0, 1]` gives the probability of the *high-risk* class.
5. A **policy layer** maps the probability to a band (`< 0.05` low, `< 0.15` watchlist,
   otherwise high) and a decision sentence. The policy sits *outside* the model on purpose:
   risk appetite is a business decision and must be tunable without retraining.
6. `risk_reasons.py` produces the top directional reasons, pros/cons, and what-if
   simulations, and everything is returned as one JSON document.

### 2.2 Data flow

```
raw CSV ──► load_data ──► select_required_columns (schema contract)
        ──► clean_data (drop fatal rows, median impute, parse revol_util %)
        ──► create_target_variable (loan_status → Low / Moderate / High, column dropped)
        ──► create_features (6 ratio features)
        ──► data/processed/credify_features.csv
        ──► stratified 80/20 split (random_state=42)
        ──► class-weighted XGBoost ──► models/credify_high_risk_model.joblib
                                     ──► SHAP summary plot → visuals/
                                     ──► inference.py / API / dashboard
```

### 2.3 Feature set

Six features only, chosen to be ratios rather than raw monetary values so they are
comparable across applicants and easy to explain:

| # | Feature | Source column | Why it is kept |
|---|---------|---------------|----------------|
| 1 | `dti` | `dti` | overall debt stress |
| 2 | `credit_utilization` | `revol_util` | revolving credit already consumed |
| 3 | `emi_to_income` | `installment / (annual_inc/12)` | monthly repayment burden |
| 4 | `loan_to_income` | `loan_amnt / annual_inc` | exposure against earning capacity |
| 5 | `active_loan_count` | `open_acc` | repayment-management load |
| 6 | `delinquency_count` | `delinq_2yrs` | behavioural repayment history |

`loan_status` is dropped immediately after the label is created, to prevent target leakage.

---

## 3. Decision records (ADR style)

### ADR-001 — Six-feature, ratio-based input instead of a wide feature matrix

* **Status:** Accepted
* **Context:** The raw dataset exposes dozens of columns (grades, sub-grades, employment
  length, purpose, address State, etc.). A wider matrix raises accuracy slightly in
  benchmarks but makes explanations long, correlated and hard to defend in a review.
* **Decision:** Keep exactly six inputs, all ratios or counts.
* **Consequences:** Every prediction can be explained on one screen; the API contract is
  small and stable; some predictive signal is knowingly left on the table.
* **Alternatives considered:** Full-column gradient boosting (rejected: poor
  explainability), linear scorecard (rejected: cannot capture the non-linear DTI effect
  seen during EDA).

### ADR-002 — XGBoost as the model family

* **Status:** Accepted
* **Context:** Tabular data, mixed feature scales, non-linear thresholds, and a hard
  requirement to explain individual predictions.
* **Decision:** `XGBClassifier` with `n_estimators=200`, `max_depth=6`,
  `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
* **Consequences:** Strong baseline on tabular data, native missing-value handling,
  SHAP-compatible; depth 6 limits overfitting on 6 features.
* **Alternatives considered:** Logistic regression (baseline kept in mind, but weaker on
  interactions), random forest (noisier probabilities), neural networks (unjustified
  complexity and inference cost for six inputs).

### ADR-003 — Two training formulations: multiclass for analysis, binary High-vs-Rest for serving

* **Status:** Accepted
* **Context:** The label is three-valued (Low / Moderate / High), but the operational
  question the product answers is *"what is the probability this applicant becomes high
  risk?"* A three-way probability is harder to threshold and harder to explain.
* **Decision:**
  * `src/train_model.py` trains the three-class model for analysis and evaluation.
  * `scripts/save_model.py` derives a binary target (`High` vs rest) and persists that
    model as `models/credify_high_risk_model.joblib`, which is what `inference.py` loads.
  * SHAP is run against the binary model (`scripts/verify_explainability.py`).
* **Consequences:** One clear number (`probability_high_risk`) drives the whole product;
  class imbalance is handled once, with balanced class weights / an explicit
  `scale_pos_weight`.
* **Trade-off:** Moderate-risk detail is currently expressed only through the watchlist
  band. A calibrated three-class UI is deferred to a later phase.

### ADR-004 — Probability output with a separate, tunable policy layer

* **Status:** Accepted
* **Context:** Banks change cut-offs frequently; embedding 0.05/0.15 inside the model would
  force retraining for a policy change.
* **Decision:** The model emits a probability; `inference.py` maps it to bands, and the
  dashboard only renders what it receives.
* **Consequences:** Policy review becomes a configuration change; the same artifact can
  serve retail and SME risk appetites.
* **Note:** The thresholds are initial values chosen from the observed probability
  distribution, not yet validated against a business cost matrix.

### ADR-005 — Rule-based explanations next to SHAP, not instead of it

* **Status:** Accepted
* **Context:** Raw SHAP values are numeric and awkward to show to a non-technical user.
  Pure rule systems, on the other hand, cannot claim to describe *this model*.
* **Decision:** SHAP supplies the offline, global feature-importance evidence
  (`visuals/shap_summary_high_risk.png`); `risk_reasons.py` supplies short, deterministic,
  human-readable strings for the online response.
* **Consequences:** The UI always has text to render, fast and offline; the risk of
  divergence between rules and model exists.
* **Mitigation (planned):** replace the fixed thresholds in `risk_reasons.py` with
  per-request SHAP contributions in the next phase.

### ADR-006 — FastAPI + static frontend, no client-side framework

* **Status:** Accepted
* **Context:** One form, one endpoint, one result page. React would add a build step,
  dependency churn and deploy complexity for no functional gain at this stage.
* **Decision:** FastAPI serves both `/predict` and the static dashboard from the same
  process; the front end is plain HTML/CSS/JS.
* **Consequences:** Single Docker image, single port (7860), trivial local setup
  (`uvicorn api.main:app --port 8000`); the trade-off is manual DOM code in `script.js`.

### ADR-007 — Model persistence as a joblib dictionary `{model, feature_names}`

* **Status:** Accepted
* **Context:** A bare estimator does not remember the exact column order it was trained on.
* **Decision:** `model_io.py` stores the estimator together with the ordered feature list,
  and `inference.py` refuses to score an input that is missing any expected feature.
* **Consequences:** Schema drift between training and serving fails loudly with a
  `ValueError` instead of producing a wrong number.
* **Limitation (known):** there is no model version field in the artifact yet; versioning
  is currently handled by file name / external model registry only.

### ADR-008 — Verification scripts as phase gates

* **Status:** Accepted
* **Context:** A capstone project moves through phases; a broken preprocessing step must
  not be discovered only when the API fails.
* **Decision:** Each phase has an independent entry point —
  `verify_preprocessing.py`, `verify_features.py`, `verify_train_model.py`,
  `verify_explainability.py`, `save_model.py`, `test_inference.py`.
* **Consequences:** Progress is demonstrable stage by stage, and regressions are localized.
* **Consequence for the mid-term report:** these scripts, plus the saved SHAP figure, are
  the evidence of completed work; the notebooks and metric re-runs are still open.

### ADR-009 — Container-first deployment

* **Status:** Accepted
* **Context:** "Works on my machine" is the most common failure mode for ML services; the
  hosting target (Hugging Face Spaces) expects a Dockerfile.
* **Decision:** `python:3.9-slim`, install `requirements.txt`, expose 7860, run uvicorn with
  `api.main:app`.
* **Consequences:** Identical local and cloud behaviour; image size is small enough for
  free-tier hosts.
* **Open issue:** the frontend currently points at a hard-coded absolute API URL in
  `static/script.js`; a relative path or environment-driven base URL is planned.

---

## 4. Cross-cutting concerns

| Concern | Approach | Status |
|---|---|---|
| Input validation | Pydantic `PredictionRequest`, types and required fields | Done |
| Schema contract | `REQUIRED_COLUMNS` whitelist; artifact feature list | Done |
| Target leakage | `loan_status` dropped right after labelling | Done |
| Class imbalance | `compute_class_weight(class_weight="balanced")` → per-sample weights | Done |
| Reproducibility | `random_state=42` for split, model and sampling | Done |
| Explainability | SHAP global summary; rule-based local reasons | Partial (local SHAP pending) |
| Error handling | HTTP 500 with traceback logging in `api/main.py` | Done (message sanitisation pending) |
| CORS | wildcard origins for development | Needs tightening before public launch |
| Monitoring / drift | — | Not started |
| Tests / CI | manual verification scripts only | Not started |

---

## 5. Known gaps at mid-term

1. **Data and model artifacts are not committed** (`data/`, `models/` are gitignored), so a
   fresh clone cannot run the verification scripts without re-downloading the dataset.
2. **`notebooks/01_eda.ipynb` and `02_shap_analysis.ipynb` are empty placeholders** — the
   analysis currently lives in scripts and needs to be written back up.
3. **Quantitative metrics are not reproducible from the repository alone**; the numbers
   recorded during earlier runs must be regenerated and cross-checked before the final
   report.
4. **Explanations are rule-based at inference time**; SHAP is offline only.
5. **No automated tests, no CI, no drift monitoring.**
6. **Hard-coded API base URL** in the front end.

These gaps define the remaining work plan for the second half of the capstone.
