# FinGuard ML
### An End-to-End Financial Transaction Risk Scoring and ML Engineering Platform

FinGuard ML is a CPU-first machine learning project for identifying potentially suspicious financial transactions using transaction-level behavioural features. It demonstrates an end-to-end ML engineering workflow, covering data preparation, feature engineering, imbalanced classification, model evaluation, API development and containerised deployment.

The project uses the synthetic PaySim mobile-money transaction dataset and a LightGBM classification model.

> **Important:** PaySim is a synthetic fraud-detection dataset. FinGuard ML is currently a financial transaction fraud-risk scoring prototype, not a validated anti-money laundering (AML) detection system. Its model scores should not be interpreted as calibrated probabilities or regulatory determinations.

## 1. Project Objectives

- Develop a machine learning pipeline for financial transaction risk scoring.
- Engineer transaction features using information available before transaction execution.
- Investigate the effect of class imbalance on model performance.
- Evaluate model performance using metrics appropriate for highly imbalanced classification.
- Expose the trained model through a REST API.
- Package the prediction service using Docker.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | LightGBM, Scikit-learn |
| Data storage | Parquet, PyArrow |
| Model persistence | Joblib, LightGBM |
| API development | FastAPI |
| API server | Uvicorn |
| Containerisation | Docker |
| Visualisation | Matplotlib, Seaborn |

The project was developed and deployed using CPU-based computing resources.

## 3. Dataset

**Dataset:** PaySim Synthetic Mobile Money Transactions

The dataset contains approximately 6.36 million transactions and includes transaction amounts, transaction types, origin and destination account balances, time-step information and fraud labels.

The original fraud distribution is highly imbalanced:

- Legitimate transactions: approximately 99.87%.
- Fraudulent transactions: approximately 0.13%.

The `isFlaggedFraud` field was excluded from model development to avoid relying on an existing fraud flag as a shortcut feature.

## 4. Data Preparation and Feature Engineering

The data preparation workflow includes:

1. Chunk-based processing of the large transaction dataset.
2. Stratified training, validation and test partitioning.
3. Numerical feature scaling using training data only.
4. Categorical encoding of transaction types.
5. Transformation and storage of processed datasets in Parquet format.

The engineered features include:

- Transaction amount.
- Origin account balance before the transaction.
- Destination account balance before the transaction.
- Transaction time step.
- Derived time-of-day proxy.
- Log-transformed transaction amount.
- One-hot encoded transaction type.

Post-transaction balance fields were excluded to maintain a pre-transaction scoring assumption.

## 5. Model Development

LightGBM was selected as the classification algorithm because of its efficiency on structured tabular data.

A controlled class-weight experiment compared unweighted training against several positive-class weighting strategies.

The unweighted model achieved the strongest validation ranking performance among the tested configurations and was selected as the baseline candidate.

## 6. Model Evaluation

The model was evaluated using:

- Precision-Recall AUC (PR-AUC).
- Receiver Operating Characteristic AUC (ROC-AUC).
- Precision at the top 500 ranked transactions.
- Recall at the top 500 ranked transactions.

### Test results

| Metric | Result |
|---|---:|
| Test transactions | 954,393 |
| Fraudulent transactions | 1,232 |
| PR-AUC | 0.5716 |
| ROC-AUC | 0.9154 |
| Precision@500 | 0.8960 |
| Recall@500 | 0.3636 |

The results indicate that the model can rank a substantial proportion of fraudulent transactions near the top of the scoring list. However, the evaluation is based on synthetic data and does not establish real-world operational effectiveness.

### Temporal evaluation

A chronological evaluation was also conducted to examine performance across later transaction periods. Additional analysis identified substantial synthetic-data artefacts in sparse transaction periods. These findings are documented in the evaluation reports and should be considered when interpreting temporal performance.

## 7. API Deployment

FinGuard ML exposes the trained model through a FastAPI REST service.

### Available endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service health check |
| `/predict` | POST | Generate a transaction risk score |
| `/docs` | GET | Interactive API documentation |

### Example prediction request

```json
{
  "amount": 15000,
  "transaction_type": "TRANSFER",
  "origin_balance_before": 25000,
  "destination_balance_before": 5000,
  "step": 100
}
```

### Example response

```json
{
  "transaction_type": "TRANSFER",
  "risk_score": 0.000031,
  "score_interpretation": "Model-generated fraud score; not a calibrated probability or an AML determination."
}
```

The risk score is a model-generated ranking score, not a calibrated probability.

## 8. Docker Deployment

The API has been packaged into a Docker image and successfully tested in a running container.

Build the image:

```bash
docker build -t finguard-ml .
``