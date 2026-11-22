# FinGuard ML
### An End-to-End Financial Transaction Risk Scoring and ML Engineering Platform

FinGuard ML is a CPU-first machine learning engineering project for identifying potentially suspicious financial transactions through transaction-level behavioural features. It demonstrates an end-to-end workflow covering data preparation, feature engineering, imbalanced classification, model evaluation, REST API development, Docker containerisation and Kubernetes orchestration.

The project uses the synthetic PaySim mobile-money transaction dataset and a LightGBM classification model. The trained model is exposed through a FastAPI service and deployed locally using Docker and Kubernetes.

> **Important:** PaySim is a synthetic fraud-detection dataset. FinGuard ML is a financial transaction fraud-risk scoring prototype, not a validated anti-money laundering (AML) detection system. Its risk scores are model-generated ranking scores, not calibrated probabilities or regulatory determinations.

## 1. Project Objectives

- Develop an end-to-end machine learning pipeline for financial transaction risk scoring.
- Engineer transaction features using information available before transaction execution.
- Investigate the effect of class imbalance on predictive performance.
- Evaluate model performance using metrics appropriate for highly imbalanced classification.
- Expose the trained model through a REST API.
- Containerise the prediction service using Docker.
- Deploy and manage the application using Kubernetes.
- Demonstrate service discovery, pod recovery, horizontal scaling, load testing, rolling updates and deployment rollback.

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
| Container orchestration | Kubernetes |
| Local Kubernetes environment | Docker Desktop Kubernetes |
| Autoscaling metrics | Kubernetes Metrics Server |
| Load testing | Python, Requests, ThreadPoolExecutor |
| Visualisation | Matplotlib, Seaborn |

The project was developed and tested using CPU-based computing resources.

## 3. Dataset

**Dataset:** PaySim Synthetic Mobile Money Transactions

PaySim is a synthetic mobile-money transaction dataset containing approximately 6.36 million records. It includes transaction amounts, transaction types, origin and destination account balances, time-step information and fraud labels.

The original fraud distribution is highly imbalanced:

- Legitimate transactions: approximately 99.87%.
- Fraudulent transactions: approximately 0.13%.

The `isFlaggedFraud` field was excluded from model development to avoid using an existing fraud flag as a shortcut feature.

## 4. Data Preparation and Feature Engineering

The data preparation workflow includes:

1. Chunk-based processing of the large transaction dataset.
2. Training, validation and test partitioning.
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
- Encoded transaction type.

Post-transaction balance fields were excluded to maintain a pre-transaction scoring assumption.

## 5. Model Development

LightGBM was selected as the classification algorithm because of its computational efficiency and suitability for structured tabular data.

A controlled class-weight experiment compared unweighted training against several positive-class weighting strategies.

The unweighted model achieved the strongest validation ranking performance among the tested configurations and was selected as the baseline candidate.

## 6. Model Evaluation

The model was evaluated using metrics suited to highly imbalanced classification:

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

The results indicate that the model ranks a substantial proportion of fraudulent transactions near the top of the scoring list. In particular, Precision@500 indicates that 89.6% of the 500 highest-ranked test transactions were labelled fraudulent in this synthetic evaluation.

These results do not establish real-world operational effectiveness.

### Temporal evaluation

A chronological evaluation was also conducted to examine model performance across later transaction periods. Additional analysis identified substantial synthetic-data artefacts in sparse transaction periods. These findings are documented in the evaluation reports and should be considered when interpreting temporal performance.

## 7. API Development

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

The score is a model-generated output. It should not be interpreted as a calibrated probability of fraud.

## 8. Docker Containerisation

The prediction API is packaged as a Docker image, allowing the application and its dependencies to run in a consistent containerised environment.

Build the application image:

```powershell
docker build -t finguard-ml:v1 .
```

Run the container:

```powershell
docker run -p 8000:8000 finguard-ml:v1
```

Access the interactive API documentation:

```text
http://localhost:8000/docs
```

The project also uses a second image tag, `finguard-ml:v2`, to demonstrate Kubernetes rolling updates and rollback between application releases. The v2 image was built from the same application code for deployment-version testing; it does not represent a new model or improved predictive performance.

## 9. Kubernetes Orchestration

The API was deployed to a local Kubernetes cluster provided by Docker Desktop. Kubernetes manages the application pods, exposes the service internally, monitors resource consumption and supports horizontal scaling.

### 9.1 Deployment

The Kubernetes Deployment manages the lifecycle of the FinGuard API pods.

The deployment configuration specifies:

- Container image: `finguard-ml:v1` as the manifest baseline.
- Container port: `8000`.
- CPU request: `250m`.
- CPU limit: `1`.
- Memory request: `256Mi`.
- Memory limit: `1Gi`.

The Deployment maintains the requested number of application replicas and recreates pods when they are deleted or become unavailable.

Apply the Deployment:

```powershell
kubectl apply -f deployment.yaml
```

Check its status:

```powershell
kubectl get deployments
kubectl get pods -l app=finguard-api
```

### 9.2 Kubernetes Service

A Kubernetes NodePort Service named `finguard-service` exposes the API on port `8000` inside the cluster and uses NodePort `30080`.

The Service selects application pods using the label:

```text
app: finguard-api
```

This allows requests to be routed to available application endpoints without clients needing to know individual pod IP addresses.

Apply the Service:

```powershell
kubectl apply -f service.yaml
```

Inspect the Service and its endpoints:

```powershell
kubectl get service finguard-service
kubectl get endpointslices -l kubernetes.io/service-name=finguard-service
```

For local access through Docker Desktop, port-forwarding provides a convenient route from the Windows host:

```powershell
kubectl port-forward service/finguard-service 8001:8000
```

The API can then be accessed at:

```text
http://localhost:8001/docs
```

Keep the port-forwarding process running while accessing the API.

### 9.3 Resource Requests and Limits

Resource requests and limits were configured to define the CPU and memory resources available to each API container.

| Resource | Request | Limit |
|---|---:|---:|
| CPU | 250m | 1 CPU |
| Memory | 256Mi | 1Gi |

These settings provide Kubernetes with resource scheduling information and establish container resource boundaries.

### 9.4 Horizontal Pod Autoscaling

A Horizontal Pod Autoscaler (HPA) was configured to scale the FinGuard API Deployment based on CPU utilisation.

The autoscaling configuration specifies:

| Parameter | Configuration |
|---|---:|
| Minimum replicas | 2 |
| Maximum replicas | 5 |
| CPU utilisation target | 60% |
| Scaling target | `finguard-api` Deployment |

Apply the autoscaler:

```powershell
kubectl apply -f hpa.yaml
```

Inspect autoscaling activity:

```powershell
kubectl get hpa
kubectl describe hpa finguard-api-hpa
```

The HPA was observed scaling the application from two replicas to as many as five during load testing. Following a reduction in workload, the replica count returned to the configured minimum.

### 9.5 Load Testing and Autoscaling Verification

A Python-based load-testing script was developed to submit concurrent prediction requests to the API.

The demonstrated test configuration used:

- Total requests: 500.
- Concurrent workers: 10.
- API endpoint: `/predict`.
- Successful HTTP responses: 500.
- Failed requests: 0.
- Observed completion time: approximately 16.16 seconds.

The load test was used to generate application traffic while monitoring CPU utilisation and HPA scaling behaviour.

The test demonstrates successful request handling under the specified local workload. It is not a comprehensive production performance or capacity benchmark.

Run the test while the API port-forward is active:

```powershell
python load_test.py
```

Monitor the application during testing:

```powershell
kubectl get hpa -w
```

In another terminal:

```powershell
kubectl get pods -l app=finguard-api -w
```

### 9.6 Service Discovery and Traffic Distribution

The Kubernetes Service was tested from inside the cluster using a temporary BusyBox pod.

The test confirmed that the Service DNS name:

```text
finguard-service
```

could resolve and route requests to the API pods.

Thirty requests were sent through the Service. Application logs showed that both available API pods received requests, demonstrating Service-level endpoint routing.

### 9.7 Pod Recovery

Pod recovery was tested by manually deleting an application pod.

The Deployment controller detected the reduced number of replicas and created a replacement pod to restore the desired state.

This demonstrates Kubernetes self-healing at the pod level. The experiment was conducted on a single-node local cluster and does not demonstrate multi-node high availability or node-failure recovery.

### 9.8 Rolling Updates and Rollback

Kubernetes rolling updates were tested by building a second application image:

```powershell
docker build -t finguard-ml:v2 .
```

The Deployment image was then updated:

```powershell
kubectl set image deployment/finguard-api finguard-api=finguard-ml:v2
```

Monitor the rollout:

```powershell
kubectl rollout status deployment/finguard-api
```

Inspect Deployment history:

```powershell
kubectl rollout history deployment/finguard-api
```

The application was subsequently rolled back to the previously recorded revision:

```powershell
kubectl rollout undo deployment/finguard-api --to-revision=2
```

The rollback completed successfully, and the Deployment image was verified as:

```text
finguard-ml:v1
```

This demonstrates Kubernetes support for controlled application updates and restoration of an earlier Deployment configuration.

## 10. Project Structure

The repository contains the machine learning application, supporting data and evaluation workflows, API implementation, container configuration and Kubernetes deployment resources.

Key Kubernetes-related files include:

```text
FinGuard-ML/
│
├── deployment.yaml
├── service.yaml
├── hpa.yaml
├── load_test.py
├── Dockerfile
├── README.md
│
├── data/
├── models/
├── reports/
└── src/
```

The directory listing above highlights the principal project components; consult the repository for the complete structure.

## 11. Reproducing the Kubernetes Deployment

### Prerequisites

- Python 3.10 or a compatible project-supported version.
- Docker Desktop.
- Kubernetes enabled in Docker Desktop.
- `kubectl` configured to use the `docker-desktop` context.
- The trained model and application files required by the Docker build.

Verify the active Kubernetes context:

```powershell
kubectl config current-context
```

Build the image:

```powershell
docker build -t finguard-ml:v1 .
```

Apply the Kubernetes resources:

```powershell
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
```

Check the deployment:

```powershell
kubectl get deployments
kubectl get pods -l app=finguard-api
kubectl get services
kubectl get hpa
```

Access the API:

```powershell
kubectl port-forward service/finguard-service 8001:8000
```

Open:

```text
http://localhost:8001/docs
```

**Note:** The local Docker Desktop Kubernetes environment uses locally available images. The `IfNotPresent` image pull policy allows Kubernetes to use the locally built image. A remote cluster would require an accessible image registry and appropriately configured image references.

## 12. Limitations and Future Work

Although the project demonstrates an end-to-end machine learning engineering workflow, several areas remain open for further development.

- **Synthetic data:** PaySim does not fully represent the complexity of real financial transaction behaviour.
- **Model calibration:** The risk score has not been established as a calibrated probability.
- **Temporal robustness:** Synthetic-data artefacts in sparse periods require further investigation.
- **Monitoring:** Continuous model performance monitoring, data drift detection and model drift monitoring have not yet been implemented.
- **Security:** Production-grade authentication, authorisation, secrets management and API rate limiting remain future improvements.
- **Deployment infrastructure:** Kubernetes experiments were conducted on a single-node Docker Desktop cluster, not a production multi-node environment.
- **Scalability evaluation:** The load test demonstrates behaviour under a specific local workload; larger-scale benchmarking remains necessary.
- **Model lifecycle:** Automated model retraining, model registry integration and CI/CD pipelines remain potential extensions.

Future work may investigate model calibration, explainable transaction scoring, drift monitoring, model version management and deployment to a managed multi-node Kubernetes environment.

## 13. Conclusion

FinGuard ML demonstrates how a machine learning model can be developed, evaluated, exposed through a REST API and managed as a containerised application using Kubernetes.

Beyond model development, the project demonstrates practical ML engineering capabilities, including resource configuration, service discovery, pod recovery, horizontal autoscaling, concurrent request testing, rolling updates and rollback.

The resulting platform provides a reproducible foundation for further work on reliable machine learning services for financial transaction risk scoring.

## 14. Author and Repository

**Author:** Deborah Olaniyan

**GitHub:** [FinGuard ML](https://github.com/deboraholaniyan-debug/FinGuard-ML)

**Project focus:** Machine Learning Engineering, Financial Transaction Risk Scoring, FastAPI, Docker and Kubernetes.
