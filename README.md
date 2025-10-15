# Federated AI Sentinel

Early detection of hospital-acquired infections using privacy-preserving federated learning.

## What It Does

This system predicts infection risk across multiple hospitals without sharing raw patient data. Each hospital trains locally. Only model updates travel to a central server. The system explains every alert, monitors for drift, and retrains safely.

Built for clinicians and engineers who need production-grade AI with full transparency.

## How It Protects Privacy

1. **No raw data leaves your hospital.** Only encrypted model weights are shared.
2. **Optional differential privacy** adds calibrated noise to weight updates.
3. **Federated learning** trains a global model while keeping patient records local.
4. **Audit logs** track every prediction request without storing PHI.

## How to Run Locally

You need Docker, Make, and Git. Everything runs on a laptop without cloud spend.

```bash
# Clone and setup
git clone https://github.com/saminathan017/Federated-AI-Sentinel-Early-Detection-of-Hospital-Acquired-Infections.git
cd federated_ai_sentinel
make setup

# Start infrastructure
make run_kafka          # Terminal 1: Kafka for event streaming
make run_prom_grafana   # Terminal 2: Monitoring stack

# Train models
make train_baseline     # XGBoost baseline
make train_temporal     # Temporal transformer
make federated_sim      # Simulate three hospital sites

# Serve and monitor
make serve_api          # Terminal 3: FastAPI inference server
make serve_ui           # Terminal 4: Clinician web app

# Open in browser
# API docs: http://localhost:8000/docs
# UI: http://localhost:5173
# Grafana: http://localhost:3000 (admin/admin)
```

## Reading the Dashboards

**Grafana Sentinel Overview**
- Request rate and latency: API health check
- Alert volume: How many high-risk cases per hour
- Model calibration: Are predicted probabilities accurate?
- Drift score: Is the patient population changing?

**Clinician UI**
- Risk cards show individual patient scores with uncertainty bars
- Heatmap shows risk by ward and time
- Explanation panel lists top clinical drivers and what-if scenarios
- Trend chart tracks infection rates over time

## Quickstart Commands

```bash
make setup              # Install dependencies and set up environment
make lint               # Run Ruff and Black formatters
make test               # Run PyTest suite
make train_baseline     # Train XGBoost baseline model
make train_temporal     # Train temporal transformer
make eval               # Evaluate model performance
make explain            # Generate SHAP explanations
make serve_api          # Start FastAPI server
make serve_ui           # Start React development server
make federated_sim      # Run three-site federated simulation
make autopush           # Watch for changes and auto-commit to GitHub
```

## Tech Stack

**Backend:** Python, PyTorch, scikit-learn, XGBoost, Flower (federated learning), FastAPI, Uvicorn

**Data:** Kafka/Redpanda, Feast (feature store), PostgreSQL, MinIO/local storage

**ML Ops:** MLflow, SHAP, Alibi Detect, Prometheus, Grafana

**Frontend:** React, Vite, TypeScript, Tailwind CSS

**Testing:** PyTest, Playwright, Ruff, Black, Mypy

**Docs:** MkDocs Material

## Repository Structure

```
federated_ai_sentinel/
├── src/
│   ├── data/               # Synthetic data generation
│   ├── ingestion/          # Kafka producers and consumers
│   ├── features/           # Windowing, labeling, Feast store
│   ├── modeling/           # Baselines, temporal models, training
│   ├── federated/          # Client nodes, server aggregator
│   ├── explainability/     # SHAP and counterfactuals
│   ├── drift/              # Drift detection and retrain triggers
│   ├── serving/            # FastAPI inference service
│   ├── monitoring/         # Prometheus and Grafana configs
│   └── ui/                 # React clinician dashboard
├── tests/                  # Unit and e2e tests
├── tools/                  # Autopush and seeding utilities
├── docs/                   # MkDocs documentation
└── devcontainer/           # Development container setup
```

## Responsible AI

This system is designed to augment clinical judgment, not replace it. All predictions include:
- Calibrated uncertainty estimates
- SHAP-based explanations of top clinical drivers
- Counterfactual what-if scenarios
- Fairness metrics across demographic groups
- Drift monitoring to detect distribution shifts

We compute equal opportunity gaps and calibration per group. We flag when model performance degrades on subpopulations.

## Clinical Safety Notice

**This is a research prototype.** It is not FDA-cleared or CE-marked. Do not use for clinical decision-making without:
1. Validation on your hospital's data
2. Review by infection control and informatics teams
3. Integration into your EHR and clinical workflows
4. Continuous monitoring and human oversight

Always confirm alerts with clinical assessment and diagnostic testing.

## License

MIT License. See LICENSE file for details.

## Contributing

Issues and pull requests welcome. Follow the coding standards in pyproject.toml. Run `make lint` and `make test` before submitting.

## Contact

For questions about clinical deployment, see docs/clinical_protocol.md.

For security concerns, see docs/threat_model.md.

