# System Architecture

## Overview

The Federated AI Sentinel is built as a modular system with clear separation of concerns.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Hospital Site A                          │
│  ┌────────────┐  ┌──────────┐  ┌─────────────────┐         │
│  │   EHR      │→ │ Features │→ │ Local Training  │         │
│  │   Data     │  │ Pipeline │  │ Client (Flower) │         │
│  └────────────┘  └──────────┘  └────────┬────────┘         │
└──────────────────────────────────────────┼──────────────────┘
                                           │ Model Updates Only
                        ┌──────────────────┼──────────────────┐
                        │   Central Server                     │
                        │  ┌───────────────▼─────────────┐    │
                        │  │ Federated Aggregator         │    │
                        │  │ (FedAvg + DP Noise)          │    │
                        │  └───────────────┬─────────────┘    │
                        │                  │                   │
                        │  ┌───────────────▼─────────────┐    │
                        │  │ Global Model Storage         │    │
                        │  └──────────────────────────────┘    │
                        └─────────────────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────┐
                        │   Inference Service                  │
                        │  ┌──────────────────────────────┐   │
                        │  │ FastAPI                      │   │
                        │  │  • /score                    │   │
                        │  │  • /explain                  │   │
                        │  │  • /counterfactual           │   │
                        │  └──────────────────────────────┘   │
                        │  ┌──────────────────────────────┐   │
                        │  │ Predictor + Calibrator +     │   │
                        │  │ Uncertainty + Explainability │   │
                        │  └──────────────────────────────┘   │
                        └─────────────────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────┐
                        │   Clinical Dashboard (React)         │
                        └─────────────────────────────────────┘
```

## Data Flow

1. **Data Generation**: Synthetic FHIR-like data for development, real EHR data in production
2. **Feature Engineering**: Sliding time windows with aggregated vitals and labs
3. **Labeling**: Hospital-acquired infection labels based on culture results
4. **Local Training**: Each site trains on its own data
5. **Federated Aggregation**: Model updates aggregated centrally
6. **Model Serving**: Global model serves predictions via API
7. **Explainability**: SHAP and counterfactuals for every prediction
8. **Monitoring**: Drift detection triggers retraining when needed

## Technology Stack

**Backend**
- Python 3.11
- PyTorch for deep learning
- Flower for federated learning
- FastAPI for serving
- MLflow for experiment tracking

**Data**
- Kafka/Redpanda for streaming
- Feast for feature store
- PostgreSQL for metadata
- Parquet for efficient storage

**Frontend**
- React 18
- TypeScript
- Vite for build
- Tailwind CSS
- Recharts for visualization

**Monitoring**
- Prometheus for metrics
- Grafana for dashboards

**Testing**
- PyTest for Python
- Playwright for UI

## Security Architecture

See [Threat Model](threat_model.md) for detailed security design.

