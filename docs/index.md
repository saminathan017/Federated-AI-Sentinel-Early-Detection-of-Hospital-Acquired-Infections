# Federated AI Sentinel Documentation

Welcome to the Federated AI Sentinel documentation. This system provides privacy-preserving early detection of hospital-acquired infections using federated learning.

## Overview

The Federated AI Sentinel is a production-ready AI system designed to predict infection risk across multiple hospitals without sharing raw patient data. Each hospital trains locally, and only model updates are shared with a central server.

## Key Features

- **Privacy-Preserving**: No raw patient data leaves your hospital
- **Federated Learning**: Train across multiple sites while keeping data local
- **Explainable AI**: SHAP-based explanations for every prediction
- **Drift Detection**: Automatic monitoring and retrain triggers
- **Production-Ready**: FastAPI service with monitoring and logging
- **Clinical UI**: Clean dashboard for clinicians

## Quick Start

```bash
# Clone repository
git clone https://github.com/saminathan017/Federated-AI-Sentinel-Early-Detection-of-Hospital-Acquired-Infections.git
cd federated_ai_sentinel

# Setup environment
make setup

# Generate synthetic data
python -m src.data.simulators.generate_synthetic_hospital_data

# Train model
make train_temporal

# Start API server
make serve_api

# Start UI
make serve_ui
```

## System Architecture

See [Architecture](architecture.md) for detailed system design.

## Clinical Use

See [Clinical Protocol](clinical_protocol.md) for guidelines on clinical deployment and interpretation.

## Security

See [Threat Model](threat_model.md) for security considerations and privacy controls.

## Support

For questions or issues, please open an issue on GitHub.

## License

MIT License - see LICENSE file for details.

