# Threat Model and Security Controls

## Privacy Goals

1. **No raw patient data leaves the hospital site**
2. **Model updates do not leak individual patient information**
3. **Inference requests do not store PHI**
4. **Audit trails track all access without exposing patient details**

## Threat Scenarios

### T1: Model Inversion Attack

**Threat**: Adversary reconstructs patient data from model updates

**Mitigation**:
- Differential privacy noise added to gradients (optional)
- Gradient clipping to bound sensitivity
- Secure aggregation protocol (stub, needs implementation)
- Minimum number of patients per site (>100 recommended)

**Status**: Partially mitigated, DP optional

### T2: Data Poisoning

**Threat**: Malicious site poisons global model

**Mitigation**:
- FedAvg with client validation
- Anomaly detection on model updates
- Byzantine-robust aggregation (future work)
- Trusted site onboarding process

**Status**: Basic validation only, needs enhancement

### T3: Membership Inference

**Threat**: Adversary determines if a patient was in training set

**Mitigation**:
- Differential privacy on model updates
- Calibrated probability outputs
- No exact memorization due to aggregation

**Status**: Partially mitigated

### T4: PHI Exposure in Logs

**Threat**: Patient identifiers logged in API requests

**Mitigation**:
- Request IDs used instead of patient IDs in logs
- Audit logs stored separately with access controls
- No PHI in error messages or stack traces

**Status**: Implemented

### T5: Model Stealing

**Threat**: Adversary extracts model via API queries

**Mitigation**:
- Rate limiting on API endpoints
- Query monitoring for suspicious patterns
- API key authentication (future work)

**Status**: Basic rate limiting only

### T6: Insider Threat

**Threat**: Hospital staff misuse patient predictions

**Mitigation**:
- Audit logging of all inference requests
- Role-based access control (future work)
- Alerts for bulk queries

**Status**: Audit logging only, RBAC needed

## Privacy Controls

### At Rest

- Model checkpoints encrypted on disk
- Audit logs encrypted
- No PHI stored in database

### In Transit

- TLS for API communication (optional, off by default)
- Encrypted gRPC for federated learning (future work)

### At Runtime

- Model loaded in memory only
- No caching of patient data
- Automatic memory cleanup

## Compliance Considerations

### HIPAA

- Designed to work with de-identified data
- Audit trails for all access
- Secure communication channels available
- Business associate agreements needed for deployment

### GDPR

- Right to explanation: SHAP provides interpretability
- Right to be forgotten: No patient data retained after training
- Data minimization: Only aggregated updates shared
- Consent: Required from each participating site

## Audit Trail

All API requests logged with:
- Request ID (not patient ID)
- Timestamp
- Endpoint called
- Response status
- Execution time

No PHI stored in logs.

## Deployment Recommendations

1. **Network Isolation**: Deploy API behind hospital firewall
2. **Authentication**: Add API key or OAuth before production
3. **TLS**: Enable HTTPS with valid certificates
4. **Monitoring**: Alert on unusual query patterns
5. **Access Control**: Restrict to authorized clinical users
6. **Audit Review**: Regular review of access logs

## Future Security Work

- [ ] Implement secure multi-party computation for aggregation
- [ ] Add homomorphic encryption for model updates
- [ ] Byzantine-robust aggregation
- [ ] Zero-knowledge proofs for verification
- [ ] Full differential privacy accounting
- [ ] Role-based access control
- [ ] API authentication and authorization

## Responsible Disclosure

Security issues should be reported to the maintainers via GitHub security advisories, not public issues.

