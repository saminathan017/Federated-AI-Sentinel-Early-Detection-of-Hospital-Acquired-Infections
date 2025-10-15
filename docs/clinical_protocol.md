# Clinical Protocol

## Purpose

This document defines how infection risk predictions should be interpreted and used in clinical practice.

## Risk Score Interpretation

### Risk Levels

- **LOW (0-20%)**: Routine monitoring, standard infection prevention
- **MODERATE (20-50%)**: Enhanced monitoring, reassess in 4-6 hours
- **HIGH (≥50%)**: Consider infection workup, review clinical status

### Prediction Horizon

All risk scores predict infection probability in the **next 24 hours**.

## Labeling Rules

A hospital-acquired infection is defined as:

1. Culture collected >48 hours after admission
2. Positive culture result (bacteria or fungi identified)
3. Clinically significant (not colonization or contamination)

Common culture types:
- Blood cultures
- Urine cultures
- Sputum cultures
- Wound cultures

## Model Limitations

### What the Model Can Do

- Identify patients at elevated infection risk before clinical detection
- Highlight key clinical drivers (vitals, labs)
- Provide quantified uncertainty estimates

### What the Model Cannot Do

- Replace clinical judgment or microbiological testing
- Identify specific pathogens or resistance patterns
- Account for social determinants or subjective findings
- Work outside the training distribution (pediatrics, burns, etc.)

## Clinical Workflow

1. **Morning Review**: Check high-risk patient list on dashboard
2. **Clinical Assessment**: Correlate model prediction with patient exam
3. **Decision Support**: Use SHAP drivers and counterfactuals to guide workup
4. **Documentation**: Note model risk score in clinical reasoning
5. **Feedback**: Confirm or reject alerts to improve future performance

## Recommended Actions by Risk Level

### HIGH Risk (≥50%)

- Review patient immediately
- Assess for SIRS criteria or organ dysfunction
- Consider blood cultures and imaging
- Evaluate need for empiric antibiotics per local guidelines
- Consult infectious disease if complex

### MODERATE Risk (20-50%)

- Reassess within 4-6 hours
- Monitor vitals closely
- Consider trending labs (CBC, CRP, lactate)
- Maintain infection control precautions
- Document clinical stability or deterioration

### LOW Risk (0-20%)

- Continue routine monitoring
- Standard infection prevention measures
- No immediate intervention needed

## Validation Requirements

Before clinical deployment:

1. Validate on local hospital data (AUROC >0.75 target)
2. Test on diverse patient subgroups
3. Review with infection control and informatics
4. Establish local alert thresholds
5. Train clinical staff on interpretation

## Monitoring After Deployment

- Track alert volume and clinician response rate
- Measure time to diagnosis for predicted infections
- Monitor false positive and false negative rates
- Review fairness metrics across demographics
- Retrain when drift detected

## Responsible Use

This is a **decision support tool**, not a diagnostic device.

- Always confirm with clinical assessment
- Never withhold appropriate care due to low model score
- Never initiate therapy based solely on model output
- Consider patient preferences and goals of care
- Document all clinical decision-making

## Disclaimer

This system is a research prototype and is not FDA-cleared or CE-marked. It should not be used for clinical decision-making without:

1. Validation on your institution's data
2. Review by clinical informatics and infection control
3. Integration into clinical workflows with appropriate safeguards
4. Continuous monitoring and human oversight

