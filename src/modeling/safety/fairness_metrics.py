"""
Compute fairness metrics across demographic groups.

Measures equal opportunity gap and calibration per group to detect bias.
"""

import numpy as np
import pandas as pd
from netcal.metrics import ECE
from sklearn.metrics import confusion_matrix, roc_auc_score


def equal_opportunity_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute equal opportunity gap across groups.
    
    Equal opportunity requires equal true positive rates (recall) across groups.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        groups: Group membership (e.g., demographic categories)
        threshold: Decision threshold
    
    Returns:
        Dictionary mapping group to TPR and the maximum gap
    """
    unique_groups = np.unique(groups)
    
    tpr_by_group = {}
    
    for group in unique_groups:
        mask = groups == group
        y_true_group = y_true[mask]
        y_pred_group = (y_pred[mask] >= threshold).astype(int)
        
        if len(y_true_group) > 0 and y_true_group.sum() > 0:
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tpr_by_group[str(group)] = tpr
        else:
            tpr_by_group[str(group)] = np.nan
    
    # Compute max gap
    tpr_values = [v for v in tpr_by_group.values() if not np.isnan(v)]
    if len(tpr_values) > 1:
        max_gap = max(tpr_values) - min(tpr_values)
    else:
        max_gap = 0.0
    
    return {"tpr_by_group": tpr_by_group, "equal_opportunity_gap": max_gap}


def calibration_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """
    Compute calibration error for each group.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        groups: Group membership
    
    Returns:
        Dictionary mapping group to calibration error (ECE)
    """
    unique_groups = np.unique(groups)
    
    ece_by_group = {}
    ece_metric = ECE(bins=10)
    
    for group in unique_groups:
        mask = groups == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        if len(y_true_group) > 10:  # Need enough samples for bins
            ece = ece_metric.measure(y_pred_group, y_true_group)
            ece_by_group[str(group)] = ece
        else:
            ece_by_group[str(group)] = np.nan
    
    return ece_by_group


def performance_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute AUROC and AUPRC for each group.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        groups: Group membership
    
    Returns:
        Nested dictionary with group -> {auroc, auprc}
    """
    from sklearn.metrics import average_precision_score
    
    unique_groups = np.unique(groups)
    
    performance = {}
    
    for group in unique_groups:
        mask = groups == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        if len(y_true_group) > 0 and len(np.unique(y_true_group)) > 1:
            auroc = roc_auc_score(y_true_group, y_pred_group)
            auprc = average_precision_score(y_true_group, y_pred_group)
            
            performance[str(group)] = {"auroc": auroc, "auprc": auprc}
        else:
            performance[str(group)] = {"auroc": np.nan, "auprc": np.nan}
    
    return performance


def fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    group_names: dict[int, str] | None = None,
) -> None:
    """
    Print a comprehensive fairness report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        groups: Group membership
        group_names: Optional mapping from group ID to name
    """
    print("\n" + "=" * 60)
    print("FAIRNESS EVALUATION REPORT")
    print("=" * 60)
    
    # Equal opportunity
    eo_results = equal_opportunity_gap(y_true, y_pred, groups)
    
    print("\nEqual Opportunity (True Positive Rate by Group):")
    for group, tpr in eo_results["tpr_by_group"].items():
        group_name = group_names.get(int(group), group) if group_names else group
        print(f"  {group_name:20s}: {tpr:.4f}")
    
    print(f"\nEqual Opportunity Gap: {eo_results['equal_opportunity_gap']:.4f}")
    
    # Calibration
    cal_results = calibration_by_group(y_true, y_pred, groups)
    
    print("\nCalibration Error (ECE) by Group:")
    for group, ece in cal_results.items():
        group_name = group_names.get(int(group), group) if group_names else group
        print(f"  {group_name:20s}: {ece:.4f}")
    
    # Performance
    perf_results = performance_by_group(y_true, y_pred, groups)
    
    print("\nPerformance by Group:")
    for group, metrics in perf_results.items():
        group_name = group_names.get(int(group), group) if group_names else group
        print(f"  {group_name}:")
        print(f"    AUROC: {metrics['auroc']:.4f}")
        print(f"    AUPRC: {metrics['auprc']:.4f}")
    
    print("=" * 60)


def main() -> None:
    """Example fairness evaluation."""
    # Generate synthetic test data with two groups
    np.random.seed(42)
    
    n_samples = 1000
    
    # Group A: well-calibrated
    y_true_a = np.random.binomial(1, 0.3, size=n_samples // 2)
    y_pred_a = np.random.beta(2, 5, size=n_samples // 2)
    
    # Group B: miscalibrated, lower performance
    y_true_b = np.random.binomial(1, 0.3, size=n_samples // 2)
    y_pred_b = np.random.beta(1, 9, size=n_samples // 2)
    
    y_true = np.concatenate([y_true_a, y_true_b])
    y_pred = np.concatenate([y_pred_a, y_pred_b])
    groups = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    group_names = {0: "Group A", 1: "Group B"}
    
    fairness_report(y_true, y_pred, groups, group_names)


if __name__ == "__main__":
    main()

