# Section 5: Cost-Aware Optimization Framework - Notebook Cells

**Installation Location**: Replace existing Section 5 cells

---

## Cell 5.1 [MARKDOWN]

```markdown
# **Section 5: Enhanced Cost-Aware Optimization Framework**

This section establishes the cost-aware evaluation infrastructure that aligns model performance with business objectives.

## **Key Components**

1. **CostConfig Class**: Centralized configuration for business costs, SMOTE settings, and cross-validation parameters
2. **Cost Calculation Utilities**: Functions to compute expected maintenance costs
3. **Threshold Optimization**: Automated search for cost-minimizing decision thresholds
4. **Sensitivity Analysis**: Robustness testing under cost parameter variations

## **Business Cost Structure**

| Outcome | Cost | Interpretation |
|---------|------|----------------|
| **False Negative (FN)** | $100 | Missed failure → Unplanned generator replacement |
| **True Positive (TP)** | $30 | Detected failure → Scheduled proactive repair |
| **False Positive (FP)** | $10 | False alarm → Inspection truck roll |
| **True Negative (TN)** | $0 | Correctly identified normal operation |

**Cost Hierarchy**: Replacement ($100) >> Repair ($30) >> Inspection ($10) >> Normal ($0)

This hierarchy reflects real-world maintenance economics where preventing catastrophic failures delivers the highest value.
```

---

## Cell 5.2 [CODE] - CostConfig Class

```python
# ===============================================
# COST CONFIGURATION CLASS
# ===============================================

class CostConfig:
    """
    Centralized configuration for cost-aware optimization and experimentation.
    
    This class manages:
    - Business cost structure (FN, TP, FP, TN)
    - SMOTE oversampling configuration
    - Cross-validation parameters
    - Random seed for reproducibility
    """
    
    # ==================== BUSINESS COSTS ====================
    FN = 100.0  # False Negative: unplanned replacement cost
    TP = 30.0   # True Positive: proactive repair cost
    FP = 10.0   # False Positive: inspection cost
    TN = 0.0    # True Negative: normal operations (no cost)
    
    # ==================== SMOTE CONFIGURATION ====================
    USE_SMOTE = True  # Toggle SMOTE oversampling
    SMOTE_RATIO = 0.5  # Target ratio (0.5 = minority 50% of majority)
    SMOTE_K_NEIGHBORS = 5  # K-neighbors for synthetic samples
    
    # ==================== CROSS-VALIDATION ====================
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    # ==================== TRAINING ====================
    EPOCHS = 100
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    
    @classmethod
    def get_cost_dict(cls):
        """Return cost structure as dictionary."""
        return {'FN': cls.FN, 'TP': cls.TP, 'FP': cls.FP, 'TN': cls.TN}
    
    @classmethod
    def display_config(cls):
        """Print current configuration."""
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print("\n📊 Business Cost Structure:")
        print(f"   FN (Replacement):  ${cls.FN:.2f}")
        print(f"   TP (Repair):       ${cls.TP:.2f}")
        print(f"   FP (Inspection):   ${cls.FP:.2f}")
        print(f"   TN (Normal):       ${cls.TN:.2f}")
        
        print(f"\n🔬 SMOTE: {'✓ ENABLED' if cls.USE_SMOTE else '✗ DISABLED'}")
        if cls.USE_SMOTE:
            print(f"   Ratio: {cls.SMOTE_RATIO}, K-neighbors: {cls.SMOTE_K_NEIGHBORS}")
        
        print(f"\n🔄 Cross-Validation: {cls.N_SPLITS}-Fold, Seed: {cls.RANDOM_STATE}")
        print(f"⚙️  Training: {cls.EPOCHS} epochs, batch {cls.BATCH_SIZE}")
        print("=" * 70)

CostConfig.display_config()
```

---

## Cell 5.3 [CODE] - Cost Utilities (Part 1)

```python
# ===============================================
# COST CALCULATION UTILITIES
# ===============================================

def calculate_expected_cost(y_true, y_pred_proba, threshold, costs=None):
    """
    Calculate expected maintenance cost for given threshold.
    
    Returns:
    --------
    expected_cost : float
    metrics : dict with confusion matrix and costs
    """
    if costs is None:
        costs = CostConfig.get_cost_dict()
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    
    # Calculate costs
    total_cost = fn * costs['FN'] + tp * costs['TP'] + fp * costs['FP'] + tn * costs['TN']
    expected_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0
    
    metrics = {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cost_fn': float(fn * costs['FN']),
        'cost_tp': float(tp * costs['TP']),
        'cost_fp': float(fp * costs['FP']),
        'total_cost': float(total_cost),
        'expected_cost': float(expected_cost)
    }
    
    return expected_cost, metrics


def optimize_threshold(y_true, y_pred_proba, costs=None, 
                       threshold_range=(0.05, 0.95), n_points=91):
    """
    Find optimal decision threshold that minimizes expected cost.
    
    Returns:
    --------
    optimal_threshold : float
    optimal_cost : float
    cost_curve : pd.DataFrame
    """
    if costs is None:
        costs = CostConfig.get_cost_dict()
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
    costs_at_thresholds = []
    
    for thresh in thresholds:
        expected_cost, _ = calculate_expected_cost(y_true, y_pred_proba, thresh, costs)
        costs_at_thresholds.append(expected_cost)
    
    optimal_idx = np.argmin(costs_at_thresholds)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs_at_thresholds[optimal_idx]
    
    cost_curve = pd.DataFrame({
        'threshold': thresholds,
        'expected_cost': costs_at_thresholds
    })
    
    return optimal_threshold, optimal_cost, cost_curve

print("✅ Cost calculation utilities loaded")
```

---

## Cell 5.4 [CODE] - Sensitivity Analysis

```python
# ===============================================
# COST SENSITIVITY ANALYSIS
# ===============================================

def cost_sensitivity_analysis(y_true, y_pred_proba, base_threshold=None, perturbation=0.20):
    """
    Analyze sensitivity of optimal threshold to cost parameter variations.
    Tests ±20% variations in FN and FP costs.
    """
    base_costs = CostConfig.get_cost_dict()
    scenarios = []
    
    fn_values = [base_costs['FN'] * (1 - perturbation),
                 base_costs['FN'],
                 base_costs['FN'] * (1 + perturbation)]
    
    fp_values = [base_costs['FP'] * (1 - perturbation),
                 base_costs['FP'],
                 base_costs['FP'] * (1 + perturbation)]
    
    for fn_cost in fn_values:
        for fp_cost in fp_values:
            test_costs = {'FN': fn_cost, 'TP': base_costs['TP'],
                         'FP': fp_cost, 'TN': base_costs['TN']}
            
            optimal_t, optimal_c, _ = optimize_threshold(y_true, y_pred_proba, test_costs)
            
            scenarios.append({
                'FN_cost': fn_cost,
                'FP_cost': fp_cost,
                'scenario': f"FN=${fn_cost:.0f}, FP=${fp_cost:.0f}",
                'optimal_threshold': optimal_t,
                'optimal_cost': optimal_c
            })
    
    sensitivity_df = pd.DataFrame(scenarios)
    
    print("\n" + "=" * 70)
    print("COST SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Threshold range: {sensitivity_df['optimal_threshold'].min():.3f} - "
          f"{sensitivity_df['optimal_threshold'].max():.3f}")
    print(f"Threshold std: ±{sensitivity_df['optimal_threshold'].std():.3f}")
    
    if base_threshold:
        max_dev = abs(sensitivity_df['optimal_threshold'] - base_threshold).max()
        print(f"Max deviation from base: ±{max_dev:.3f}")
        print(f"Robustness: {'✓ STABLE' if max_dev < 0.10 else '⚠ VARIABLE'}")
    
    print("=" * 70)
    
    return sensitivity_df

print("✅ Cost sensitivity analysis loaded")
```

---

## Cell 5.5 [MARKDOWN] - Summary

```markdown
## **Cost Optimization Strategy**

The cost-aware threshold optimization operates as follows:

1. **Grid Search**: Evaluate expected cost at 91 thresholds spanning 0.05 to 0.95
2. **Cost Calculation**: For each threshold, compute confusion matrix and apply business costs
3. **Optimal Selection**: Choose threshold τ* that minimizes expected cost per turbine
4. **Sensitivity Testing**: Verify robustness by testing ±20% cost variations

**Key Insight**: Traditional ML metrics (accuracy, F1) assume equal misclassification costs. In ReneWind's case, missing a failure (FN) costs 10× more than a false alarm (FP), making cost-aware optimization essential.

**Expected Improvement**: Threshold optimization typically reduces maintenance costs by 15-30% compared to the default threshold of 0.5.
```

