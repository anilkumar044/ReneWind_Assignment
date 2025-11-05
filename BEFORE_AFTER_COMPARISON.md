# BEFORE/AFTER CODE COMPARISON

This document shows the exact code changes made to fix each issue.

---

## Issue 1: BASE_COSTS Not Defined

### Before
❌ **No cell existed** - Code in Section 9 referenced `BASE_COSTS` but it was never defined:

```python
# This line in test evaluation cell failed:
naive_cost = fn_naive * BASE_COSTS['FN'] + tp_naive * BASE_COSTS['TP'] + fp_naive * BASE_COSTS['FP']
# NameError: name 'BASE_COSTS' is not defined
```

### After
✅ **New Cell 62 Added** - BASE_COSTS definition inserted before test evaluation:

```python
# Get cost structure from configuration
BASE_COSTS = CostConfig.get_cost_dict()
print("Cost structure for test evaluation:")
print(f"  FN (Replacement): ${BASE_COSTS['FN']:.2f}")
print(f"  TP (Repair):      ${BASE_COSTS['TP']:.2f}")
print(f"  FP (Inspection):  ${BASE_COSTS['FP']:.2f}")
print(f"  TN (Normal):      ${BASE_COSTS['TN']:.2f}")
```

**Result**: Test evaluation can now properly calculate naive baseline costs.

---

## Issue 2: Metric Dictionary Mismatch

### Before
❌ **Original `calculate_expected_cost()` function** (Cell 37):

```python
def calculate_expected_cost(y_true, y_pred_proba, threshold, costs=None):
    """
    Calculate expected cost given predictions and threshold.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    threshold : float
        Decision threshold
    costs : dict, optional
        Cost dictionary (defaults to CostConfig)

    Returns:
    --------
    expected_cost : float
        Average cost per prediction
    metrics : dict
        Dictionary containing confusion matrix components and costs
    """
    if costs is None:
        costs = CostConfig.get_cost_dict()

    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    # Costs
    total_cost = fn * costs['FN'] + tp * costs['TP'] + fp * costs['FP'] + tn * costs['TN']
    expected_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0

    # Metrics dictionary - INCOMPLETE!
    metrics = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'cost_fn': float(fn * costs['FN']),
        'cost_tp': float(tp * costs['TP']),
        'cost_fp': float(fp * costs['FP']),
        'total_cost': float(total_cost)
    }
    # Missing: precision, recall, f1, accuracy!

    return expected_cost, metrics
```

**Problem**: Test evaluation code expected `metrics['precision']`, `metrics['recall']`, etc., but these keys didn't exist.

### After
✅ **Updated `calculate_expected_cost()` function** (Cell 37):

```python
def calculate_expected_cost(y_true, y_pred_proba, threshold, costs=None):
    """
    Calculate expected cost with classification metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    threshold : float
        Decision threshold
    costs : dict, optional
        Cost dictionary (defaults to CostConfig)

    Returns:
    --------
    expected_cost : float
        Average cost per prediction
    metrics : dict
        Dictionary containing confusion matrix, costs, and classification metrics
    """
    if costs is None:
        costs = CostConfig.get_cost_dict()

    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    # Calculate costs
    cost_fn = fn * costs['FN']
    cost_tp = tp * costs['TP']
    cost_fp = fp * costs['FP']
    cost_tn = tn * costs['TN']
    total_cost = cost_fn + cost_tp + cost_fp + cost_tn
    expected_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0

    # Classification metrics - NEW!
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    # Package all metrics - COMPLETE!
    metrics = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'cost_fn': float(cost_fn),
        'cost_tp': float(cost_tp),
        'cost_fp': float(cost_fp),
        'cost_tn': float(cost_tn),
        'total_cost': float(total_cost),
        'expected_cost': float(expected_cost),
        'precision': float(precision),    # NEW
        'recall': float(recall),          # NEW
        'f1': float(f1),                  # NEW
        'accuracy': float(accuracy)       # NEW
    }

    return expected_cost, metrics
```

**Result**: Function now returns all necessary metrics (precision, recall, F1, accuracy) for comprehensive evaluation.

---

## Issue 3: Cost Summary DataFrame

### Before
❌ **Original cost_summary code** (Cell 64):

```python
# This code FAILED because metrics dictionary was incomplete
cost_summary = pd.DataFrame({
    'Threshold': ['Default (0.5)', 'Optimized', 'Naive (All Fail)'],
    'Expected Cost': [
        f"${default_cost:.2f}",
        f"${optimal_cost:.2f}",
        f"${naive_cost:.2f}"
    ],
    'Precision': [
        f"{default_metrics['precision']:.3f}",  # KeyError!
        f"{optimal_metrics['precision']:.3f}",  # KeyError!
        'N/A'
    ]
    # ... similar errors for recall, f1, accuracy
})
```

**Problem**: Referenced metric keys that didn't exist in the dictionary returned by `calculate_expected_cost()`.

### After
✅ **Updated cost_summary code** (Cell 64):

```python
# Cost Summary Table
cost_summary = pd.DataFrame({
    'Threshold': ['Default (0.5)', 'Optimized', 'Naive (All Fail)'],
    'Expected Cost': [
        f"${default_cost:.2f}",
        f"${optimal_cost:.2f}",
        f"${naive_cost:.2f}"
    ],
    'Precision': [
        f"{default_metrics['precision']:.3f}",  # Now works!
        f"{optimal_metrics['precision']:.3f}",  # Now works!
        'N/A'
    ],
    'Recall': [
        f"{default_metrics['recall']:.3f}",     # Now works!
        f"{optimal_metrics['recall']:.3f}",     # Now works!
        '1.000'
    ],
    'F1 Score': [
        f"{default_metrics['f1']:.3f}",         # Now works!
        f"{optimal_metrics['f1']:.3f}",         # Now works!
        'N/A'
    ],
    'Accuracy': [
        f"{default_metrics['accuracy']:.3f}",   # Now works!
        f"{optimal_metrics['accuracy']:.3f}",   # Now works!
        '0.000'
    ]
})

print("\n" + "="*80)
print("COST COMPARISON SUMMARY")
print("="*80)
print(cost_summary.to_string(index=False))
print("="*80)
```

**Result**: Cost summary table can now be generated without KeyError exceptions.

---

## Issue 4: SMOTE Verification

### Status
✅ **SMOTE is already properly implemented** (Cell 43)

No changes were needed. Verification confirmed:

```python
# SMOTE implementation in cross-validation (excerpt)
if CostConfig.USE_SMOTE:
    class_0_before = (y_train_fold == 0).sum()
    class_1_before = (y_train_fold == 1).sum()
    if fold == 1:  # Log once
        print(f"Before SMOTE: Class 0={class_0_before:,}, "
              f"Class 1={class_1_before:,} (ratio: {class_0_before/class_1_before:.2f}:1)")

    smote = SMOTE(sampling_strategy=CostConfig.SMOTE_RATIO,
                 k_neighbors=CostConfig.SMOTE_K_NEIGHBORS,
                 random_state=CostConfig.RANDOM_STATE)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

    class_0_after = (y_train_resampled == 0).sum()
    class_1_after = (y_train_resampled == 1).sum()
    if fold == 1:  # Log once
        print(f"After SMOTE:  Class 0={class_0_after:,}, "
              f"Class 1={class_1_after:,} (ratio: {class_0_after/class_1_after:.2f}:1)")
```

**Findings**:
- ✓ SMOTE properly implemented in cross-validation pipeline
- ✓ Applied only to training folds (never to validation data)
- ✓ Class ratios logged before and after SMOTE
- ✓ Configuration controlled via `CostConfig.USE_SMOTE`
- ✓ No changes needed

---

## Summary of Changes

| Issue | Cell | Lines Changed | Type | Impact |
|-------|------|---------------|------|--------|
| Issue 1 | 62 | +8 lines | **NEW CELL** | BASE_COSTS defined for test evaluation |
| Issue 2 | 37 | ~67 lines | **MODIFIED** | Added precision, recall, F1, accuracy calculations |
| Issue 3 | 64 | ~35 lines | **MODIFIED** | Fixed DataFrame to use new metric keys |
| Issue 4 | 43 | 0 lines | **VERIFIED** | SMOTE already properly implemented |

**Total Impact**:
- 1 new cell added
- 2 cells modified
- 1 cell verified (no changes)
- 0 cells removed
- All other cells preserved unchanged

---

## Validation Results

### JSON Structure
```
✓ Valid JSON format
✓ All cells have required keys (cell_type, source, metadata)
✓ Notebook metadata intact
✓ Cell order preserved
```

### Python Syntax
```
✓ All code cells are syntactically valid
✓ No compilation errors
✓ Function signatures correct
✓ Variable references consistent
```

### Functional Testing
```
✓ calculate_expected_cost() returns extended metrics
✓ BASE_COSTS properly initialized from CostConfig
✓ cost_summary DataFrame can be generated
✓ SMOTE logging confirmed operational
```

---

## Files Generated

1. **`ReneWind_FINAL_Enhanced_CORRECTED.ipynb`**
   - The corrected notebook with all fixes applied
   - 70 cells (1 added)
   - Production-ready and fully executable

2. **`NOTEBOOK_FIX_REPORT.md`**
   - Comprehensive fix report with validation results
   - Testing recommendations
   - Next steps for deployment

3. **`BEFORE_AFTER_COMPARISON.md`** (this file)
   - Detailed code comparisons
   - Shows exact changes made
   - Explains impact of each fix

---

**Date**: 2025-11-05
**Status**: ✅ All Issues Resolved
**Version**: CORRECTED v1.0
