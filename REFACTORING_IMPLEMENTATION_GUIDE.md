# ReneWind Notebook - Practical Refactoring & Enhancement Guide

**Goal**: Enrich visualizations, add SMOTE, track 35 training runs, leverage existing code structure

**Approach**: Refactor existing utilities → Enhance visualizations → Integrate SMOTE → Track all CV runs

---

## Current State Analysis

### ✅ What You Already Have

| Component | Status | Location | Action |
|-----------|--------|----------|--------|
| `calculate_expected_cost()` | ✅ Exists | Section 5 | **Move earlier, add to utility module** |
| `cost_sensitivity_analysis()` | ✅ Exists | Section 5 | **Keep, enhance visualization** |
| `StratifiedKFold` setup | ✅ Exists | Section 4 | **Keep, integrate with SMOTE** |
| Training function | ✅ Exists | Section 6 | **Enhance to track 35 runs** |
| `plot_cost_curve()` | ✅ Exists | Section 6 | **Reuse for enhanced visuals** |
| `plot_training_history()` | ✅ Exists | Section 6 | **Reuse for comparison plots** |
| `plot_roc_pr_curves()` | ✅ Exists | Section 6 | **Reuse for per-fold overlays** |
| EDA plots (histograms, PCA, correlation) | ✅ Exists | Section 3 | **Keep as-is** |
| Model comparison bar charts | ✅ Exists | Section 8 | **Enhance with cost metrics** |
| Business narrative markdown | ✅ Exists | Multiple sections | **Keep and expand** |

### ⚠️ What Needs Enhancement

| Enhancement | Current State | Target State |
|-------------|--------------|--------------|
| **Cost utilities** | Scattered in Section 5 | Consolidated utility module in Section 5 |
| **SMOTE integration** | Not present | Config-driven SMOTE in CV loop |
| **35-run tracking** | Implicit (happens but not shown) | Explicit tracking + visualizations |
| **Fold-level visualizations** | Only aggregated metrics | Per-fold box plots, heatmaps, overlays |
| **Cost-aware plots** | Basic cost curve | Enhanced: 3D surface, sensitivity heatmap |
| **Model comparison** | Accuracy/AUC bars | Add cost-centric radar chart, box plots |

---

## Refactoring Plan (5 Steps)

### Step 1: Consolidate Utility Module (Section 5)

**Current**: Functions defined but not organized
**Target**: Clean utility module with all cost/threshold functions

**Implementation**:

```python
# ===============================================
# SECTION 5: COST-AWARE OPTIMIZATION UTILITIES
# ===============================================

class CostConfig:
    """Central configuration for business costs and optimization."""

    # Business costs (USD)
    FN = 100.0  # False Negative: unplanned replacement
    TP = 30.0   # True Positive: proactive repair
    FP = 10.0   # False Positive: inspection cost
    TN = 0.0    # True Negative: normal operations

    # SMOTE configuration
    USE_SMOTE = True  # Toggle for experimentation
    SMOTE_RATIO = 0.5  # Target minority:majority ratio (0.5 = 2:1)
    SMOTE_K_NEIGHBORS = 5

    # Cross-validation
    N_SPLITS = 5
    RANDOM_STATE = 42

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
        print(f"Cost Structure:")
        print(f"  FN (Replacement): ${cls.FN:.2f}")
        print(f"  TP (Repair):      ${cls.TP:.2f}")
        print(f"  FP (Inspection):  ${cls.FP:.2f}")
        print(f"  TN (Normal):      ${cls.TN:.2f}")
        print(f"\nSMOTE: {'ENABLED' if cls.USE_SMOTE else 'DISABLED'}")
        if cls.USE_SMOTE:
            print(f"  Sampling ratio:   {cls.SMOTE_RATIO}")
            print(f"  K-neighbors:      {cls.SMOTE_K_NEIGHBORS}")
        print(f"\nCross-Validation: {cls.N_SPLITS}-fold StratifiedKFold")
        print(f"Random seed:      {cls.RANDOM_STATE}")
        print("=" * 70)

# Display configuration
CostConfig.display_config()


# ===============================================
# COST CALCULATION UTILITIES
# ===============================================

def calculate_expected_cost(y_true, y_pred_proba, threshold, costs=None):
    """
    Calculate expected maintenance cost for given threshold.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (1 = failure, 0 = no failure)
    y_pred_proba : array-like
        Predicted probabilities for positive class
    threshold : float
        Decision threshold (0-1)
    costs : dict, optional
        Cost structure {'FN': x, 'TP': y, 'FP': z, 'TN': w}
        If None, uses CostConfig defaults

    Returns:
    --------
    expected_cost : float
        Expected cost per turbine
    metrics : dict
        Confusion matrix counts and costs
    """
    if costs is None:
        costs = CostConfig.get_cost_dict()

    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate confusion matrix
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
    n_samples = len(y_true)
    expected_cost = total_cost / n_samples

    metrics = {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'cost_fn': cost_fn, 'cost_tp': cost_tp,
        'cost_fp': cost_fp, 'cost_tn': cost_tn,
        'total_cost': total_cost,
        'expected_cost': expected_cost
    }

    return expected_cost, metrics


def optimize_threshold(y_true, y_pred_proba, costs=None,
                       threshold_range=(0.05, 0.95), n_points=91):
    """
    Find optimal decision threshold that minimizes expected cost.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    costs : dict, optional
        Cost structure, uses CostConfig if None
    threshold_range : tuple
        (min, max) threshold values to test
    n_points : int
        Number of thresholds to test

    Returns:
    --------
    optimal_threshold : float
        Threshold that minimizes cost
    optimal_cost : float
        Minimum expected cost
    cost_curve : pd.DataFrame
        Full cost curve for visualization
    """
    if costs is None:
        costs = CostConfig.get_cost_dict()

    # Test thresholds
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
    costs_at_thresholds = []

    for thresh in thresholds:
        expected_cost, _ = calculate_expected_cost(y_true, y_pred_proba, thresh, costs)
        costs_at_thresholds.append(expected_cost)

    # Find optimal
    optimal_idx = np.argmin(costs_at_thresholds)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs_at_thresholds[optimal_idx]

    # Create cost curve DataFrame
    cost_curve = pd.DataFrame({
        'threshold': thresholds,
        'expected_cost': costs_at_thresholds
    })

    return optimal_threshold, optimal_cost, cost_curve


def cost_sensitivity_analysis(y_true, y_pred_proba, base_threshold=None,
                               perturbation=0.20):
    """
    Analyze sensitivity of optimal threshold to cost parameter changes.

    Tests ±20% variations in FN and FP costs to assess robustness.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    base_threshold : float, optional
        Baseline optimal threshold to compare against
    perturbation : float
        Perturbation factor (0.20 = ±20%)

    Returns:
    --------
    sensitivity_results : pd.DataFrame
        Results for each cost scenario
    """
    base_costs = CostConfig.get_cost_dict()

    scenarios = []

    # Test different cost scenarios
    fn_values = [base_costs['FN'] * (1 - perturbation),
                 base_costs['FN'],
                 base_costs['FN'] * (1 + perturbation)]

    fp_values = [base_costs['FP'] * (1 - perturbation),
                 base_costs['FP'],
                 base_costs['FP'] * (1 + perturbation)]

    for fn_cost in fn_values:
        for fp_cost in fp_values:
            test_costs = {
                'FN': fn_cost,
                'TP': base_costs['TP'],
                'FP': fp_cost,
                'TN': base_costs['TN']
            }

            optimal_t, optimal_c, _ = optimize_threshold(y_true, y_pred_proba, test_costs)

            scenarios.append({
                'FN_cost': fn_cost,
                'FP_cost': fp_cost,
                'scenario': f"FN=${fn_cost:.0f}, FP=${fp_cost:.0f}",
                'optimal_threshold': optimal_t,
                'optimal_cost': optimal_c
            })

    sensitivity_df = pd.DataFrame(scenarios)

    # Summary statistics
    print("\n📊 Cost Sensitivity Analysis")
    print("=" * 70)
    print(f"Threshold range: {sensitivity_df['optimal_threshold'].min():.3f} - "
          f"{sensitivity_df['optimal_threshold'].max():.3f}")
    print(f"Threshold std: ±{sensitivity_df['optimal_threshold'].std():.3f}")
    print(f"Cost range: ${sensitivity_df['optimal_cost'].min():.2f} - "
          f"${sensitivity_df['optimal_cost'].max():.2f}")

    if base_threshold is not None:
        print(f"\nBase optimal threshold: {base_threshold:.3f}")
        print(f"Max deviation: ±{abs(sensitivity_df['optimal_threshold'] - base_threshold).max():.3f}")

    return sensitivity_df


print("✅ Cost optimization utilities loaded")
```

**Integration Point**: This goes in **Section 5** (replace current cost function cells)

---

### Step 2: Enhanced Cross-Validation Loop with SMOTE

**Current**: Basic CV loop
**Target**: Track all 35 runs, integrate SMOTE, capture detailed metrics

**Implementation**:

```python
# ===============================================
# SECTION 6: ENHANCED CROSS-VALIDATION PIPELINE
# ===============================================

from imblearn.over_sampling import SMOTE

class CVResultsTracker:
    """Track detailed results across all 35 training runs (7 models × 5 folds)."""

    def __init__(self):
        self.all_runs = []  # List of all 35 run results
        self.model_summaries = {}  # Aggregated results per model

    def add_run(self, model_name, fold_idx, results):
        """Add single fold result."""
        run_data = {
            'model_name': model_name,
            'fold': fold_idx + 1,
            'run_id': f"{model_name}_fold{fold_idx+1}",
            **results
        }
        self.all_runs.append(run_data)

    def get_summary_df(self):
        """Get DataFrame of all 35 runs."""
        return pd.DataFrame(self.all_runs)

    def get_model_summary(self, model_name):
        """Get aggregated statistics for one model (5 folds)."""
        model_runs = [r for r in self.all_runs if r['model_name'] == model_name]

        if not model_runs:
            return None

        # Aggregate metrics
        summary = {
            'model_name': model_name,
            'n_folds': len(model_runs),
            'mean_auc': np.mean([r['auc'] for r in model_runs]),
            'std_auc': np.std([r['auc'] for r in model_runs]),
            'mean_optimal_cost': np.mean([r['optimal_cost'] for r in model_runs]),
            'std_optimal_cost': np.std([r['optimal_cost'] for r in model_runs]),
            'mean_default_cost': np.mean([r['default_cost'] for r in model_runs]),
            'mean_recall_optimal': np.mean([r['recall_optimal'] for r in model_runs]),
            'std_recall_optimal': np.std([r['recall_optimal'] for r in model_runs]),
            'mean_precision_optimal': np.mean([r['precision_optimal'] for r in model_runs]),
            'mean_f1_optimal': np.mean([r['f1_optimal'] for r in model_runs]),
            'mean_optimal_threshold': np.mean([r['optimal_threshold'] for r in model_runs]),
        }

        return summary

    def get_all_model_summaries(self):
        """Get summary for all models."""
        unique_models = list(set([r['model_name'] for r in self.all_runs]))
        summaries = [self.get_model_summary(m) for m in unique_models]
        return pd.DataFrame(summaries)


# Initialize tracker
cv_tracker = CVResultsTracker()


def train_model_with_enhanced_cv(model_fn, model_name, use_class_weights=False):
    """
    Train model with enhanced 5-fold CV, SMOTE support, and detailed tracking.

    Parameters:
    -----------
    model_fn : callable
        Function that returns a compiled Keras model
    model_name : str
        Name of the model for tracking
    use_class_weights : bool
        Whether to use class weights during training

    Returns:
    --------
    fold_results : list
        Detailed results for each fold
    """
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)
    print(f"Configuration: SMOTE={'ON' if CostConfig.USE_SMOTE else 'OFF'}, "
          f"Class Weights={'ON' if use_class_weights else 'OFF'}")

    skf = StratifiedKFold(n_splits=CostConfig.N_SPLITS,
                          shuffle=True,
                          random_state=CostConfig.RANDOM_STATE)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'─' * 70}")
        print(f"Fold {fold_idx + 1}/{CostConfig.N_SPLITS}")
        print(f"{'─' * 70}")

        # Split data
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()

        # Leak-safe preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_fold = imputer.fit_transform(X_train_fold)
        X_val_fold = imputer.transform(X_val_fold)

        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        # SMOTE (only on training data, never on validation)
        original_train_size = len(y_train_fold)
        if CostConfig.USE_SMOTE:
            class_0_count = (y_train_fold == 0).sum()
            class_1_count = (y_train_fold == 1).sum()

            print(f"Before SMOTE: Class 0={class_0_count}, Class 1={class_1_count} "
                  f"(ratio {class_0_count/class_1_count:.1f}:1)")

            smote = SMOTE(sampling_strategy=CostConfig.SMOTE_RATIO,
                         k_neighbors=CostConfig.SMOTE_K_NEIGHBORS,
                         random_state=CostConfig.RANDOM_STATE)

            X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

            class_0_count_after = (y_train_fold == 0).sum()
            class_1_count_after = (y_train_fold == 1).sum()

            print(f"After SMOTE:  Class 0={class_0_count_after}, Class 1={class_1_count_after} "
                  f"(ratio {class_0_count_after/class_1_count_after:.1f}:1)")
            print(f"Synthetic samples added: {len(y_train_fold) - original_train_size}")

        # Class weights (computed on potentially SMOTE-augmented data)
        class_weights = None
        if use_class_weights:
            weights_array = compute_class_weight('balanced',
                                                 classes=np.unique(y_train_fold),
                                                 y=y_train_fold)
            class_weights = {0: weights_array[0], 1: weights_array[1]}
            print(f"Class weights: {{0: {class_weights[0]:.3f}, 1: {class_weights[1]:.3f}}}")

        # Create and train model
        model = model_fn()

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=5, min_lr=1e-7, verbose=0)

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=100,
            batch_size=32,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Predictions
        y_val_pred_proba = model.predict(X_val_fold, verbose=0).flatten()

        # Calculate metrics at default threshold (0.5)
        y_val_pred_default = (y_val_pred_proba >= 0.5).astype(int)
        default_cost, default_metrics = calculate_expected_cost(
            y_val_fold, y_val_pred_proba, threshold=0.5
        )

        # Optimize threshold
        optimal_threshold, optimal_cost, cost_curve = optimize_threshold(
            y_val_fold, y_val_pred_proba
        )

        # Metrics at optimal threshold
        y_val_pred_optimal = (y_val_pred_proba >= optimal_threshold).astype(int)

        # Calculate classification metrics
        from sklearn.metrics import (roc_auc_score, precision_score,
                                     recall_score, f1_score, roc_curve,
                                     precision_recall_curve)

        auc = roc_auc_score(y_val_fold, y_val_pred_proba)

        recall_default = recall_score(y_val_fold, y_val_pred_default, zero_division=0)
        recall_optimal = recall_score(y_val_fold, y_val_pred_optimal, zero_division=0)

        precision_default = precision_score(y_val_fold, y_val_pred_default, zero_division=0)
        precision_optimal = precision_score(y_val_fold, y_val_pred_optimal, zero_division=0)

        f1_default = f1_score(y_val_fold, y_val_pred_default, zero_division=0)
        f1_optimal = f1_score(y_val_fold, y_val_pred_optimal, zero_division=0)

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_proba)

        # PR curve data
        precision_curve, recall_curve, _ = precision_recall_curve(y_val_fold, y_val_pred_proba)

        # Store results
        fold_data = {
            'auc': auc,
            'optimal_threshold': optimal_threshold,
            'optimal_cost': optimal_cost,
            'default_cost': default_cost,
            'cost_savings': default_cost - optimal_cost,
            'recall_default': recall_default,
            'recall_optimal': recall_optimal,
            'precision_default': precision_default,
            'precision_optimal': precision_optimal,
            'f1_default': f1_default,
            'f1_optimal': f1_optimal,
            'confusion_matrix_optimal': default_metrics,
            'cost_curve': cost_curve,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'training_history': history.history,
            'smote_used': CostConfig.USE_SMOTE,
            'class_weights_used': use_class_weights,
            'training_samples': len(y_train_fold),
            'validation_samples': len(y_val_fold)
        }

        fold_results.append(fold_data)

        # Add to global tracker
        cv_tracker.add_run(model_name, fold_idx, fold_data)

        # Print fold summary
        print(f"\n✓ Fold {fold_idx + 1} Complete:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Optimal τ: {optimal_threshold:.3f}")
        print(f"  Cost @ τ=0.5: ${default_cost:.2f}")
        print(f"  Cost @ τ*:    ${optimal_cost:.2f} (saves ${cost_savings:.2f})")
        print(f"  Recall @ τ*:  {recall_optimal:.3f}")
        print(f"  Precision @ τ*: {precision_optimal:.3f}")

    # Print model summary
    print(f"\n{'='*70}")
    print(f"{model_name} - 5-FOLD SUMMARY")
    print(f"{'='*70}")
    print(f"Mean AUC:           {np.mean([f['auc'] for f in fold_results]):.4f} ± "
          f"{np.std([f['auc'] for f in fold_results]):.4f}")
    print(f"Mean Optimal Cost:  ${np.mean([f['optimal_cost'] for f in fold_results]):.2f} ± "
          f"${np.std([f['optimal_cost'] for f in fold_results]):.2f}")
    print(f"Mean Recall @ τ*:   {np.mean([f['recall_optimal'] for f in fold_results]):.3f} ± "
          f"{np.std([f['recall_optimal'] for f in fold_results]):.3f}")
    print(f"Mean Optimal τ:     {np.mean([f['optimal_threshold'] for f in fold_results]):.3f}")
    print(f"{'='*70}\n")

    return fold_results


print("✅ Enhanced CV pipeline with SMOTE support loaded")
```

**Integration Point**: This goes in **Section 6** (replace current training function)

---

### Step 3: Enhanced Visualization Suite

**Current**: Basic plots exist
**Target**: Reuse and extend with fold-level details

**Implementation**:

```python
# ===============================================
# SECTION 6.5: ENHANCED VISUALIZATION SUITE
# ===============================================

def plot_cv_performance_boxplots(cv_tracker):
    """
    Box plots showing performance distribution across 5 folds for all models.
    """
    df = cv_tracker.get_summary_df()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = ['auc', 'recall_optimal', 'precision_optimal',
               'f1_optimal', 'optimal_cost', 'optimal_threshold']
    titles = ['AUC', 'Recall @ τ*', 'Precision @ τ*',
              'F1 @ τ*', 'Optimal Cost ($)', 'Optimal Threshold']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # Prepare data for box plot
        models = df['model_name'].unique()
        data = [df[df['model_name'] == m][metric].values for m in models]
        labels = [m.replace('Model ', 'M') for m in models]

        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       notch=True, showmeans=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    # Remove extra subplot
    fig.delaxes(axes[5])

    fig.suptitle('Model Performance Distribution Across 5 Folds (35 Training Runs Total)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_detailed_35_runs_table(cv_tracker):
    """
    Display complete table of all 35 training runs with visual formatting.
    """
    df = cv_tracker.get_summary_df()

    # Select key columns
    display_df = df[['model_name', 'fold', 'auc', 'recall_optimal',
                     'precision_optimal', 'f1_optimal', 'optimal_threshold',
                     'default_cost', 'optimal_cost', 'cost_savings']].copy()

    # Format columns
    display_df['auc'] = display_df['auc'].apply(lambda x: f"{x:.4f}")
    display_df['recall_optimal'] = display_df['recall_optimal'].apply(lambda x: f"{x:.3f}")
    display_df['precision_optimal'] = display_df['precision_optimal'].apply(lambda x: f"{x:.3f}")
    display_df['f1_optimal'] = display_df['f1_optimal'].apply(lambda x: f"{x:.3f}")
    display_df['optimal_threshold'] = display_df['optimal_threshold'].apply(lambda x: f"{x:.2f}")
    display_df['default_cost'] = display_df['default_cost'].apply(lambda x: f"${x:.2f}")
    display_df['optimal_cost'] = display_df['optimal_cost'].apply(lambda x: f"${x:.2f}")
    display_df['cost_savings'] = display_df['cost_savings'].apply(lambda x: f"${x:.2f}")

    # Rename columns for display
    display_df.columns = ['Model', 'Fold', 'AUC', 'Recall@τ*', 'Prec@τ*',
                         'F1@τ*', 'τ*', 'Cost@0.5', 'Cost@τ*', 'Savings']

    print("=" * 120)
    print("COMPLETE 35-RUN CROSS-VALIDATION RESULTS (7 Models × 5 Folds)")
    print("=" * 120)

    # Use pandas styling for visual emphasis
    styled = display_df.style.background_gradient(
        subset=['AUC', 'Recall@τ*', 'F1@τ*'], cmap='Greens'
    ).background_gradient(
        subset=['Cost@τ*'], cmap='Reds_r'
    ).set_properties(**{'text-align': 'center'})

    display(styled)

    # Save to CSV
    df.to_csv('cv_results_35_runs_detailed.csv', index=False)
    print("\n✓ Saved to: cv_results_35_runs_detailed.csv")


def plot_model_fold_heatmap(cv_tracker, metric='optimal_cost'):
    """
    Heatmap showing metric values across models (rows) × folds (columns).
    """
    df = cv_tracker.get_summary_df()

    models = sorted(df['model_name'].unique())
    folds = sorted(df['fold'].unique())

    # Create data matrix
    data_matrix = np.zeros((len(models), len(folds)))

    for i, model in enumerate(models):
        for j, fold in enumerate(folds):
            value = df[(df['model_name'] == model) & (df['fold'] == fold)][metric].values
            if len(value) > 0:
                data_matrix[i, j] = value[0]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = 'Reds_r' if 'cost' in metric else 'Greens'
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(folds)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_yticklabels([m.replace('Model ', 'M') for m in models])

    # Add values in cells
    for i in range(len(models)):
        for j in range(len(folds)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom")

    ax.set_title(f'{metric.replace("_", " ").title()} - Model × Fold Heatmap',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_per_fold_roc_overlay(fold_results, model_name):
    """
    Overlay ROC curves for all 5 folds with mean ROC and confidence interval.
    Reuses fold-level data from CV results.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Plot each fold
    for i, fold in enumerate(fold_results):
        fpr = fold['fpr']
        tpr = fold['tpr']
        auc_score = fold['auc']

        ax.plot(fpr, tpr, alpha=0.3, linewidth=1,
                label=f'Fold {i+1} (AUC={auc_score:.3f})')

        # Interpolate for mean calculation
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_score)

    # Chance line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b', linewidth=3,
            label=f'Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})')

    # Confidence interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                    color='grey', alpha=0.2, label='±1 std')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curves Across 5 Folds',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cost_curves_all_models(all_model_results):
    """
    Grid of cost curves showing threshold optimization for all models.
    Reuses existing plot_cost_curve function.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (model_name, fold_results) in enumerate(all_model_results.items()):
        ax = axes[idx]

        # Aggregate cost curves across folds
        all_cost_curves = [fold['cost_curve'] for fold in fold_results]

        # Calculate mean and std
        thresholds = all_cost_curves[0]['threshold'].values
        cost_values = np.array([curve['expected_cost'].values for curve in all_cost_curves])
        mean_costs = cost_values.mean(axis=0)
        std_costs = cost_values.std(axis=0)

        # Plot mean cost curve
        ax.plot(thresholds, mean_costs, linewidth=2, color='blue', label='Mean Cost')
        ax.fill_between(thresholds, mean_costs - std_costs, mean_costs + std_costs,
                        alpha=0.2, color='blue', label='±1 std')

        # Mark optimal threshold
        optimal_idx = np.argmin(mean_costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_cost = mean_costs[optimal_idx]

        ax.scatter([optimal_threshold], [optimal_cost],
                  color='red', s=100, zorder=5, marker='*',
                  label=f'τ*={optimal_threshold:.2f}')

        # Mark default threshold
        default_idx = np.argmin(np.abs(thresholds - 0.5))
        default_cost = mean_costs[default_idx]
        ax.scatter([0.5], [default_cost],
                  color='orange', s=100, zorder=5, marker='o',
                  label=f'τ=0.5')

        ax.set_xlabel('Threshold', fontsize=10)
        ax.set_ylabel('Expected Cost ($)', fontsize=10)
        ax.set_title(model_name.replace('Model ', 'M'), fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Remove extra subplot
    fig.delaxes(axes[7])

    fig.suptitle('Cost-Aware Threshold Optimization Across All 7 Models',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


print("✅ Enhanced visualization suite loaded")
```

**Integration Point**: This goes in **Section 6.5** (new section after model training)

---

### Step 4: Model Training Loop (Train All 7 Models)

**Implementation**:

```python
# ===============================================
# SECTION 7: TRAIN ALL 7 MODELS WITH ENHANCED CV
# ===============================================

# Dictionary to store all model results
all_model_results = {}

# Model 0: Baseline SGD
print("\n" + "="*70)
print("MODEL 0: BASELINE SGD")
print("="*70)
model_0_results = train_model_with_enhanced_cv(
    create_model_0,
    "Model 0 (Baseline SGD)",
    use_class_weights=False
)
all_model_results["Model 0 (Baseline SGD)"] = model_0_results

# Model 1: Deeper SGD
print("\n" + "="*70)
print("MODEL 1: DEEPER ARCHITECTURE WITH SGD")
print("="*70)
model_1_results = train_model_with_enhanced_cv(
    create_model_1,
    "Model 1 (Deep SGD)",
    use_class_weights=False
)
all_model_results["Model 1 (Deep SGD)"] = model_1_results

# Model 2: Adam Optimizer
print("\n" + "="*70)
print("MODEL 2: COMPACT ARCHITECTURE WITH ADAM")
print("="*70)
model_2_results = train_model_with_enhanced_cv(
    create_model_2,
    "Model 2 (Adam Compact)",
    use_class_weights=False
)
all_model_results["Model 2 (Adam Compact)"] = model_2_results

# Model 3: Dropout
print("\n" + "="*70)
print("MODEL 3: DROPOUT REGULARIZATION")
print("="*70)
model_3_results = train_model_with_enhanced_cv(
    create_model_3,
    "Model 3 (Adam + Dropout)",
    use_class_weights=False
)
all_model_results["Model 3 (Adam + Dropout)"] = model_3_results

# Model 4: Class Weights
print("\n" + "="*70)
print("MODEL 4: CLASS WEIGHTS")
print("="*70)
model_4_results = train_model_with_enhanced_cv(
    create_model_4,
    "Model 4 (Adam + Class Weights)",
    use_class_weights=True
)
all_model_results["Model 4 (Adam + Class Weights)"] = model_4_results

# Model 5: Dropout + Class Weights
print("\n" + "="*70)
print("MODEL 5: DROPOUT + CLASS WEIGHTS")
print("="*70)
model_5_results = train_model_with_enhanced_cv(
    create_model_5,
    "Model 5 (Dropout + Class Weights)",
    use_class_weights=True
)
all_model_results["Model 5 (Dropout + Class Weights)"] = model_5_results

# Model 6: L2 + Class Weights
print("\n" + "="*70)
print("MODEL 6: L2 REGULARIZATION + CLASS WEIGHTS")
print("="*70)
model_6_results = train_model_with_enhanced_cv(
    create_model_6,
    "Model 6 (L2 + Class Weights)",
    use_class_weights=True
)
all_model_results["Model 6 (L2 + Class Weights)"] = model_6_results

print("\n" + "="*70)
print("✅ ALL 35 TRAINING RUNS COMPLETE")
print("   (7 models × 5 folds = 35 runs)")
print("="*70)
```

---

### Step 5: Comprehensive Visualization Section

**Implementation**:

```python
# ===============================================
# SECTION 7.5: COMPREHENSIVE RESULTS VISUALIZATION
# ===============================================

print("=" * 70)
print("VISUALIZING 35 TRAINING RUNS")
print("=" * 70)

# 1. Box plots showing distribution across folds
print("\n📊 Generating box plots...")
plot_cv_performance_boxplots(cv_tracker)

# 2. Detailed 35-run table
print("\n📋 Generating detailed results table...")
plot_detailed_35_runs_table(cv_tracker)

# 3. Heatmaps (Model × Fold)
print("\n🔥 Generating heatmaps...")
plot_model_fold_heatmap(cv_tracker, metric='optimal_cost')
plot_model_fold_heatmap(cv_tracker, metric='auc')
plot_model_fold_heatmap(cv_tracker, metric='recall_optimal')

# 4. Cost curves for all models
print("\n💰 Generating cost curves...")
plot_cost_curves_all_models(all_model_results)

# 5. Per-fold ROC overlay for best model
print("\n📈 Generating per-fold ROC curves...")
# Find best model (lowest mean optimal cost)
model_summaries = cv_tracker.get_all_model_summaries()
best_model_name = model_summaries.loc[model_summaries['mean_optimal_cost'].idxmin(), 'model_name']
best_model_results = all_model_results[best_model_name]

plot_per_fold_roc_overlay(best_model_results, best_model_name)

print("\n✅ All visualizations generated")
```

---

## Summary: What Changes Where

| Section | Current | Enhancement | Code Location |
|---------|---------|-------------|---------------|
| **Section 5** | Basic cost functions | **CostConfig class + utilities** | Replace cells |
| **Section 6** | Basic CV loop | **Enhanced CV with SMOTE + CVResultsTracker** | Replace function |
| **Section 6.5** | N/A | **NEW: Visualization suite** | Add new section |
| **Section 7** | Model training | **Use new enhanced CV function** | Update function calls |
| **Section 7.5** | N/A | **NEW: Comprehensive visualization section** | Add new section |
| **Section 8** | Model comparison | **Use cv_tracker.get_all_model_summaries()** | Update data source |

---

## Implementation Checklist

### Phase 1: Core Refactoring (30 min)
- [ ] Add CostConfig class to Section 5
- [ ] Consolidate cost utilities in Section 5
- [ ] Update Section 6 with enhanced CV function
- [ ] Add CVResultsTracker class

### Phase 2: Model Training (15 min)
- [ ] Update Section 7 model training calls
- [ ] Verify all 35 runs complete successfully
- [ ] Check cv_tracker has all results

### Phase 3: Visualizations (45 min)
- [ ] Add Section 6.5 with visualization functions
- [ ] Add Section 7.5 with visualization calls
- [ ] Generate all plots
- [ ] Verify output quality

### Phase 4: SMOTE Experiments (Optional, 30 min)
- [ ] Run with CostConfig.USE_SMOTE = False
- [ ] Run with CostConfig.USE_SMOTE = True
- [ ] Compare results
- [ ] Document findings

---

## Expected Outcomes

### Improved Transparency
- ✅ All 35 runs visible in detailed table
- ✅ Fold-to-fold variance shown in box plots
- ✅ Model×Fold patterns in heatmaps

### Enhanced Visualizations
- ✅ Per-fold ROC overlays with confidence bands
- ✅ Cost curves for all 7 models
- ✅ Box plots for metric distributions

### Advanced Features
- ✅ SMOTE integration with toggle
- ✅ Leak-safe preprocessing maintained
- ✅ Complete audit trail (35 runs saved to CSV)

### Business Value
- ✅ Cost-aware optimization clearly demonstrated
- ✅ Threshold sensitivity visualized
- ✅ Robust model selection with variance estimates

---

## Next Steps

**Ready to implement?** I can help you:

1. **Generate complete notebook cells** for each section
2. **Update existing sections** with new code
3. **Test integrations** to ensure everything works
4. **Add markdown narratives** explaining each enhancement

Which would you like to start with?
