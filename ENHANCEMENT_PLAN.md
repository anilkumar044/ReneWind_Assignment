# ReneWind Notebook Enhancement Plan

**Version**: 2.0 (Enhanced)
**Current Score**: 60/60 (100%)
**Enhancement Goal**: Exceed expectations with professional-grade visualizations and advanced features

---

## Executive Summary

This plan outlines strategic enhancements to transform the already-excellent notebook into a **showcase-quality portfolio piece** while maintaining 100% rubric compliance.

### Enhancement Categories

1. **Enriched Visualizations** (12 new plots)
2. **5-Fold CV Performance Showcase** (Transparency into 35 training runs)
3. **SMOTE Oversampling Integration** (Advanced imbalance handling)
4. **Cost-Aware Optimization Visuals** (Business alignment demonstration)
5. **Interactive Model Comparison** (Professional dashboards)

**Total New Code**: ~500 lines
**New Visualizations**: 12-15 publication-quality plots
**Rubric Impact**: Maintains 60/60, exceeds presentation expectations

---

## 1. Enriched Visualizations

### 1.1 Cross-Validation Performance Distribution (Box Plots)

**Purpose**: Show model stability across folds

**Current State**: Only mean ± std reported in text
**Enhanced State**: Box plots showing metric distributions

```python
# ===============================================
# CROSS-VALIDATION PERFORMANCE DISTRIBUTION
# ===============================================

def plot_cv_performance_distribution(cv_results_dict):
    """
    Visualize performance variability across 5 folds for all models.
    Shows: Recall, Precision, F1, AUC, Optimal Cost
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Distribution Across 5 Folds',
                 fontsize=16, fontweight='bold')

    metrics = ['recall_at_optimal', 'precision_at_optimal', 'f1_at_optimal',
               'auc', 'optimal_cost', 'optimal_threshold']
    metric_labels = ['Recall @ τ*', 'Precision @ τ*', 'F1 @ τ*',
                     'AUC', 'Optimal Cost ($)', 'Optimal Threshold']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]

        # Prepare data for box plot
        data = []
        labels = []
        for model_name, results in cv_results_dict.items():
            data.append([fold[metric] for fold in results])
            labels.append(model_name.replace('Model ', 'M'))

        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        notch=True, showmeans=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    print("📊 Box plots show:")
    print("   - Median (orange line)")
    print("   - Mean (green triangle)")
    print("   - IQR (box)")
    print("   - Whiskers (1.5×IQR)")
    print("   - Notches (95% CI around median)")

# Usage after training all models
plot_cv_performance_distribution(all_cv_results)
```

**Value**: Shows model stability, identifies high-variance models

---

### 1.2 Per-Fold ROC Curves (All Folds Overlaid)

**Purpose**: Demonstrate consistency of discrimination ability

```python
# ===============================================
# PER-FOLD ROC CURVES WITH CONFIDENCE BANDS
# ===============================================

def plot_fold_roc_curves(cv_results, model_name):
    """
    Plot ROC curve for each fold with mean ROC and confidence interval.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Plot each fold
    for i, fold_result in enumerate(cv_results):
        fpr = fold_result['fpr']
        tpr = fold_result['tpr']
        auc_score = fold_result['auc']

        ax.plot(fpr, tpr, alpha=0.3, linewidth=1,
                label=f'Fold {i+1} (AUC={auc_score:.3f})')

        # Interpolate for mean calculation
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_score)

    # Plot chance line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')

    # Plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b', linewidth=3,
            label=f'Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})')

    # Add confidence interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                    color='grey', alpha=0.2, label='±1 std. dev.')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curves Across 5 Folds',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# Usage
plot_fold_roc_curves(model_5_cv_results, 'Model 5 (Best Model)')
```

**Value**: Proves model consistency, shows discrimination power stability

---

### 1.3 Cost Curve Visualization (All Models)

**Purpose**: Show cost-aware optimization across threshold space

```python
# ===============================================
# COST CURVES ACROSS THRESHOLD SPACE
# ===============================================

def plot_cost_curves_comparison(cv_results_dict):
    """
    Plot cost curves for all models to visualize threshold optimization.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (model_name, results) in enumerate(cv_results_dict.items()):
        ax = axes[idx]

        # Aggregate cost curves across folds
        thresholds = np.linspace(0.05, 0.95, 91)
        fold_costs = []

        for fold in results:
            # Assuming each fold stores cost_curve data
            fold_costs.append(fold['cost_curve'])

        fold_costs = np.array(fold_costs)
        mean_costs = fold_costs.mean(axis=0)
        std_costs = fold_costs.std(axis=0)

        # Plot mean cost curve
        ax.plot(thresholds, mean_costs, linewidth=2, label='Mean Cost')
        ax.fill_between(thresholds,
                        mean_costs - std_costs,
                        mean_costs + std_costs,
                        alpha=0.2, label='±1 std')

        # Mark optimal threshold
        optimal_idx = np.argmin(mean_costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_cost = mean_costs[optimal_idx]

        ax.scatter([optimal_threshold], [optimal_cost],
                  color='red', s=100, zorder=5, marker='*',
                  label=f'Optimal τ*={optimal_threshold:.2f}')

        # Mark default threshold
        default_idx = np.argmin(np.abs(thresholds - 0.5))
        default_cost = mean_costs[default_idx]
        ax.scatter([0.5], [default_cost],
                  color='orange', s=100, zorder=5, marker='o',
                  label=f'Default τ=0.5')

        ax.set_xlabel('Decision Threshold', fontsize=10)
        ax.set_ylabel('Expected Cost ($)', fontsize=10)
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Remove extra subplot
    fig.delaxes(axes[7])

    fig.suptitle('Cost-Aware Threshold Optimization Across All Models',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

**Value**: Visualizes business-aligned optimization, shows cost savings

---

### 1.4 Training History Comparison (All Models)

**Purpose**: Show convergence patterns and training dynamics

```python
# ===============================================
# TRAINING HISTORY VISUALIZATION
# ===============================================

def plot_training_histories(history_dict):
    """
    Compare training dynamics across all 7 models.
    Shows: Loss curves, validation loss, convergence speed
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (model_name, history) in enumerate(history_dict.items()):
        ax = axes[idx]

        # Plot training and validation loss (averaged across folds)
        epochs = range(1, len(history['loss']) + 1)

        ax.plot(epochs, history['loss'], 'b-', linewidth=2,
                label='Training Loss')
        ax.plot(epochs, history['val_loss'], 'r-', linewidth=2,
                label='Validation Loss')

        # Mark early stopping point if applicable
        if 'stopped_epoch' in history:
            ax.axvline(x=history['stopped_epoch'],
                      color='green', linestyle='--', alpha=0.7,
                      label=f'Early Stop (epoch {history["stopped_epoch"]})')

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=10)
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization

    fig.delaxes(axes[7])
    fig.suptitle('Training Convergence Across All Models (35 Runs Total)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

**Value**: Shows training efficiency, identifies overfitting, demonstrates proper early stopping

---

### 1.5 Radar Chart (Model Comparison)

**Purpose**: Multi-dimensional performance comparison

```python
# ===============================================
# RADAR CHART - MULTI-METRIC COMPARISON
# ===============================================

def plot_model_comparison_radar(comparison_df):
    """
    Create radar chart comparing models across multiple metrics.
    """
    from math import pi

    # Normalize metrics to 0-1 scale for fair comparison
    metrics = ['mean_recall_optimal', 'mean_precision_optimal',
               'mean_f1_optimal', 'mean_auc']
    labels = ['Recall@τ*', 'Precision@τ*', 'F1@τ*', 'AUC']

    # Normalize (higher is better for all these metrics)
    normalized_df = comparison_df[metrics].copy()
    for col in metrics:
        normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / \
                             (normalized_df[col].max() - normalized_df[col].min())

    # Add cost metric (inverted - lower is better)
    normalized_df['cost_score'] = 1 - (comparison_df['mean_optimal_cost'] -
                                       comparison_df['mean_optimal_cost'].min()) / \
                                      (comparison_df['mean_optimal_cost'].max() -
                                       comparison_df['mean_optimal_cost'].min())

    metrics.append('cost_score')
    labels.append('Cost Score')

    # Number of variables
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(comparison_df)))

    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        values = normalized_df.iloc[idx][metrics].values.tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2,
                label=row['model_name'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Multi-Metric Model Performance Comparison',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()
```

**Value**: Intuitive multi-dimensional comparison, great for presentations

---

## 2. 5-Fold CV Performance Showcase

### 2.1 Comprehensive Fold-by-Fold Results Table

**Purpose**: Full transparency into 35 training runs (7 models × 5 folds)

```python
# ===============================================
# DETAILED FOLD-BY-FOLD RESULTS TABLE
# ===============================================

def create_detailed_cv_results_table(all_cv_results):
    """
    Create comprehensive table showing all 35 training runs.
    """
    rows = []

    for model_name, fold_results in all_cv_results.items():
        for fold_idx, fold_data in enumerate(fold_results):
            rows.append({
                'Model': model_name,
                'Fold': fold_idx + 1,
                'AUC': f"{fold_data['auc']:.4f}",
                'Recall@0.5': f"{fold_data['recall_default']:.3f}",
                'Recall@τ*': f"{fold_data['recall_optimal']:.3f}",
                'Precision@τ*': f"{fold_data['precision_optimal']:.3f}",
                'F1@τ*': f"{fold_data['f1_optimal']:.3f}",
                'τ*': f"{fold_data['optimal_threshold']:.2f}",
                'Cost@0.5': f"${fold_data['default_cost']:.2f}",
                'Cost@τ*': f"${fold_data['optimal_cost']:.2f}",
                'Savings': f"${fold_data['default_cost'] - fold_data['optimal_cost']:.2f}"
            })

    df_detailed = pd.DataFrame(rows)

    print("=" * 100)
    print("COMPLETE 35-RUN CROSS-VALIDATION RESULTS (7 Models × 5 Folds)")
    print("=" * 100)
    display(df_detailed.style.background_gradient(
        subset=['AUC', 'Recall@τ*', 'F1@τ*'], cmap='Greens'
    ).background_gradient(
        subset=['Cost@τ*'], cmap='Reds_r'
    ))

    return df_detailed

# Generate table
detailed_results = create_detailed_cv_results_table(all_cv_results)

# Save to CSV for reference
detailed_results.to_csv('cv_detailed_results_35_runs.csv', index=False)
print("\n✓ Saved to: cv_detailed_results_35_runs.csv")
```

**Value**: Complete transparency, reproducibility, audit trail

---

### 2.2 Heatmap: Performance Across Models × Folds

**Purpose**: Visual pattern recognition in cross-validation

```python
# ===============================================
# HEATMAP: MODEL × FOLD PERFORMANCE
# ===============================================

def plot_cv_heatmap(all_cv_results, metric='optimal_cost'):
    """
    Heatmap showing metric values across all models and folds.
    """
    # Prepare data matrix
    models = list(all_cv_results.keys())
    folds = range(1, 6)

    data_matrix = np.zeros((len(models), 5))

    for i, model_name in enumerate(models):
        for j in range(5):
            data_matrix[i, j] = all_cv_results[model_name][j][metric]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use reversed colormap for cost (lower is better)
    cmap = 'Reds_r' if 'cost' in metric else 'Greens'

    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_yticklabels([m.replace('Model ', 'M') for m in models])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add values in cells
    for i in range(len(models)):
        for j in range(5):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom")

    ax.set_title(f'{metric.replace("_", " ").title()} Across Models and Folds',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

# Usage
plot_cv_heatmap(all_cv_results, metric='optimal_cost')
plot_cv_heatmap(all_cv_results, metric='auc')
plot_cv_heatmap(all_cv_results, metric='recall_optimal')
```

**Value**: Identifies patterns, shows fold-to-fold consistency

---

## 3. SMOTE Oversampling Integration

### 3.1 Configuration System

**Purpose**: Toggle SMOTE on/off for comparison

```python
# ===============================================
# CONFIGURATION - SMOTE TOGGLE
# ===============================================

class Config:
    """
    Central configuration for preprocessing options.
    """
    # SMOTE Configuration
    USE_SMOTE = True  # Toggle SMOTE oversampling
    SMOTE_RATIO = 0.5  # Target ratio after SMOTE (0.5 = 2:1 majority:minority)
    SMOTE_K_NEIGHBORS = 5  # Number of nearest neighbors for SMOTE

    # Cross-Validation
    N_SPLITS = 5
    RANDOM_STATE = 42

    # Cost Structure
    COST_FN = 100.0
    COST_TP = 30.0
    COST_FP = 10.0
    COST_TN = 0.0

print("=" * 70)
print("CONFIGURATION")
print("=" * 70)
print(f"SMOTE Oversampling: {'ENABLED' if Config.USE_SMOTE else 'DISABLED'}")
if Config.USE_SMOTE:
    print(f"  - Target ratio: {Config.SMOTE_RATIO:.2f}")
    print(f"  - K-neighbors: {Config.SMOTE_K_NEIGHBORS}")
print(f"Cross-Validation Folds: {Config.N_SPLITS}")
print(f"Random Seed: {Config.RANDOM_STATE}")
```

---

### 3.2 Leak-Safe SMOTE in CV Loop

**Purpose**: Apply SMOTE per fold (never on validation data)

```python
# ===============================================
# LEAK-SAFE SMOTE INTEGRATION
# ===============================================

from imblearn.over_sampling import SMOTE

def train_with_smote_cv(model_fn, model_name, use_class_weights=False):
    """
    Train model with optional SMOTE oversampling (applied per fold).
    """
    skf = StratifiedKFold(n_splits=Config.N_SPLITS,
                          shuffle=True,
                          random_state=Config.RANDOM_STATE)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/{Config.N_SPLITS}")
        print(f"{'='*70}")

        # Split data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Preprocessing (leak-safe)
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_fold = imputer.fit_transform(X_train_fold)
        X_val_fold = imputer.transform(X_val_fold)

        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        # SMOTE (ONLY on training data)
        if Config.USE_SMOTE:
            print(f"\n📊 Before SMOTE:")
            print(f"   Class 0: {(y_train_fold == 0).sum()}")
            print(f"   Class 1: {(y_train_fold == 1).sum()}")
            print(f"   Ratio: {(y_train_fold == 0).sum() / (y_train_fold == 1).sum():.2f}:1")

            smote = SMOTE(sampling_strategy=Config.SMOTE_RATIO,
                         k_neighbors=Config.SMOTE_K_NEIGHBORS,
                         random_state=Config.RANDOM_STATE)

            X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

            print(f"\n📊 After SMOTE:")
            print(f"   Class 0: {(y_train_fold == 0).sum()}")
            print(f"   Class 1: {(y_train_fold == 1).sum()}")
            print(f"   Ratio: {(y_train_fold == 0).sum() / (y_train_fold == 1).sum():.2f}:1")

        # Compute class weights (if enabled)
        class_weights = None
        if use_class_weights:
            weights_array = compute_class_weight('balanced',
                                                 classes=np.unique(y_train_fold),
                                                 y=y_train_fold)
            class_weights = {0: weights_array[0], 1: weights_array[1]}

        # Train model
        model = model_fn()

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            class_weight=class_weights,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Evaluate and store results
        # ... (rest of evaluation code)

        fold_results.append(fold_data)

    return fold_results
```

**Value**: Advanced imbalance handling, maintains leak-safety

---

### 3.3 SMOTE vs No-SMOTE Comparison

**Purpose**: Show impact of SMOTE on model performance

```python
# ===============================================
# SMOTE IMPACT ANALYSIS
# ===============================================

def compare_smote_impact():
    """
    Train same model with and without SMOTE, compare results.
    """
    print("=" * 70)
    print("SMOTE IMPACT ANALYSIS - MODEL 5 (BEST MODEL)")
    print("=" * 70)

    # Train without SMOTE
    Config.USE_SMOTE = False
    results_no_smote = train_with_smote_cv(create_model_5, "Model 5",
                                            use_class_weights=True)

    # Train with SMOTE
    Config.USE_SMOTE = True
    results_with_smote = train_with_smote_cv(create_model_5, "Model 5",
                                             use_class_weights=True)

    # Compare
    comparison = pd.DataFrame({
        'Configuration': ['Without SMOTE', 'With SMOTE'],
        'Mean Recall@τ*': [
            np.mean([f['recall_optimal'] for f in results_no_smote]),
            np.mean([f['recall_optimal'] for f in results_with_smote])
        ],
        'Mean Precision@τ*': [
            np.mean([f['precision_optimal'] for f in results_no_smote]),
            np.mean([f['precision_optimal'] for f in results_with_smote])
        ],
        'Mean F1@τ*': [
            np.mean([f['f1_optimal'] for f in results_no_smote]),
            np.mean([f['f1_optimal'] for f in results_with_smote])
        ],
        'Mean Cost@τ*': [
            np.mean([f['optimal_cost'] for f in results_no_smote]),
            np.mean([f['optimal_cost'] for f in results_with_smote])
        ]
    })

    display(comparison.style.background_gradient(cmap='RdYlGn', axis=0))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall comparison
    axes[0].bar(['No SMOTE', 'With SMOTE'],
                comparison['Mean Recall@τ*'],
                color=['#ff9999', '#66b3ff'])
    axes[0].set_ylabel('Recall @ Optimal Threshold')
    axes[0].set_title('Recall Comparison')
    axes[0].grid(axis='y', alpha=0.3)

    # Cost comparison
    axes[1].bar(['No SMOTE', 'With SMOTE'],
                comparison['Mean Cost@τ*'],
                color=['#ff9999', '#66b3ff'])
    axes[1].set_ylabel('Expected Cost ($)')
    axes[1].set_title('Cost Comparison')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return comparison

# Run comparison
smote_comparison = compare_smote_impact()
```

**Value**: Demonstrates advanced technique knowledge, shows experimental rigor

---

## 4. Enhanced Cost-Aware Optimization Visuals

### 4.1 3D Cost Surface Plot

**Purpose**: Show cost as function of threshold and model choice

```python
# ===============================================
# 3D COST SURFACE VISUALIZATION
# ===============================================

from mpl_toolkits.mplot3d import Axes3D

def plot_cost_surface_3d(all_cv_results):
    """
    3D surface plot: Models × Thresholds × Cost
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data
    models = list(all_cv_results.keys())
    thresholds = np.linspace(0.05, 0.95, 91)

    X, Y = np.meshgrid(range(len(models)), thresholds)
    Z = np.zeros_like(X, dtype=float)

    for i, model_name in enumerate(models):
        # Average cost curve across folds
        cost_curves = [fold['cost_curve'] for fold in all_cv_results[model_name]]
        mean_cost_curve = np.mean(cost_curves, axis=0)
        Z[:, i] = mean_cost_curve

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8,
                          linewidth=0, antialiased=True)

    # Mark optimal points
    for i, model_name in enumerate(models):
        optimal_threshold = np.mean([f['optimal_threshold']
                                     for f in all_cv_results[model_name]])
        optimal_cost = np.mean([f['optimal_cost']
                               for f in all_cv_results[model_name]])
        ax.scatter([i], [optimal_threshold], [optimal_cost],
                  color='red', s=100, marker='*', zorder=5)

    ax.set_xlabel('Model Index', fontsize=12)
    ax.set_ylabel('Decision Threshold', fontsize=12)
    ax.set_zlabel('Expected Cost ($)', fontsize=12)
    ax.set_title('Cost Surface Across Models and Thresholds',
                 fontsize=14, fontweight='bold')

    # Set x-ticks to model names
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('Model ', 'M') for m in models],
                       rotation=45, ha='right')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()

plot_cost_surface_3d(all_cv_results)
```

**Value**: Stunning visualization, shows optimization landscape

---

### 4.2 Cost Sensitivity Analysis Heatmap

**Purpose**: Show robustness to cost parameter changes

```python
# ===============================================
# COST SENSITIVITY HEATMAP
# ===============================================

def plot_cost_sensitivity_heatmap(model_results, base_costs):
    """
    Show how optimal threshold changes with cost parameter variations.
    """
    # Test different cost scenarios
    fn_range = np.arange(80, 121, 10)  # ±20% from $100
    fp_range = np.arange(8, 13, 1)     # ±20% from $10

    threshold_matrix = np.zeros((len(fn_range), len(fp_range)))
    cost_matrix = np.zeros((len(fn_range), len(fp_range)))

    for i, fn_cost in enumerate(fn_range):
        for j, fp_cost in enumerate(fp_range):
            # Recalculate optimal threshold with new costs
            test_costs = {
                'FN': fn_cost,
                'TP': base_costs['TP'],
                'FP': fp_cost,
                'TN': base_costs['TN']
            }

            # Use validation predictions to find new optimal threshold
            optimal_threshold, optimal_cost = optimize_threshold(
                y_val, y_pred_proba, test_costs
            )

            threshold_matrix[i, j] = optimal_threshold
            cost_matrix[i, j] = optimal_cost

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Optimal threshold heatmap
    im1 = axes[0].imshow(threshold_matrix, cmap='viridis', aspect='auto')
    axes[0].set_xticks(range(len(fp_range)))
    axes[0].set_yticks(range(len(fn_range)))
    axes[0].set_xticklabels([f'${x}' for x in fp_range])
    axes[0].set_yticklabels([f'${x}' for x in fn_range])
    axes[0].set_xlabel('FP Cost (Inspection)', fontsize=12)
    axes[0].set_ylabel('FN Cost (Replacement)', fontsize=12)
    axes[0].set_title('Optimal Threshold Under Cost Variations', fontweight='bold')

    # Add values
    for i in range(len(fn_range)):
        for j in range(len(fp_range)):
            axes[0].text(j, i, f'{threshold_matrix[i, j]:.2f}',
                        ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im1, ax=axes[0], label='Optimal Threshold')

    # Expected cost heatmap
    im2 = axes[1].imshow(cost_matrix, cmap='Reds', aspect='auto')
    axes[1].set_xticks(range(len(fp_range)))
    axes[1].set_yticks(range(len(fn_range)))
    axes[1].set_xticklabels([f'${x}' for x in fp_range])
    axes[1].set_yticklabels([f'${x}' for x in fn_range])
    axes[1].set_xlabel('FP Cost (Inspection)', fontsize=12)
    axes[1].set_ylabel('FN Cost (Replacement)', fontsize=12)
    axes[1].set_title('Expected Cost Under Cost Variations', fontweight='bold')

    # Add values
    for i in range(len(fn_range)):
        for j in range(len(fp_range)):
            axes[1].text(j, i, f'${cost_matrix[i, j]:.1f}',
                        ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im2, ax=axes[1], label='Expected Cost ($)')

    plt.tight_layout()
    plt.show()

    print("\n📊 Sensitivity Analysis Shows:")
    print(f"   Threshold range: {threshold_matrix.min():.2f} - {threshold_matrix.max():.2f}")
    print(f"   Cost range: ${cost_matrix.min():.2f} - ${cost_matrix.max():.2f}")
    print(f"   Threshold stability: ±{threshold_matrix.std():.3f}")
```

**Value**: Proves robustness, answers "what-if" questions

---

## 5. Implementation Roadmap

### Phase 1: Quick Wins (30 minutes)
1. Add box plot for CV performance distribution
2. Add detailed fold-by-fold results table
3. Add training history comparison plots

### Phase 2: Core Enhancements (1-2 hours)
4. Implement per-fold ROC curves with confidence bands
5. Add cost curve comparison visualization
6. Create radar chart for model comparison
7. Add heatmap for model×fold performance

### Phase 3: Advanced Features (2-3 hours)
8. Integrate SMOTE with configuration system
9. Implement SMOTE comparison analysis
10. Create 3D cost surface plot
11. Build cost sensitivity heatmap

### Phase 4: Polish & Documentation (30 minutes)
12. Add markdown explanations for each visualization
13. Update business insights with new findings
14. Create visualization index/table of contents

---

## 6. Expected Impact

### Rubric Score
- **Before**: 60/60 (100%)
- **After**: 60/60 (100%) + **Exceeds Expectations**

### Presentation Quality Boost
- Professional-grade visualizations
- Publication-ready figures
- Dashboard-style comparisons
- Complete transparency (35 runs)

### Portfolio Value
- Demonstrates advanced ML techniques (SMOTE, CV, threshold optimization)
- Shows data visualization mastery
- Proves attention to detail and rigor
- Ready for stakeholder presentations

---

## 7. Code Organization

### Recommended Structure

```python
# Section 6.5: ENHANCED CROSS-VALIDATION VISUALIZATIONS
# (Insert after initial CV results)

# 6.5.1 - Box Plots
plot_cv_performance_distribution(all_cv_results)

# 6.5.2 - Detailed Results Table
detailed_results = create_detailed_cv_results_table(all_cv_results)

# 6.5.3 - Heatmaps
plot_cv_heatmap(all_cv_results, 'optimal_cost')
plot_cv_heatmap(all_cv_results, 'auc')

# 6.5.4 - Training Histories
plot_training_histories(training_histories)

# Section 7.5: ENHANCED MODEL COMPARISON
# (Insert after Section 8)

# 7.5.1 - Radar Chart
plot_model_comparison_radar(comparison_df)

# 7.5.2 - Cost Curves
plot_cost_curves_comparison(all_cv_results)

# 7.5.3 - Per-Fold ROC
plot_fold_roc_curves(best_model_cv_results, 'Best Model')

# Section 9.5: ADVANCED VISUALIZATIONS
# (Insert after test evaluation)

# 9.5.1 - 3D Cost Surface
plot_cost_surface_3d(all_cv_results)

# 9.5.2 - Cost Sensitivity
plot_cost_sensitivity_heatmap(final_model_results, BASE_COSTS)

# Section 11.5: SMOTE ANALYSIS (OPTIONAL)
# (Insert before conclusions)

# 11.5.1 - SMOTE Comparison
smote_comparison = compare_smote_impact()
```

---

## 8. Next Steps

1. **Review this plan** - Prioritize which enhancements you want
2. **Choose implementation level**:
   - **Minimal**: Items 1-3 (box plots, table, training history)
   - **Standard**: Items 1-7 (all visualization enhancements)
   - **Complete**: All items including SMOTE

3. **I can help implement**:
   - Generate complete code for each section
   - Update existing cells in notebook
   - Test visualizations
   - Add markdown documentation

**Ready to implement?** Let me know which enhancements you'd like to start with!
