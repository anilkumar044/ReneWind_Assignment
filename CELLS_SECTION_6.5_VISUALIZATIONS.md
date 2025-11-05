# Section 6.5: Enhanced Visualization Suite - Notebook Cells

**Installation Location**: NEW SECTION - Insert after Section 6, before Section 7

---

## Cell 6.5.1 [MARKDOWN]

```markdown
# **Section 6.5: Enhanced Visualization Suite**

This section provides comprehensive visualizations to showcase the 35 training runs:

1. **Box Plots**: Performance distribution across 5 folds for all models
2. **Detailed Table**: Complete 35-run results with formatting
3. **Heatmaps**: Model × Fold performance patterns
4. **Cost Curves**: Threshold optimization visualized for all models
5. **ROC Overlays**: Per-fold ROC curves with confidence bands

These visualizations provide full transparency into model training and validation.
```

---

## Cell 6.5.2 [CODE] - Box Plot Visualization

```python
# ===============================================
# BOX PLOTS - CV PERFORMANCE DISTRIBUTION
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
        
        models = sorted(df['model_name'].unique())
        data = [df[df['model_name'] == m][metric].values for m in models]
        labels = [m.replace('Model ', 'M').split('(')[0].strip() for m in models]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       notch=True, showmeans=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Remove extra subplot
    if len(axes) > len(metrics):
        fig.delaxes(axes[-1])
    
    fig.suptitle('Model Performance Distribution Across 5 Folds (35 Training Runs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("✅ Box plot function loaded")
```

---

## Cell 6.5.3 [CODE] - Detailed Results Table

```python
# ===============================================
# DETAILED 35-RUN RESULTS TABLE
# ===============================================

def plot_detailed_35_runs_table(cv_tracker):
    """
    Display complete table of all 35 training runs with visual formatting.
    """
    df = cv_tracker.get_summary_df()
    
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
    
    display_df.columns = ['Model', 'Fold', 'AUC', 'Recall@τ*', 'Prec@τ*',
                         'F1@τ*', 'τ*', 'Cost@0.5', 'Cost@τ*', 'Savings']
    
    print("=" * 120)
    print("COMPLETE 35-RUN CROSS-VALIDATION RESULTS (7 Models × 5 Folds)")
    print("=" * 120)
    
    display(display_df)
    
    # Save to CSV
    df.to_csv('cv_results_35_runs_detailed.csv', index=False)
    print("\n✓ Saved to: cv_results_35_runs_detailed.csv")

print("✅ Detailed table function loaded")
```

---

## Cell 6.5.4 [CODE] - Heatmap Visualization

```python
# ===============================================
# MODEL × FOLD HEATMAP
# ===============================================

def plot_model_fold_heatmap(cv_tracker, metric='optimal_cost'):
    """
    Heatmap showing metric values across models (rows) × folds (columns).
    """
    df = cv_tracker.get_summary_df()
    
    models = sorted(df['model_name'].unique())
    folds = sorted(df['fold'].unique())
    
    data_matrix = np.zeros((len(models), len(folds)))
    
    for i, model in enumerate(models):
        for j, fold in enumerate(folds):
            value = df[(df['model_name'] == model) & (df['fold'] == fold)][metric].values
            if len(value) > 0:
                data_matrix[i, j] = value[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = 'Reds_r' if 'cost' in metric else 'Greens'
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')
    
    ax.set_xticks(np.arange(len(folds)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_yticklabels([m.replace('Model ', 'M').split('(')[0].strip() for m in models])
    
    for i in range(len(models)):
        for j in range(len(folds)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom")
    
    ax.set_title(f'{metric.replace("_", " ").title()} - Model × Fold Heatmap',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

print("✅ Heatmap function loaded")
```

---

## Cell 6.5.5 [CODE] - Cost Curves Comparison

```python
# ===============================================
# COST CURVES - ALL MODELS
# ===============================================

def plot_cost_curves_all_models(all_model_results):
    """
    Grid of cost curves showing threshold optimization for all models.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (model_name, fold_results) in enumerate(all_model_results.items()):
        ax = axes[idx]
        
        all_cost_curves = [fold['cost_curve'] for fold in fold_results]
        
        thresholds = all_cost_curves[0]['threshold'].values
        cost_values = np.array([curve['expected_cost'].values for curve in all_cost_curves])
        mean_costs = cost_values.mean(axis=0)
        std_costs = cost_values.std(axis=0)
        
        ax.plot(thresholds, mean_costs, linewidth=2, color='blue', label='Mean Cost')
        ax.fill_between(thresholds, mean_costs - std_costs, mean_costs + std_costs,
                        alpha=0.2, color='blue')
        
        optimal_idx = np.argmin(mean_costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_cost = mean_costs[optimal_idx]
        
        ax.scatter([optimal_threshold], [optimal_cost],
                  color='red', s=100, zorder=5, marker='*',
                  label=f'τ*={optimal_threshold:.2f}')
        
        default_idx = np.argmin(np.abs(thresholds - 0.5))
        default_cost = mean_costs[default_idx]
        ax.scatter([0.5], [default_cost],
                  color='orange', s=100, zorder=5, marker='o',
                  label='τ=0.5')
        
        ax.set_xlabel('Threshold', fontsize=10)
        ax.set_ylabel('Expected Cost ($)', fontsize=10)
        ax.set_title(model_name.replace('Model ', 'M').split('(')[0].strip(), 
                     fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    
    if len(axes) > len(all_model_results):
        fig.delaxes(axes[-1])
    
    fig.suptitle('Cost-Aware Threshold Optimization Across All 7 Models',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("✅ Cost curves function loaded")
```

---

## Cell 6.5.6 [CODE] - Per-Fold ROC Curves

```python
# ===============================================
# PER-FOLD ROC CURVES WITH CONFIDENCE BANDS
# ===============================================

def plot_per_fold_roc_overlay(fold_results, model_name):
    """
    Overlay ROC curves for all 5 folds with mean ROC and confidence interval.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, fold in enumerate(fold_results):
        fpr = fold['fpr']
        tpr = fold['tpr']
        auc_score = fold['auc']
        
        ax.plot(fpr, tpr, alpha=0.3, linewidth=1,
                label=f'Fold {i+1} (AUC={auc_score:.3f})')
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_score)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b', linewidth=3,
            label=f'Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})')
    
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

print("✅ Per-fold ROC function loaded")
```

---

## Cell 6.5.7 [MARKDOWN] - Summary

```markdown
## **Visualization Strategy**

These visualizations serve multiple purposes:

1. **Box Plots**: Show performance variability across folds, identifying stable vs. unstable models
2. **Detailed Table**: Provides complete audit trail of all 35 runs for reproducibility
3. **Heatmaps**: Reveal patterns in model-fold interactions (e.g., which folds are hardest)
4. **Cost Curves**: Demonstrate business-aligned optimization visually
5. **ROC Overlays**: Prove consistent discrimination ability across folds with confidence intervals

**Interpretation Tips**:
- Smaller boxes = more stable performance
- Notches overlap = no significant difference between models
- Confidence bands show uncertainty in ROC curve estimates
- Red star (τ*) shows optimal threshold vs. orange circle (default 0.5)
```

