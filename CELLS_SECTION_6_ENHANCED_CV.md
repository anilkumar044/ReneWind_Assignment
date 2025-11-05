# Section 6: Enhanced Cross-Validation Pipeline - Notebook Cells

**Installation Location**: Replace existing Section 6 cells

---

## Cell 6.1 [MARKDOWN]

```markdown
# **Section 6: Enhanced Cross-Validation Pipeline with SMOTE**

This section implements a production-grade training pipeline with:

1. **5-Fold Stratified Cross-Validation**: Maintains class distribution across folds
2. **Leak-Safe Preprocessing**: Imputation and scaling fitted only on training folds
3. **Optional SMOTE Oversampling**: Synthetic minority oversampling for class imbalance
4. **Comprehensive Tracking**: All 35 training runs (7 models × 5 folds) captured

## **SMOTE Integration**

**Challenge**: Dataset exhibits severe class imbalance (only 3.6% failures)

**Solution**: SMOTE generates synthetic failure examples by interpolating between k-nearest neighbors in feature space.

**Critical Detail**: SMOTE applied **only to training folds**, never to validation data.

**Configuration**: Toggle via `CostConfig.USE_SMOTE = True/False`
```

---

## Cell 6.2 [CODE] - Import SMOTE

```python
# ===============================================
# SMOTE IMPORT
# ===============================================

from imblearn.over_sampling import SMOTE

print("✅ SMOTE imported from imbalanced-learn")
print(f"   Status: {'ENABLED' if CostConfig.USE_SMOTE else 'DISABLED'}")
```

---

## Cell 6.3 [CODE] - CVResultsTracker Class

```python
# ===============================================
# CROSS-VALIDATION RESULTS TRACKER
# ===============================================

class CVResultsTracker:
    """
    Track all 35 training runs (7 models × 5 folds).
    Provides detailed metrics and aggregation capabilities.
    """
    
    def __init__(self):
        self.all_runs = []
    
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
        """Get aggregated statistics for one model."""
        model_runs = [r for r in self.all_runs if r['model_name'] == model_name]
        if not model_runs:
            return None
        
        return {
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
    
    def get_all_model_summaries(self):
        """Get summary for all models."""
        unique_models = sorted(list(set([r['model_name'] for r in self.all_runs])))
        summaries = [self.get_model_summary(m) for m in unique_models]
        return pd.DataFrame(summaries)
    
    def save_to_csv(self, filepath='cv_results_35_runs.csv'):
        """Save all runs to CSV."""
        df = self.get_summary_df()
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(df)} runs to: {filepath}")
    
    def get_best_model(self, criterion='mean_optimal_cost', minimize=True):
        """Identify best model by criterion."""
        summaries = self.get_all_model_summaries()
        best_idx = summaries[criterion].idxmin() if minimize else summaries[criterion].idxmax()
        return summaries.loc[best_idx, 'model_name'], summaries.loc[best_idx, criterion]
    
    def __len__(self):
        return len(self.all_runs)
    
    def __repr__(self):
        n_models = len(set([r['model_name'] for r in self.all_runs]))
        return f"CVResultsTracker({len(self)} runs, {n_models} models)"

# Initialize global tracker
cv_tracker = CVResultsTracker()
print(f"✅ CVResultsTracker initialized: {cv_tracker}")
```

---

## Cell 6.4 [CODE] - Enhanced CV Training Function (Part 1/2)

```python
# ===============================================
# ENHANCED CV TRAINING FUNCTION
# ===============================================

def train_model_with_enhanced_cv(model_fn, model_name, use_class_weights=False, verbose=1):
    """
    Train model with enhanced 5-fold cross-validation.
    
    Features:
    - Stratified K-Fold CV
    - Leak-safe preprocessing per fold
    - Optional SMOTE (training only)
    - Cost-aware threshold optimization
    - Comprehensive metric tracking
    
    Parameters:
    -----------
    model_fn : callable
        Function returning compiled Keras model
    model_name : str
        Descriptive name for tracking
    use_class_weights : bool
        Use computed class weights during training
    verbose : int
        Verbosity (0=silent, 1=progress, 2=detailed)
    
    Returns:
    --------
    fold_results : list of dict
        Detailed results for each fold
    """
    
    if verbose >= 1:
        print("\n" + "=" * 70)
        print(f"TRAINING: {model_name}")
        print("=" * 70)
        print(f"SMOTE: {'✓' if CostConfig.USE_SMOTE else '✗'}, "
              f"Class Weights: {'✓' if use_class_weights else '✗'}")
    
    skf = StratifiedKFold(n_splits=CostConfig.N_SPLITS, 
                          shuffle=True, 
                          random_state=CostConfig.RANDOM_STATE)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        
        if verbose >= 1:
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
        X_train_fold = imputer.fit_transform(X_train_fold)
        X_val_fold = imputer.transform(X_val_fold)
        
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        original_train_size = len(y_train_fold)
        
        # SMOTE (training only)
        if CostConfig.USE_SMOTE:
            class_0_before = (y_train_fold == 0).sum()
            class_1_before = (y_train_fold == 1).sum()
            
            if verbose >= 1:
                print(f"Before SMOTE: Class 0={class_0_before:,}, "
                      f"Class 1={class_1_before:,} ({class_0_before/class_1_before:.1f}:1)")
            
            smote = SMOTE(sampling_strategy=CostConfig.SMOTE_RATIO,
                         k_neighbors=CostConfig.SMOTE_K_NEIGHBORS,
                         random_state=CostConfig.RANDOM_STATE)
            
            X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
            
            class_0_after = (y_train_fold == 0).sum()
            class_1_after = (y_train_fold == 1).sum()
            synthetic_added = len(y_train_fold) - original_train_size
            
            if verbose >= 1:
                print(f"After SMOTE:  Class 0={class_0_after:,}, "
                      f"Class 1={class_1_after:,} ({class_0_after/class_1_after:.1f}:1)")
                print(f"Synthetic samples: {synthetic_added:,}")
        
        # Class weights
        class_weights = None
        if use_class_weights:
            weights_array = compute_class_weight('balanced',
                                                 classes=np.unique(y_train_fold),
                                                 y=y_train_fold)
            class_weights = {0: weights_array[0], 1: weights_array[1]}
            
            if verbose >= 2:
                print(f"Class weights: {{0: {class_weights[0]:.3f}, 1: {class_weights[1]:.3f}}}")
        
        # Create and train model
        model = model_fn()
        
        early_stop = EarlyStopping(monitor='val_loss', 
                                   patience=CostConfig.EARLY_STOPPING_PATIENCE,
                                   restore_best_weights=True, verbose=0)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=CostConfig.REDUCE_LR_PATIENCE,
                                      min_lr=1e-7, verbose=0)
        
        history = model.fit(X_train_fold, y_train_fold,
                           validation_data=(X_val_fold, y_val_fold),
                           epochs=CostConfig.EPOCHS,
                           batch_size=CostConfig.BATCH_SIZE,
                           class_weight=class_weights,
                           callbacks=[early_stop, reduce_lr],
                           verbose=0)
        
        epochs_trained = len(history.history['loss'])
        
        # Continue in next cell...

print("✅ Enhanced CV function loaded (part 1)")
```

---

## Cell 6.5 [CODE] - Enhanced CV Training Function (Part 2/2)

```python
# ===============================================
# ENHANCED CV TRAINING FUNCTION (CONTINUED)
# ===============================================

# NOTE: This continues the train_model_with_enhanced_cv function
# Add this code INSIDE the for fold_idx loop, after model training

        # Predictions
        y_val_pred_proba = model.predict(X_val_fold, verbose=0).flatten()
        
        # Metrics at default threshold
        y_val_pred_default = (y_val_pred_proba >= 0.5).astype(int)
        default_cost, default_metrics = calculate_expected_cost(
            y_val_fold, y_val_pred_proba, threshold=0.5
        )
        
        # Optimize threshold
        optimal_threshold, optimal_cost, cost_curve = optimize_threshold(
            y_val_fold, y_val_pred_proba
        )
        
        y_val_pred_optimal = (y_val_pred_proba >= optimal_threshold).astype(int)
        
        # Classification metrics
        from sklearn.metrics import (roc_auc_score, precision_score, 
                                     recall_score, f1_score, roc_curve,
                                     precision_recall_curve, confusion_matrix)
        
        auc = roc_auc_score(y_val_fold, y_val_pred_proba)
        
        recall_default = recall_score(y_val_fold, y_val_pred_default, zero_division=0)
        recall_optimal = recall_score(y_val_fold, y_val_pred_optimal, zero_division=0)
        
        precision_default = precision_score(y_val_fold, y_val_pred_default, zero_division=0)
        precision_optimal = precision_score(y_val_fold, y_val_pred_optimal, zero_division=0)
        
        f1_default = f1_score(y_val_fold, y_val_pred_default, zero_division=0)
        f1_optimal = f1_score(y_val_fold, y_val_pred_optimal, zero_division=0)
        
        # Curve data
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val_fold, y_val_pred_proba)
        
        # Store results
        fold_data = {
            'auc': float(auc),
            'optimal_threshold': float(optimal_threshold),
            'optimal_cost': float(optimal_cost),
            'default_cost': float(default_cost),
            'cost_savings': float(default_cost - optimal_cost),
            'recall_default': float(recall_default),
            'recall_optimal': float(recall_optimal),
            'precision_default': float(precision_default),
            'precision_optimal': float(precision_optimal),
            'f1_default': float(f1_default),
            'f1_optimal': float(f1_optimal),
            'cost_curve': cost_curve,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'training_history': history.history,
            'epochs_trained': epochs_trained,
            'smote_used': CostConfig.USE_SMOTE,
            'class_weights_used': use_class_weights,
            'training_samples': len(y_train_fold),
            'validation_samples': len(y_val_fold)
        }
        
        fold_results.append(fold_data)
        cv_tracker.add_run(model_name, fold_idx, fold_data)
        
        if verbose >= 1:
            print(f"\n✓ Fold {fold_idx + 1} Complete:")
            print(f"  AUC: {auc:.4f}, Optimal τ: {optimal_threshold:.3f}")
            print(f"  Cost @ τ=0.5: ${default_cost:.2f}, Cost @ τ*: ${optimal_cost:.2f}")
            print(f"  Savings: ${default_cost - optimal_cost:.2f}")
            print(f"  Recall @ τ*: {recall_optimal:.3f}, Precision: {precision_optimal:.3f}")
    
    # Model summary (after all folds)
    if verbose >= 1:
        print(f"\n{'=' * 70}")
        print(f"{model_name} - 5-FOLD SUMMARY")
        print(f"{'=' * 70}")
        print(f"Mean AUC:        {np.mean([f['auc'] for f in fold_results]):.4f} ± "
              f"{np.std([f['auc'] for f in fold_results]):.4f}")
        print(f"Mean Opt Cost:   ${np.mean([f['optimal_cost'] for f in fold_results]):.2f} ± "
              f"${np.std([f['optimal_cost'] for f in fold_results]):.2f}")
        print(f"Mean Recall@τ*:  {np.mean([f['recall_optimal'] for f in fold_results]):.3f} ± "
              f"{np.std([f['recall_optimal'] for f in fold_results]):.3f}")
        print(f"Mean Opt τ:      {np.mean([f['optimal_threshold'] for f in fold_results]):.3f}")
        print("=" * 70)
    
    return fold_results

print("✅ Enhanced CV training function complete")
```

---

## Cell 6.6 [MARKDOWN] - Summary

```markdown
## **Preprocessing Strategy**

**Key Principles**:

1. **Leak-Safe Implementation**: Imputation and scaling fitted on training folds only, then applied to validation folds
2. **SMOTE Discipline**: Synthetic samples generated exclusively from training data
3. **Stratified Sampling**: Each fold maintains the original 27:1 class ratio
4. **Independent Evaluation**: Validation folds contain only real, unseen data

**35 Training Runs**:
- 7 models × 5 folds = 35 independent training runs
- Each run tracked in `cv_tracker` for full transparency
- Results aggregated for reliable performance estimates

**Why This Matters**: A single train/test split can be misleading due to variance. 5-fold CV provides robust estimates with confidence intervals.
```

