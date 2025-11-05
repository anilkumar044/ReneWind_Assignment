# ReneWind Notebook - Complete Implementation Cells
# PART 1: Sections 5 & 6

"""
Usage Instructions:
1. Each section below contains complete, runnable code cells
2. Copy the code blocks and insert them at the specified locations
3. Run cells sequentially to ensure proper execution
4. Markdown cells are provided for documentation
"""

# =============================================================================
# SECTION 5: Enhanced Cost-Aware Optimization Framework
# =============================================================================

# -----------------------------------------------------------------------------
# Cell 5.1 - Markdown Introduction
# -----------------------------------------------------------------------------
MARKDOWN_5_1 = """
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
"""

# -----------------------------------------------------------------------------
# Cell 5.2 - CostConfig Class
# -----------------------------------------------------------------------------
CODE_5_2 = """
# ===============================================
# COST CONFIGURATION CLASS
# ===============================================

class CostConfig:
    \"\"\"
    Centralized configuration for cost-aware optimization and experimentation.

    This class manages:
    - Business cost structure (FN, TP, FP, TN)
    - SMOTE oversampling configuration
    - Cross-validation parameters
    - Random seed for reproducibility
    \"\"\"

    # ==================== BUSINESS COSTS ====================
    # Derived from ReneWind maintenance economics
    FN = 100.0  # False Negative: unplanned replacement cost
    TP = 30.0   # True Positive: proactive repair cost
    FP = 10.0   # False Positive: inspection cost
    TN = 0.0    # True Negative: normal operations (no cost)

    # ==================== SMOTE CONFIGURATION ====================
    USE_SMOTE = True  # Toggle SMOTE oversampling (set False to disable)
    SMOTE_RATIO = 0.5  # Target ratio after SMOTE (0.5 = minority will be 50% of majority)
    SMOTE_K_NEIGHBORS = 5  # Number of nearest neighbors for synthetic sample generation

    # ==================== CROSS-VALIDATION ====================
    N_SPLITS = 5  # Number of folds for StratifiedKFold
    RANDOM_STATE = 42  # Seed for reproducibility

    # ==================== TRAINING CONFIGURATION ====================
    EPOCHS = 100
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5

    @classmethod
    def get_cost_dict(cls):
        \"\"\"Return cost structure as dictionary.\"\"\"
        return {
            'FN': cls.FN,
            'TP': cls.TP,
            'FP': cls.FP,
            'TN': cls.TN
        }

    @classmethod
    def display_config(cls):
        \"\"\"Print current configuration in formatted table.\"\"\"
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print("\\n📊 Business Cost Structure:")
        print(f"   FN (Replacement):  ${cls.FN:.2f}")
        print(f"   TP (Repair):       ${cls.TP:.2f}")
        print(f"   FP (Inspection):   ${cls.FP:.2f}")
        print(f"   TN (Normal):       ${cls.TN:.2f}")
        print(f"   Cost Hierarchy:    FN > TP > FP > TN")

        print(f"\\n🔬 SMOTE Oversampling: {'✓ ENABLED' if cls.USE_SMOTE else '✗ DISABLED'}")
        if cls.USE_SMOTE:
            print(f"   Sampling Strategy: {cls.SMOTE_RATIO} (minority = {cls.SMOTE_RATIO*100:.0f}% of majority)")
            print(f"   K-Neighbors:       {cls.SMOTE_K_NEIGHBORS}")

        print(f"\\n🔄 Cross-Validation:")
        print(f"   Strategy:          {cls.N_SPLITS}-Fold StratifiedKFold")
        print(f"   Random Seed:       {cls.RANDOM_STATE}")

        print(f"\\n⚙️ Training Parameters:")
        print(f"   Epochs:            {cls.EPOCHS}")
        print(f"   Batch Size:        {cls.BATCH_SIZE}")
        print(f"   Early Stop Patience: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"   Reduce LR Patience:  {cls.REDUCE_LR_PATIENCE}")

        print("=" * 70)

# Display configuration
CostConfig.display_config()
"""

# Save to file for easy reference
if __name__ == "__main__":
    print("ReneWind Notebook Implementation Cells - Part 1")
    print("=" * 70)
    print("This file contains code for Sections 5 & 6")
    print("Copy and paste into your Jupyter notebook")
    print("=" * 70)
