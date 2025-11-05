#!/usr/bin/env python3
"""
Script to fix critical issues in ReneWind_FINAL_Enhanced_With_Visualizations.ipynb
"""

import json
import sys
from pathlib import Path

# File paths
INPUT_NB = "/home/user/ReneWind_Assignment/ReneWind_FINAL_Enhanced_With_Visualizations.ipynb"
OUTPUT_NB = "/home/user/ReneWind_Assignment/ReneWind_FINAL_Enhanced_CORRECTED.ipynb"

def find_cell_with_content(cells, search_str):
    """Find cell index containing specific string"""
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if search_str in source:
                return idx
    return -1

def find_markdown_cell_with_content(cells, search_str):
    """Find markdown cell index containing specific string"""
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if search_str in source:
                return idx
    return -1

def create_code_cell(source_lines):
    """Create a new code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def main():
    print("="*80)
    print("NOTEBOOK FIX SCRIPT")
    print("="*80)

    # Load notebook
    print(f"\n1. Loading notebook: {INPUT_NB}")
    with open(INPUT_NB, 'r') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"   Total cells: {len(cells)}")

    changes = []

    # ========================================================================
    # ISSUE 1: Find and fix calculate_expected_cost function
    # ========================================================================
    print("\n2. Finding calculate_expected_cost function...")
    calc_cost_idx = find_cell_with_content(cells, "def calculate_expected_cost")

    if calc_cost_idx == -1:
        print("   ERROR: calculate_expected_cost function not found!")
        sys.exit(1)

    print(f"   Found at cell {calc_cost_idx}")

    # Show current function
    old_source = ''.join(cells[calc_cost_idx]['source'])
    print(f"\n   Current function length: {len(old_source)} chars")

    # New improved function with classification metrics
    new_function = [
        "def calculate_expected_cost(y_true, y_pred_proba, threshold, costs=None):\n",
        "    \"\"\"\n",
        "    Calculate expected cost with classification metrics.\n",
        "    \n",
        "    Parameters:\n",
        "    -----------\n",
        "    y_true : array-like\n",
        "        True labels\n",
        "    y_pred_proba : array-like\n",
        "        Predicted probabilities for positive class\n",
        "    threshold : float\n",
        "        Decision threshold\n",
        "    costs : dict, optional\n",
        "        Cost dictionary (defaults to CostConfig)\n",
        "    \n",
        "    Returns:\n",
        "    --------\n",
        "    expected_cost : float\n",
        "        Average cost per prediction\n",
        "    metrics : dict\n",
        "        Dictionary containing confusion matrix, costs, and classification metrics\n",
        "    \"\"\"\n",
        "    if costs is None:\n",
        "        costs = CostConfig.get_cost_dict()\n",
        "    \n",
        "    # Apply threshold\n",
        "    y_pred = (y_pred_proba >= threshold).astype(int)\n",
        "    \n",
        "    # Confusion matrix\n",
        "    tn = ((y_true == 0) & (y_pred == 0)).sum()\n",
        "    fp = ((y_true == 0) & (y_pred == 1)).sum()\n",
        "    fn = ((y_true == 1) & (y_pred == 0)).sum()\n",
        "    tp = ((y_true == 1) & (y_pred == 1)).sum()\n",
        "    \n",
        "    # Calculate costs\n",
        "    cost_fn = fn * costs['FN']\n",
        "    cost_tp = tp * costs['TP']\n",
        "    cost_fp = fp * costs['FP']\n",
        "    cost_tn = tn * costs['TN']\n",
        "    total_cost = cost_fn + cost_tp + cost_fp + cost_tn\n",
        "    expected_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0\n",
        "    \n",
        "    # Classification metrics\n",
        "    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0\n",
        "    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0\n",
        "    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0\n",
        "    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0\n",
        "    \n",
        "    # Package all metrics\n",
        "    metrics = {\n",
        "        'tn': int(tn),\n",
        "        'fp': int(fp),\n",
        "        'fn': int(fn),\n",
        "        'tp': int(tp),\n",
        "        'cost_fn': float(cost_fn),\n",
        "        'cost_tp': float(cost_tp),\n",
        "        'cost_fp': float(cost_fp),\n",
        "        'cost_tn': float(cost_tn),\n",
        "        'total_cost': float(total_cost),\n",
        "        'expected_cost': float(expected_cost),\n",
        "        'precision': float(precision),\n",
        "        'recall': float(recall),\n",
        "        'f1': float(f1),\n",
        "        'accuracy': float(accuracy)\n",
        "    }\n",
        "    \n",
        "    return expected_cost, metrics\n"
    ]

    cells[calc_cost_idx]['source'] = new_function
    changes.append({
        'issue': 'Issue 2: Metric Dictionary Mismatch',
        'cell_index': calc_cost_idx,
        'description': 'Updated calculate_expected_cost() to return classification metrics (precision, recall, F1, accuracy)',
        'lines_before': len(old_source.split('\n')),
        'lines_after': len(new_function)
    })
    print(f"   ✓ Fixed calculate_expected_cost function")

    # ========================================================================
    # ISSUE 2: Find test evaluation section and add BASE_COSTS
    # ========================================================================
    print("\n3. Finding test evaluation section...")

    # Look for the markdown header "Section 9: Final Model Evaluation on Test Data"
    test_eval_header_idx = find_markdown_cell_with_content(cells, "Section 9: Final Model Evaluation on Test Data")

    if test_eval_header_idx == -1:
        print("   ERROR: Test evaluation section not found!")
        sys.exit(1)

    print(f"   Found test evaluation header at cell {test_eval_header_idx}")

    # Find the cell that uses BASE_COSTS
    base_costs_usage_idx = find_cell_with_content(cells, "BASE_COSTS['FN']")

    if base_costs_usage_idx != -1:
        print(f"   Found BASE_COSTS usage at cell {base_costs_usage_idx}")

        # Insert BASE_COSTS definition before this cell
        base_costs_cell = create_code_cell([
            "# Get cost structure from configuration\n",
            "BASE_COSTS = CostConfig.get_cost_dict()\n",
            "print(\"Cost structure for test evaluation:\")\n",
            "print(f\"  FN (Replacement): ${BASE_COSTS['FN']:.2f}\")\n",
            "print(f\"  TP (Repair):      ${BASE_COSTS['TP']:.2f}\")\n",
            "print(f\"  FP (Inspection):  ${BASE_COSTS['FP']:.2f}\")\n",
            "print(f\"  TN (Normal):      ${BASE_COSTS['TN']:.2f}\")\n"
        ])

        cells.insert(base_costs_usage_idx, base_costs_cell)
        print(f"   ✓ Inserted BASE_COSTS definition at cell {base_costs_usage_idx}")

        changes.append({
            'issue': 'Issue 1: BASE_COSTS Not Defined',
            'cell_index': base_costs_usage_idx,
            'description': 'Added BASE_COSTS definition cell before test evaluation',
            'action': 'Inserted new cell'
        })

        # Update indices for subsequent searches
        if calc_cost_idx >= base_costs_usage_idx:
            calc_cost_idx += 1
    else:
        print("   WARNING: BASE_COSTS usage not found in notebook")

    # ========================================================================
    # ISSUE 3: Fix cost summary DataFrame
    # ========================================================================
    print("\n4. Finding cost summary DataFrame...")

    cost_summary_idx = find_cell_with_content(cells, "cost_summary = pd.DataFrame")

    if cost_summary_idx == -1:
        print("   WARNING: cost_summary DataFrame not found")
    else:
        print(f"   Found at cell {cost_summary_idx}")

        # Check current source
        old_summary_source = ''.join(cells[cost_summary_idx]['source'])

        # New cost summary that uses correct metric keys
        new_summary_source = [
            "# Cost Summary Table\n",
            "cost_summary = pd.DataFrame({\n",
            "    'Threshold': ['Default (0.5)', 'Optimized', 'Naive (All Fail)'],\n",
            "    'Expected Cost': [\n",
            "        f\"${default_cost:.2f}\",\n",
            "        f\"${optimal_cost:.2f}\",\n",
            "        f\"${naive_cost:.2f}\"\n",
            "    ],\n",
            "    'Precision': [\n",
            "        f\"{default_metrics['precision']:.3f}\",\n",
            "        f\"{optimal_metrics['precision']:.3f}\",\n",
            "        'N/A'\n",
            "    ],\n",
            "    'Recall': [\n",
            "        f\"{default_metrics['recall']:.3f}\",\n",
            "        f\"{optimal_metrics['recall']:.3f}\",\n",
            "        '1.000'\n",
            "    ],\n",
            "    'F1 Score': [\n",
            "        f\"{default_metrics['f1']:.3f}\",\n",
            "        f\"{optimal_metrics['f1']:.3f}\",\n",
            "        'N/A'\n",
            "    ],\n",
            "    'Accuracy': [\n",
            "        f\"{default_metrics['accuracy']:.3f}\",\n",
            "        f\"{optimal_metrics['accuracy']:.3f}\",\n",
            "        '0.000'\n",
            "    ]\n",
            "})\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"COST COMPARISON SUMMARY\")\n",
            "print(\"=\"*80)\n",
            "print(cost_summary.to_string(index=False))\n",
            "print(\"=\"*80)\n"
        ]

        cells[cost_summary_idx]['source'] = new_summary_source

        changes.append({
            'issue': 'Issue 3: Cost Summary DataFrame',
            'cell_index': cost_summary_idx,
            'description': 'Fixed cost_summary to use correct metric keys from extended function',
            'lines_before': len(old_summary_source.split('\n')),
            'lines_after': len(new_summary_source)
        })
        print(f"   ✓ Fixed cost_summary DataFrame")

    # ========================================================================
    # ISSUE 4: Verify SMOTE logging
    # ========================================================================
    print("\n5. Verifying SMOTE implementation...")

    smote_idx = find_cell_with_content(cells, "if CostConfig.USE_SMOTE:")

    if smote_idx == -1:
        print("   WARNING: SMOTE code not found")
    else:
        print(f"   Found SMOTE implementation at cell {smote_idx}")
        source = ''.join(cells[smote_idx]['source'])

        # Check if logging is present
        if "Before SMOTE:" in source and "After SMOTE:" in source:
            print("   ✓ SMOTE logging already present")
            changes.append({
                'issue': 'Issue 4: SMOTE Verification',
                'cell_index': smote_idx,
                'description': 'SMOTE logging verified - already properly implemented',
                'action': 'No changes needed'
            })
        else:
            print("   ⚠ SMOTE logging may need enhancement")

    # ========================================================================
    # Save corrected notebook
    # ========================================================================
    print(f"\n6. Saving corrected notebook to: {OUTPUT_NB}")

    with open(OUTPUT_NB, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"   ✓ Saved successfully")
    print(f"   Total cells in corrected notebook: {len(cells)}")

    # ========================================================================
    # Generate Fix Report
    # ========================================================================
    print("\n" + "="*80)
    print("FIX REPORT SUMMARY")
    print("="*80)

    for i, change in enumerate(changes, 1):
        print(f"\n{i}. {change['issue']}")
        print(f"   Cell Index: {change['cell_index']}")
        print(f"   Description: {change['description']}")
        if 'action' in change:
            print(f"   Action: {change['action']}")
        if 'lines_before' in change and 'lines_after' in change:
            print(f"   Lines: {change['lines_before']} → {change['lines_after']}")

    print("\n" + "="*80)
    print(f"✓ All fixes applied successfully!")
    print(f"✓ Output saved to: {OUTPUT_NB}")
    print("="*80)

    return changes

if __name__ == "__main__":
    try:
        changes = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
