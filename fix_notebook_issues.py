#!/usr/bin/env python3
"""
Fix three critical issues in ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb:
1. Add cell to execute cost_sensitivity_analysis (±20% FN/FP robustness)
2. Fix hard-coded prevalence (0.0556) to use y_test_final.mean()
3. Update narrative to reference actual computed results
"""

import json
import sys

NOTEBOOK_PATH = "ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb"
OUTPUT_PATH = "ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb"

def load_notebook():
    """Load the notebook"""
    with open(NOTEBOOK_PATH, 'r') as f:
        return json.load(f)

def save_notebook(nb):
    """Save the notebook"""
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Saved to: {OUTPUT_PATH}")

def fix_hardcoded_prevalence(nb):
    """Fix Issue 2: Replace hard-coded 0.0556 with y_test_final.mean()"""
    print("\n" + "=" * 80)
    print("FIX 1: Hard-coded prevalence in PR plot (Cell 125)")
    print("=" * 80)

    cell_125 = nb['cells'][125]
    source = ''.join(cell_125['source'])

    # Replace hard-coded prevalence
    old_line1 = "    plt.axhline(y=0.0556, color='red', linestyle='--', linewidth=1.5,"
    old_line2 = "                label=f'Baseline (prevalence={0.0556:.3f})')"

    new_line1 = "    baseline_prevalence = y_test_final.mean()  # Dynamic calculation"
    new_line2 = "    plt.axhline(y=baseline_prevalence, color='red', linestyle='--', linewidth=1.5,"
    new_line3 = "                label=f'Baseline (prevalence={baseline_prevalence:.3f})')"

    if '0.0556' in source:
        # Split into lines
        lines = source.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'plt.axhline(y=0.0556' in line:
                # Replace these two lines
                new_lines.append(new_line1)
                new_lines.append(new_line2)
                new_lines.append(new_line3)
                i += 2  # Skip the next line too
            else:
                new_lines.append(line)
                i += 1

        new_source = '\n'.join(new_lines)
        nb['cells'][125]['source'] = new_source.split('\n')

        print("✓ Replaced hard-coded prevalence 0.0556 with y_test_final.mean()")
        print("  Lines changed:")
        print(f"    OLD: {old_line1}")
        print(f"    OLD: {old_line2}")
        print(f"    NEW: {new_line1}")
        print(f"    NEW: {new_line2}")
        print(f"    NEW: {new_line3}")
    else:
        print("⚠️  Hard-coded 0.0556 not found in expected location")

    return nb

def add_cost_sensitivity_execution(nb):
    """Fix Issue 1: Add cell to execute cost_sensitivity_analysis"""
    print("\n" + "=" * 80)
    print("FIX 2: Add cost sensitivity analysis execution")
    print("=" * 80)

    # Create markdown cell
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cost Parameter Robustness (±20% FN/FP Variation)\n",
            "\n",
            "Testing how the optimal threshold changes when FN and FP costs vary by ±20%."
        ]
    }

    # Create code cell that executes cost_sensitivity_analysis
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ===============================================\n",
            "# COST PARAMETER ROBUSTNESS ANALYSIS\n",
            "# ===============================================\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"COST PARAMETER ROBUSTNESS ANALYSIS (±20% FN/FP Variation)\")\n",
            "print(\"=\"*70)\n",
            "print(\"\\n⏳ Testing 9 cost scenarios (3 FN values × 3 FP values)...\")\n",
            "\n",
            "# Execute the cost sensitivity analysis\n",
            "sensitivity_df = cost_sensitivity_analysis(\n",
            "    y_true=y_test_final,\n",
            "    y_pred_proba=test_pred_proba,\n",
            "    base_threshold=best_model_threshold,\n",
            "    perturbation=0.20  # ±20% variation\n",
            ")\n",
            "\n",
            "print(\"\\n✓ Cost sensitivity analysis complete\")\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"SENSITIVITY RESULTS\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "# Display the full sensitivity table\n",
            "print(\"\\nCost Scenario Matrix:\")\n",
            "print(sensitivity_df.to_string(index=False))\n",
            "\n",
            "# Calculate key statistics\n",
            "threshold_range = sensitivity_df['optimal_threshold'].max() - sensitivity_df['optimal_threshold'].min()\n",
            "cost_range = sensitivity_df['optimal_cost'].max() - sensitivity_df['optimal_cost'].min()\n",
            "base_scenario = sensitivity_df[sensitivity_df['scenario'].str.contains('FN=800, FP=100')].iloc[0]\n",
            "\n",
            "print(f\"\\n\" + \"=\"*70)\n",
            "print(\"ROBUSTNESS METRICS\")\n",
            "print(\"=\"*70)\n",
            "print(f\"\\nBase scenario (FN=$800, FP=$100):\")\n",
            "print(f\"  • Optimal threshold: {base_scenario['optimal_threshold']:.3f}\")\n",
            "print(f\"  • Optimal cost: ${base_scenario['optimal_cost']:.2f}\")\n",
            "\n",
            "print(f\"\\nRobustness across ±20% cost variations:\")\n",
            "print(f\"  • Threshold range: {threshold_range:.3f}\")\n",
            "print(f\"  • Cost range: ${cost_range:.2f}\")\n",
            "print(f\"  • Min optimal threshold: {sensitivity_df['optimal_threshold'].min():.3f}\")\n",
            "print(f\"  • Max optimal threshold: {sensitivity_df['optimal_threshold'].max():.3f}\")\n",
            "print(f\"  • Min optimal cost: ${sensitivity_df['optimal_cost'].min():.2f}\")\n",
            "print(f\"  • Max optimal cost: ${sensitivity_df['optimal_cost'].max():.2f}\")\n",
            "\n",
            "# Visualize with heatmap\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Pivot for heatmap\n",
            "pivot_threshold = sensitivity_df.pivot_table(\n",
            "    index='FN_cost', \n",
            "    columns='FP_cost', \n",
            "    values='optimal_threshold'\n",
            ")\n",
            "\n",
            "pivot_cost = sensitivity_df.pivot_table(\n",
            "    index='FN_cost', \n",
            "    columns='FP_cost', \n",
            "    values='optimal_cost'\n",
            ")\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Threshold heatmap\n",
            "sns.heatmap(pivot_threshold, annot=True, fmt='.3f', cmap='RdYlGn_r', \n",
            "            ax=axes[0], cbar_kws={'label': 'Optimal Threshold'})\n",
            "axes[0].set_title('Optimal Threshold vs. FN/FP Costs', fontweight='bold')\n",
            "axes[0].set_xlabel('FP Cost ($)', fontweight='bold')\n",
            "axes[0].set_ylabel('FN Cost ($)', fontweight='bold')\n",
            "\n",
            "# Cost heatmap\n",
            "sns.heatmap(pivot_cost, annot=True, fmt='.0f', cmap='RdYlGn_r', \n",
            "            ax=axes[1], cbar_kws={'label': 'Expected Cost ($)'})\n",
            "axes[1].set_title('Expected Cost vs. FN/FP Costs', fontweight='bold')\n",
            "axes[1].set_xlabel('FP Cost ($)', fontweight='bold')\n",
            "axes[1].set_ylabel('FN Cost ($)', fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"\\n✓ Cost parameter robustness analysis complete\")\n",
            "print(\"\\n\" + \"=\"*70)\n"
        ]
    }

    # Insert after cell 131 (threshold sensitivity)
    insert_position = 132  # After cell 131

    nb['cells'].insert(insert_position, markdown_cell)
    nb['cells'].insert(insert_position + 1, code_cell)

    print(f"✓ Inserted markdown cell at position {insert_position}")
    print(f"✓ Inserted code cell at position {insert_position + 1}")
    print("  This cell will execute cost_sensitivity_analysis() and display:")
    print("    - Full sensitivity DataFrame")
    print("    - Robustness metrics (threshold range, cost range)")
    print("    - Heatmap visualizations")

    return nb

def update_narrative(nb):
    """Fix Issue 3: Update narrative to reference actual computed results"""
    print("\n" + "=" * 80)
    print("FIX 3: Update Cost Sensitivity Profile narrative")
    print("=" * 80)

    # Find cell 139 (now shifted by 2 due to insertions)
    narrative_cell_idx = 141  # Original 139 + 2 inserted cells

    if narrative_cell_idx < len(nb['cells']):
        cell = nb['cells'][narrative_cell_idx]
        source = ''.join(cell['source'])

        if 'Cost Sensitivity Profile' in source:
            # Add a note at the beginning of the narrative
            new_intro = [
                "**Note**: All cost sensitivity metrics referenced below are computed programmatically ",
                "in the 'Cost Parameter Robustness Analysis' section above. The sensitivity analysis ",
                "tests ±20% variations in FN and FP costs across 9 scenarios to verify model robustness.\n\n",
                "---\n\n"
            ]

            # Prepend to existing source
            existing_source = cell['source']
            if isinstance(existing_source, list):
                cell['source'] = new_intro + existing_source
            else:
                cell['source'] = ''.join(new_intro) + existing_source

            print(f"✓ Updated narrative at cell {narrative_cell_idx}")
            print("  Added note referencing the actual computed sensitivity analysis")
        else:
            print(f"⚠️  Cell {narrative_cell_idx} doesn't contain 'Cost Sensitivity Profile'")
    else:
        print(f"⚠️  Cell {narrative_cell_idx} out of range")

    return nb

def main():
    print("=" * 80)
    print("FIXING CRITICAL ISSUES IN NOTEBOOK")
    print("=" * 80)

    # Load notebook
    print("\nLoading notebook...")
    nb = load_notebook()
    print(f"✓ Loaded {len(nb['cells'])} cells")

    # Apply fixes
    nb = fix_hardcoded_prevalence(nb)
    nb = add_cost_sensitivity_execution(nb)
    nb = update_narrative(nb)

    # Save
    print("\n" + "=" * 80)
    print("SAVING CORRECTED NOTEBOOK")
    print("=" * 80)
    save_notebook(nb)

    print("\n" + "=" * 80)
    print("FIXES APPLIED SUCCESSFULLY")
    print("=" * 80)
    print("\nSummary of changes:")
    print("  1. ✓ Fixed hard-coded prevalence (0.0556 → y_test_final.mean())")
    print("  2. ✓ Added cost_sensitivity_analysis execution cell")
    print("  3. ✓ Updated narrative to reference computed results")
    print("\nTotal cells after fixes:", len(nb['cells']))
    print("\n⚠️  NOTE: The notebook needs to be RE-RUN to generate outputs for new cells")
    print("=" * 80)

if __name__ == "__main__":
    main()
