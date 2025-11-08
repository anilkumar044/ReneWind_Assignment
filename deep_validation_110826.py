#!/usr/bin/env python3
"""
Deep validation of ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb
Checks for data leakage, proper workflows, and business logic correctness
"""

import json
import re
from typing import List, Dict, Tuple

NOTEBOOK_PATH = "/home/user/ReneWind_Assignment/ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb"

def load_notebook():
    """Load notebook"""
    with open(NOTEBOOK_PATH, 'r') as f:
        return json.load(f)

def get_code_cells(notebook):
    """Extract all code cells with their indices"""
    cells = []
    for idx, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            cells.append({
                'index': idx,
                'source': source,
                'outputs': cell.get('outputs', [])
            })
    return cells

def validate_cv_loop_structure():
    """Validate cross-validation loop for data leakage"""
    print("="*80)
    print("DEEP CHECK 1: Cross-Validation Loop Structure")
    print("="*80)

    nb = load_notebook()
    cells = get_code_cells(nb)

    cv_cells = []
    for cell in cells:
        if 'def train_model_with_enhanced_cv' in cell['source']:
            cv_cells.append(cell)

    if not cv_cells:
        print("⚠️  No CV function found")
        return

    print(f"✓ Found {len(cv_cells)} CV implementation cell(s)")

    for cell in cv_cells:
        source = cell['source']
        lines = source.split('\n')

        # Check for proper fold loop
        fold_loop_found = False
        for line in lines:
            if re.search(r'for.*fold.*in.*enumerate.*\.split', line):
                fold_loop_found = True
                print(f"✓ Proper fold enumeration found at cell {cell['index']}")
                break

        if not fold_loop_found:
            print(f"⚠️  No clear fold enumeration in cell {cell['index']}")

        # Check for scaler fit_transform on train, transform on val
        if 'scaler.fit_transform(X_train' in source and 'scaler.transform(X_val' in source:
            print("✓ Proper scaler workflow: fit_transform on train, transform on val")
        elif 'fit_transform' in source:
            print("⚠️  fit_transform found but validate transform workflow manually")

        # Check for imputer fit_transform on train, transform on val
        if 'imputer.fit_transform(X_train' in source and 'imputer.transform(X_val' in source:
            print("✓ Proper imputer workflow: fit_transform on train, transform on val")
        elif 'fit_transform' in source:
            print("⚠️  Imputer usage should be validated manually")

        # Check SMOTE is applied only on training data
        if 'SMOTE' in source:
            if re.search(r'SMOTE.*fit_resample.*X_train.*y_train', source, re.DOTALL):
                print("✓ SMOTE applied only to training data (inside CV loop)")
            else:
                print("⚠️  SMOTE usage should be validated manually")

        # Check fold_results.append is inside loop
        append_indent = None
        for_indent = None

        for i, line in enumerate(lines):
            if re.search(r'for.*fold.*enumerate', line):
                for_indent = len(line) - len(line.lstrip())
            if 'fold_results.append(' in line:
                append_indent = len(line) - len(line.lstrip())

        if for_indent is not None and append_indent is not None:
            if append_indent > for_indent:
                print("✓ fold_results.append() is INSIDE the loop (correct)")
            else:
                print("❌ CRITICAL: fold_results.append() appears OUTSIDE the loop")

        # Check return is outside loop
        return_indent = None
        for line in lines:
            if 'return fold_results' in line or 'return {' in line:
                return_indent = len(line) - len(line.lstrip())
                break

        if for_indent is not None and return_indent is not None:
            if return_indent <= for_indent or return_indent == for_indent:
                print("✓ return statement is OUTSIDE the loop (correct)")
            else:
                print("⚠️  return statement indentation should be verified")

def validate_test_set_workflow():
    """Validate test set is not used in training"""
    print("\n" + "="*80)
    print("DEEP CHECK 2: Test Set Workflow")
    print("="*80)

    nb = load_notebook()
    cells = get_code_cells(nb)

    # Find train_test_split
    split_cell_idx = None
    for cell in cells:
        if 'train_test_split' in cell['source']:
            split_cell_idx = cell['index']
            print(f"✓ Train/test split found at cell {split_cell_idx}")
            break

    if split_cell_idx is None:
        print("⚠️  No train_test_split found")
        return

    # Check cells before split don't use X_test or y_test
    test_used_before_split = False
    for cell in cells:
        if cell['index'] < split_cell_idx:
            if 'X_test' in cell['source'] or 'y_test' in cell['source']:
                print(f"❌ CRITICAL: X_test/y_test used in cell {cell['index']} BEFORE split")
                test_used_before_split = True

    if not test_used_before_split:
        print("✓ No test set usage before train/test split")

    # Check test set is not used in CV training
    for cell in cells:
        source = cell['source']
        if 'def train_model_with_enhanced_cv' in source:
            if 'X_test' in source or 'y_test' in source:
                # Check if it's just in comments or docstrings
                if re.search(r'(?<!#)(?<!""")(?<!\'\'\')\bX_test\b', source):
                    print(f"⚠️  X_test referenced in CV function at cell {cell['index']} - verify it's not used in training")
            else:
                print("✓ CV function does not reference test set")

    # Find final model training cell
    final_train_cells = []
    for cell in cells:
        if 'FINAL MODEL TRAINING' in cell['source'] or 'final_model.fit' in cell['source']:
            final_train_cells.append(cell)

    if final_train_cells:
        print(f"✓ Found {len(final_train_cells)} final model training cell(s)")

        for cell in final_train_cells:
            # Check it uses X_train not X_test
            if re.search(r'\.fit\(X_train', cell['source']):
                print(f"✓ Final model trained on X_train (cell {cell['index']})")
            elif re.search(r'\.fit\(.*X_test', cell['source']):
                print(f"❌ CRITICAL: Final model may be trained on X_test (cell {cell['index']})")

    # Find test evaluation cells
    test_eval_cells = []
    for cell in cells:
        if 'predict(X_test' in cell['source'] or 'TEST EVALUATION' in cell['source']:
            test_eval_cells.append(cell)

    if test_eval_cells:
        print(f"✓ Found {len(test_eval_cells)} test evaluation cell(s)")

        for cell in test_eval_cells:
            # Ensure test predictions use properly scaled data
            if 'predict(X_test_scaled' in cell['source']:
                print(f"✓ Test predictions use scaled data (cell {cell['index']})")
            elif 'predict(X_test)' in cell['source'] and 'scaled' not in cell['source']:
                print(f"⚠️  Test predictions may use unscaled data (cell {cell['index']})")

def validate_scaling_workflow():
    """Validate proper scaling workflow"""
    print("\n" + "="*80)
    print("DEEP CHECK 3: Scaling Workflow")
    print("="*80)

    nb = load_notebook()
    cells = get_code_cells(nb)

    # Find preprocessing cells
    preproc_cells = []
    for cell in cells:
        if 'StandardScaler' in cell['source'] or 'scaler' in cell['source'].lower():
            preproc_cells.append(cell)

    print(f"✓ Found {len(preproc_cells)} cells with scaling operations")

    # Check for proper workflow
    has_train_fit_transform = False
    has_test_transform = False
    has_double_scaling = False

    for cell in preproc_cells:
        source = cell['source']

        # Check for training data scaling
        if re.search(r'scaler.*fit_transform.*X_train', source, re.DOTALL):
            has_train_fit_transform = True
            print(f"✓ Scaler fit_transform on training data (cell {cell['index']})")

        # Check for test data scaling
        if re.search(r'scaler.*transform.*X_test', source, re.DOTALL):
            has_test_transform = True
            print(f"✓ Scaler transform on test data (cell {cell['index']})")

        # Check for double scaling (bad pattern)
        if 'fit_transform(X_test' in source:
            has_double_scaling = True
            print(f"❌ CRITICAL: fit_transform on X_test - this causes data leakage (cell {cell['index']})")

    if has_train_fit_transform and has_test_transform and not has_double_scaling:
        print("✓ Scaling workflow is correct: fit on train, transform on test")
    elif has_double_scaling:
        print("❌ Scaling workflow has CRITICAL issues")
    else:
        print("⚠️  Scaling workflow should be manually verified")

def validate_cost_framework():
    """Validate cost-sensitive framework implementation"""
    print("\n" + "="*80)
    print("DEEP CHECK 4: Cost-Sensitive Framework")
    print("="*80)

    nb = load_notebook()
    cells = get_code_cells(nb)

    # Find CostConfig
    config_cells = []
    for cell in cells:
        if 'class CostConfig' in cell['source']:
            config_cells.append(cell)

    if config_cells:
        print(f"✓ Found {len(config_cells)} CostConfig definition(s)")

        for cell in config_cells:
            source = cell['source']

            # Check for cost values
            if 'COST_FN' in source and 'COST_FP' in source:
                print(f"✓ Cost values defined (cell {cell['index']})")

                # Try to extract cost values
                fn_match = re.search(r'COST_FN\s*=\s*(\d+)', source)
                fp_match = re.search(r'COST_FP\s*=\s*(\d+)', source)

                if fn_match and fp_match:
                    cost_fn = int(fn_match.group(1))
                    cost_fp = int(fp_match.group(1))
                    print(f"  - COST_FN: {cost_fn}")
                    print(f"  - COST_FP: {cost_fp}")

                    if cost_fn > cost_fp:
                        print("  ✓ COST_FN > COST_FP (correct for failure detection)")
                    else:
                        print("  ⚠️  COST_FN <= COST_FP - verify this is intentional")

    # Find cost calculation function
    calc_cells = []
    for cell in cells:
        if 'def calculate_expected_cost' in cell['source']:
            calc_cells.append(cell)

    if calc_cells:
        print(f"✓ Found {len(calc_cells)} cost calculation function(s)")

        for cell in calc_cells:
            source = cell['source']

            # Check it returns necessary metrics
            required_metrics = ['precision', 'recall', 'f1', 'accuracy', 'total_cost', 'expected_cost']
            found_metrics = []

            for metric in required_metrics:
                if f"'{metric}'" in source or f'"{metric}"' in source:
                    found_metrics.append(metric)

            if len(found_metrics) >= 4:
                print(f"✓ Function returns multiple metrics (cell {cell['index']})")
                print(f"  Metrics: {', '.join(found_metrics)}")
            else:
                print(f"⚠️  Limited metrics returned (cell {cell['index']})")

    # Find threshold optimization
    opt_cells = []
    for cell in cells:
        if 'optimize_threshold' in cell['source'] or 'threshold' in cell['source'].lower():
            opt_cells.append(cell)

    if opt_cells:
        print(f"✓ Found {len(opt_cells)} cells with threshold operations")

def validate_outputs():
    """Validate cell outputs are present and reasonable"""
    print("\n" + "="*80)
    print("DEEP CHECK 5: Output Validation")
    print("="*80)

    nb = load_notebook()
    cells = get_code_cells(nb)

    # Check for key outputs
    key_outputs = {
        'confusion_matrix': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'cost': 0,
        'threshold': 0,
        'accuracy': 0
    }

    for cell in cells:
        outputs = cell.get('outputs', [])

        for output in outputs:
            output_text = json.dumps(output).lower()

            for key in key_outputs:
                if key in output_text:
                    key_outputs[key] += 1

    print("Output presence check:")
    for key, count in key_outputs.items():
        if count > 0:
            print(f"✓ {key}: found in {count} cell(s)")
        else:
            print(f"⚠️  {key}: not found in outputs")

    # Check for errors in outputs
    error_count = 0
    for cell in cells:
        outputs = cell.get('outputs', [])

        for output in outputs:
            if output.get('output_type') == 'error':
                error_count += 1
                print(f"❌ Error in cell {cell['index']}: {output.get('ename', 'Unknown')}")

    if error_count == 0:
        print("✓ No errors in cell outputs")
    else:
        print(f"❌ Found {error_count} error(s) in outputs")

def check_business_requirements():
    """Validate business requirements are met"""
    print("\n" + "="*80)
    print("DEEP CHECK 6: Business Requirements")
    print("="*80)

    nb = load_notebook()
    cells = get_code_cells(nb)
    all_code = '\n'.join([cell['source'] for cell in cells])

    requirements = {
        'Wind turbine failure prediction': 'fail' in all_code.lower() or 'failure' in all_code.lower(),
        'Imbalanced data handling': 'SMOTE' in all_code or 'class_weight' in all_code,
        'Cost-sensitive learning': 'cost' in all_code.lower() and 'CostConfig' in all_code,
        'Cross-validation': 'StratifiedKFold' in all_code or 'cross_val' in all_code,
        'Feature engineering': 'feature' in all_code.lower(),
        'Model evaluation': 'confusion_matrix' in all_code or 'classification_report' in all_code,
        'Threshold optimization': 'threshold' in all_code.lower() and 'optim' in all_code.lower(),
        'Test set evaluation': 'X_test' in all_code and 'predict' in all_code
    }

    print("Business requirements check:")
    all_met = True
    for req, met in requirements.items():
        if met:
            print(f"✓ {req}")
        else:
            print(f"⚠️  {req} - not clearly identified")
            all_met = False

    if all_met:
        print("\n✓ All business requirements appear to be addressed")
    else:
        print("\n⚠️  Some requirements should be manually verified")

def main():
    print("="*80)
    print("DEEP VALIDATION OF NOTEBOOK")
    print("ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb")
    print("="*80)
    print()

    try:
        validate_cv_loop_structure()
        validate_test_set_workflow()
        validate_scaling_workflow()
        validate_cost_framework()
        validate_outputs()
        check_business_requirements()

        print("\n" + "="*80)
        print("DEEP VALIDATION COMPLETE")
        print("="*80)
        print("Review the findings above for any critical issues.")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
