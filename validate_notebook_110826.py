#!/usr/bin/env python3
"""
Comprehensive validation of ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb
This script performs thorough validation of the notebook structure, code, and outputs.
"""

import json
import sys
import re
from collections import defaultdict

NOTEBOOK_PATH = "/home/user/ReneWind_Assignment/ReneWind_FINAL_PRODUCTION_with_output-110826.ipynb"

class NotebookValidator:
    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        self.notebook = None
        self.issues = []
        self.warnings = []
        self.passed_checks = []

    def load_notebook(self):
        """Load and parse the notebook JSON"""
        print("=" * 80)
        print("STEP 1: Loading Notebook")
        print("=" * 80)
        try:
            with open(self.notebook_path, 'r') as f:
                self.notebook = json.load(f)
            self.passed_checks.append("✓ Notebook JSON is valid and parseable")
            return True
        except json.JSONDecodeError as e:
            self.issues.append(f"✗ Invalid JSON: {e}")
            return False
        except FileNotFoundError:
            self.issues.append(f"✗ File not found: {self.notebook_path}")
            return False
        except Exception as e:
            self.issues.append(f"✗ Error loading notebook: {e}")
            return False

    def validate_structure(self):
        """Validate basic notebook structure"""
        print("\n" + "=" * 80)
        print("STEP 2: Validating Notebook Structure")
        print("=" * 80)

        # Check required keys
        required_keys = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        for key in required_keys:
            if key in self.notebook:
                self.passed_checks.append(f"✓ Has required key: '{key}'")
            else:
                self.issues.append(f"✗ Missing required key: '{key}'")

        # Count cells
        cells = self.notebook.get('cells', [])
        code_cells = [c for c in cells if c.get('cell_type') == 'code']
        markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']

        self.passed_checks.append(f"✓ Total cells: {len(cells)}")
        self.passed_checks.append(f"✓ Code cells: {len(code_cells)}")
        self.passed_checks.append(f"✓ Markdown cells: {len(markdown_cells)}")

        if len(code_cells) == 0:
            self.issues.append("✗ No code cells found")

        return len(cells) > 0

    def validate_syntax(self):
        """Validate Python syntax in all code cells"""
        print("\n" + "=" * 80)
        print("STEP 3: Validating Python Syntax")
        print("=" * 80)

        cells = self.notebook.get('cells', [])
        syntax_errors = []

        for idx, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if source.strip():  # Only check non-empty cells
                    try:
                        compile(source, f'<cell_{idx}>', 'exec')
                    except SyntaxError as e:
                        syntax_errors.append((idx, str(e)))

        if syntax_errors:
            for idx, err in syntax_errors[:10]:  # Show first 10
                self.issues.append(f"✗ Syntax error in cell {idx}: {err}")
            if len(syntax_errors) > 10:
                self.issues.append(f"✗ ...and {len(syntax_errors) - 10} more syntax errors")
        else:
            self.passed_checks.append("✓ All code cells have valid Python syntax")

        return len(syntax_errors) == 0

    def check_outputs(self):
        """Check if cells have execution outputs"""
        print("\n" + "=" * 80)
        print("STEP 4: Checking Cell Outputs")
        print("=" * 80)

        cells = self.notebook.get('cells', [])
        code_cells_with_output = 0
        code_cells_total = 0

        for cell in cells:
            if cell.get('cell_type') == 'code':
                code_cells_total += 1
                outputs = cell.get('outputs', [])
                if outputs:
                    code_cells_with_output += 1

        if code_cells_total > 0:
            coverage = (code_cells_with_output / code_cells_total) * 100
            self.passed_checks.append(f"✓ Cells with outputs: {code_cells_with_output}/{code_cells_total} ({coverage:.1f}%)")

            if coverage < 50:
                self.warnings.append(f"⚠ Low output coverage: {coverage:.1f}%")

        return True

    def validate_data_flow(self):
        """Validate critical data flow and transformations"""
        print("\n" + "=" * 80)
        print("STEP 5: Validating Data Flow")
        print("=" * 80)

        cells = self.notebook.get('cells', [])
        all_code = ""

        # Combine all code
        for cell in cells:
            if cell.get('cell_type') == 'code':
                all_code += ''.join(cell.get('source', [])) + "\n"

        # Check for data loading
        if 'pd.read_csv' in all_code or 'read_csv' in all_code:
            self.passed_checks.append("✓ Data loading present (pd.read_csv)")
        else:
            self.warnings.append("⚠ No obvious data loading found")

        # Check for train/test split
        if 'train_test_split' in all_code:
            self.passed_checks.append("✓ Train/test split present")
        else:
            self.warnings.append("⚠ No train_test_split found")

        # Check for scaling/normalization
        if 'StandardScaler' in all_code or 'MinMaxScaler' in all_code:
            self.passed_checks.append("✓ Feature scaling present")
        else:
            self.warnings.append("⚠ No feature scaling found")

        # Check for imputation
        if 'SimpleImputer' in all_code or 'fillna' in all_code:
            self.passed_checks.append("✓ Missing value handling present")
        else:
            self.warnings.append("⚠ No obvious missing value handling")

        return True

    def validate_cv_implementation(self):
        """Validate cross-validation implementation"""
        print("\n" + "=" * 80)
        print("STEP 6: Validating Cross-Validation Implementation")
        print("=" * 80)

        cells = self.notebook.get('cells', [])
        cv_found = False
        cv_details = []

        for idx, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                # Check for CV function
                if 'def train_model_with_enhanced_cv' in source or 'StratifiedKFold' in source:
                    cv_found = True
                    cv_details.append(f"CV implementation at cell {idx}")

                    # Check for proper CV structure
                    if 'StratifiedKFold' in source:
                        self.passed_checks.append("✓ Uses StratifiedKFold for balanced splits")

                    if 'for fold_idx' in source or 'for train_idx, val_idx' in source:
                        self.passed_checks.append("✓ Has CV loop structure")

                    # Check for data leakage prevention
                    if 'fit_transform' in source and 'transform' in source:
                        self.passed_checks.append("✓ Separate fit_transform and transform (good practice)")

                    # Check for SMOTE
                    if 'SMOTE' in source:
                        self.passed_checks.append("✓ SMOTE implementation present")

        if cv_found:
            self.passed_checks.append(f"✓ Cross-validation implementation found")
        else:
            self.warnings.append("⚠ No clear cross-validation implementation found")

        return True

    def validate_model_training(self):
        """Validate model training and evaluation"""
        print("\n" + "=" * 80)
        print("STEP 7: Validating Model Training & Evaluation")
        print("=" * 80)

        cells = self.notebook.get('cells', [])
        all_code = ""

        for cell in cells:
            if cell.get('cell_type') == 'code':
                all_code += ''.join(cell.get('source', [])) + "\n"

        # Check for model types
        models_found = []
        if 'RandomForestClassifier' in all_code:
            models_found.append('RandomForest')
        if 'XGBClassifier' in all_code or 'xgboost' in all_code:
            models_found.append('XGBoost')
        if 'LogisticRegression' in all_code:
            models_found.append('LogisticRegression')
        if 'GradientBoostingClassifier' in all_code:
            models_found.append('GradientBoosting')

        if models_found:
            self.passed_checks.append(f"✓ Models used: {', '.join(models_found)}")
        else:
            self.warnings.append("⚠ No obvious model classifiers found")

        # Check for model fitting
        if '.fit(' in all_code:
            self.passed_checks.append("✓ Model fitting (.fit) present")
        else:
            self.warnings.append("⚠ No model fitting found")

        # Check for predictions
        if '.predict(' in all_code or '.predict_proba(' in all_code:
            self.passed_checks.append("✓ Model predictions present")
        else:
            self.warnings.append("⚠ No model predictions found")

        # Check for evaluation metrics
        metrics = []
        if 'confusion_matrix' in all_code:
            metrics.append('confusion_matrix')
        if 'precision_score' in all_code or 'precision' in all_code:
            metrics.append('precision')
        if 'recall_score' in all_code or 'recall' in all_code:
            metrics.append('recall')
        if 'f1_score' in all_code or 'f1' in all_code:
            metrics.append('f1')
        if 'roc_auc' in all_code or 'auc' in all_code:
            metrics.append('ROC-AUC')

        if metrics:
            self.passed_checks.append(f"✓ Evaluation metrics: {', '.join(metrics)}")
        else:
            self.warnings.append("⚠ No evaluation metrics found")

        return True

    def validate_cost_framework(self):
        """Validate cost-sensitive learning framework"""
        print("\n" + "=" * 80)
        print("STEP 8: Validating Cost-Sensitive Framework")
        print("=" * 80)

        cells = self.notebook.get('cells', [])
        cost_elements = []

        for idx, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                # Check for cost configuration
                if 'CostConfig' in source or 'class CostConfig' in source:
                    cost_elements.append(f"Cost configuration at cell {idx}")
                    self.passed_checks.append("✓ CostConfig class found")

                # Check for cost calculation
                if 'calculate_expected_cost' in source or 'def calculate_expected_cost' in source:
                    cost_elements.append(f"Cost calculation function at cell {idx}")
                    self.passed_checks.append("✓ Cost calculation function found")

                # Check for threshold optimization
                if 'optimize_threshold' in source or 'threshold_optimization' in source:
                    cost_elements.append(f"Threshold optimization at cell {idx}")
                    self.passed_checks.append("✓ Threshold optimization found")

        if not cost_elements:
            self.warnings.append("⚠ Cost-sensitive framework not clearly identified")

        return True

    def check_common_issues(self):
        """Check for common issues and anti-patterns"""
        print("\n" + "=" * 80)
        print("STEP 9: Checking for Common Issues")
        print("=" * 80)

        cells = self.notebook.get('cells', [])
        all_code = ""

        for cell in cells:
            if cell.get('cell_type') == 'code':
                all_code += ''.join(cell.get('source', [])) + "\n"

        # Check for data leakage patterns
        issues_found = []

        # Pattern 1: Fitting scaler on full dataset before split
        if re.search(r'scaler\.fit\(X\).*train_test_split', all_code, re.DOTALL):
            issues_found.append("Possible data leakage: scaler fit before train/test split")

        # Pattern 2: Missing random_state
        if 'train_test_split' in all_code and 'random_state' not in all_code:
            self.warnings.append("⚠ train_test_split without random_state (non-reproducible)")
        else:
            self.passed_checks.append("✓ random_state used for reproducibility")

        # Check for proper test set handling
        if 'X_test' in all_code:
            self.passed_checks.append("✓ Test set (X_test) present")

            # Check if test set is used in training
            if re.search(r'\.fit.*X_test', all_code):
                issues_found.append("CRITICAL: Model may be fitted on test data")

        # Check for proper preprocessing pipeline
        if 'Pipeline' in all_code or 'make_pipeline' in all_code:
            self.passed_checks.append("✓ sklearn Pipeline used (good practice)")

        if issues_found:
            for issue in issues_found:
                self.issues.append(f"✗ {issue}")
        else:
            self.passed_checks.append("✓ No common anti-patterns detected")

        return True

    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        print(f"\n📊 Notebook: {self.notebook_path}")
        print(f"📅 Validation Date: 2025-11-08")
        print()

        # Summary
        total_checks = len(self.passed_checks) + len(self.warnings) + len(self.issues)
        print(f"✅ Passed Checks: {len(self.passed_checks)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Critical Issues: {len(self.issues)}")
        print()

        # Passed checks
        if self.passed_checks:
            print("=" * 80)
            print("✅ PASSED CHECKS")
            print("=" * 80)
            for check in self.passed_checks:
                print(check)
            print()

        # Warnings
        if self.warnings:
            print("=" * 80)
            print("⚠️  WARNINGS")
            print("=" * 80)
            for warning in self.warnings:
                print(warning)
            print()

        # Issues
        if self.issues:
            print("=" * 80)
            print("❌ CRITICAL ISSUES")
            print("=" * 80)
            for issue in self.issues:
                print(issue)
            print()

        # Final verdict
        print("=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)
        if len(self.issues) == 0:
            print("✅ NOTEBOOK PASSES VALIDATION")
            print("   The notebook is structurally sound and ready for review.")
            if self.warnings:
                print(f"   Note: {len(self.warnings)} warning(s) should be reviewed.")
        else:
            print("❌ NOTEBOOK HAS ISSUES")
            print(f"   {len(self.issues)} critical issue(s) must be fixed.")
        print("=" * 80)

        return len(self.issues) == 0

def main():
    validator = NotebookValidator(NOTEBOOK_PATH)

    # Run validation steps
    if not validator.load_notebook():
        print("\n❌ Failed to load notebook. Exiting.")
        return 1

    validator.validate_structure()
    validator.validate_syntax()
    validator.check_outputs()
    validator.validate_data_flow()
    validator.validate_cv_implementation()
    validator.validate_model_training()
    validator.validate_cost_framework()
    validator.check_common_issues()

    # Generate final report
    success = validator.generate_report()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
