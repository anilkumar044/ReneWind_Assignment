#!/usr/bin/env python3
import json

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

print("=" * 80)
print("VERIFYING TEST EVALUATION PREPROCESSING")
print("=" * 80)

# Find the preprocessing cell
preprocessing_cell = None
for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'PREPROCESS FULL TRAINING DATA FOR FINAL MODEL' in source:
            preprocessing_cell = idx
            print(f"\n✓ Found preprocessing cell at index {idx}")
            print("\nPreprocessing cell content:")
            print("-" * 80)
            print(source[:500] + "...")
            break

# Find the cell that uses preprocessed data
test_eval_cell = None
for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'X_train_full' in source and 'y_train_full' in source:
            test_eval_cell = idx
            print(f"\n✓ Found test evaluation cell at index {idx}")
            
            # Check if it uses preprocessed data
            if 'X_train_scaled' in source:
                print("  ✓ Uses X_train_scaled (preprocessed)")
            else:
                print("  ✗ Still uses raw X")
            
            # Show relevant lines
            print("\nRelevant lines from test evaluation cell:")
            print("-" * 80)
            for line in source.split('\n')[:30]:
                if 'X_train_full' in line or 'X_test' in line or 'y_train_full' in line:
                    print(f"  {line}")
            break

# Verify order
if preprocessing_cell and test_eval_cell:
    print("\n" + "=" * 80)
    print("CELL ORDER VERIFICATION")
    print("=" * 80)
    if preprocessing_cell < test_eval_cell:
        print(f"✓ CORRECT: Preprocessing (cell {preprocessing_cell}) comes BEFORE test eval (cell {test_eval_cell})")
    else:
        print(f"✗ ERROR: Preprocessing (cell {preprocessing_cell}) comes AFTER test eval (cell {test_eval_cell})")

# Check if test predictions use preprocessed data
print("\n" + "=" * 80)
print("CHECKING TEST PREDICTIONS")
print("=" * 80)

for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'final_model.predict' in source:
            print(f"\nCell {idx} contains final_model.predict:")
            
            for line in source.split('\n'):
                if 'predict' in line:
                    print(f"  {line.strip()}")
                    if 'X_test_scaled' in line:
                        print("    ✓ Uses X_test_scaled (correct)")
                    elif 'test_data[features]' in line:
                        print("    ✗ Uses raw test_data[features] (incorrect)")

