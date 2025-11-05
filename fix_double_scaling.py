#!/usr/bin/env python3
"""
Fix the double-scaling issue in test evaluation
"""

import json
import re

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

print("=" * 80)
print("FIXING DOUBLE-SCALING ISSUE")
print("=" * 80)

# Find cell 64 (test evaluation)
cell_64 = notebook['cells'][64]
source = ''.join(cell_64['source'])

print("\nOriginal cell 64 (first 50 lines):")
print("-" * 80)
for i, line in enumerate(source.split('\n')[:50]):
    print(f"{i:3d}: {line}")

# The issue: cell 64 tries to scale already-scaled data
# We need to remove the redundant scaling lines and fix variable names

# Replace the problematic lines
new_source = source

# Remove/fix the problematic scaling
new_source = re.sub(
    r'X_test_final = test_features\.values\s*\n',
    '',
    new_source
)

new_source = re.sub(
    r'X_train_scaled = scaler_final\.fit_transform\(X_train_full\)\s*\n',
    '',
    new_source
)

new_source = re.sub(
    r'X_test_scaled = scaler_final\.transform\(X_test_final\)\s*\n',
    '',
    new_source
)

# Fix double-scaled variable name
new_source = new_source.replace('X_test_scaled_scaled', 'X_test_scaled')

# Also remove scaler_final if it's being created
new_source = re.sub(
    r'scaler_final = StandardScaler\(\)\s*\n',
    '',
    new_source
)

# Update the cell
notebook['cells'][64]['source'] = new_source.split('\n')
notebook['cells'][64]['source'] = [
    line + '\n' if i < len(notebook['cells'][64]['source']) - 1 else line
    for i, line in enumerate(notebook['cells'][64]['source'])
]

print("\n" + "=" * 80)
print("UPDATED CELL 64")
print("=" * 80)
print("Changes made:")
print("  - Removed redundant X_test_final assignment")
print("  - Removed redundant scaler_final creation")
print("  - Removed redundant X_train_scaled scaling")
print("  - Removed redundant X_test_scaled scaling")
print("  - Fixed X_test_scaled_scaled to X_test_scaled")

# Save
with open('ReneWind_FINAL_PRODUCTION.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\n✓ Saved updated notebook")

# Verify
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)
print("\nUpdated cell 64 (first 50 lines):")
print("-" * 80)
new_source_lines = ''.join(notebook['cells'][64]['source']).split('\n')
for i, line in enumerate(new_source_lines[:50]):
    if line.strip():
        print(f"{i:3d}: {line}")

