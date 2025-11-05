#!/usr/bin/env python3
"""
Fix test data usage in final model evaluation
"""

import json

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

print("=" * 80)
print("FIXING TEST DATA USAGE")
print("=" * 80)

# Cell 64 - update to use correct variables
cell_64 = notebook['cells'][64]
source = ''.join(cell_64['source'])

# The issue: test_features is trying to use raw test_data
# We already have X_test_scaled from cell 63

# Replace the problematic lines
new_source = source

# Remove test_features line and y_test_final extraction from raw data
new_source = new_source.replace(
    'test_features = test_data.drop(columns=[\'Target\'], errors=\'ignore\')',
    '# X_test_scaled already prepared in preprocessing cell above'
)

new_source = new_source.replace(
    'y_test_final = test_data[\'Target\'].values if \'Target\' in test_data.columns else None',
    'y_test_final = test_data[\'Target\'].values if \'Target\' in test_data.columns else None'
)

# Also fix the fit call to use X_train_full directly (not X_train_scaled which is the same)
new_source = new_source.replace(
    '    X_train_scaled, y_train_full,',
    '    X_train_full, y_train_full,'
)

# Update the cell
notebook['cells'][64]['source'] = new_source.split('\n')
notebook['cells'][64]['source'] = [
    line + '\n' if i < len(notebook['cells'][64]['source']) - 1 else line
    for i, line in enumerate(notebook['cells'][64]['source'])
]

print("✓ Updated cell 64:")
print("  - Removed redundant test_features extraction")
print("  - Using X_test_scaled from preprocessing cell")
print("  - Using X_train_full (already scaled) for training")

# Save
with open('ReneWind_FINAL_PRODUCTION.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\n✓ Saved updated notebook")

