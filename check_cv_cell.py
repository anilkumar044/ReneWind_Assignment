#!/usr/bin/env python3
import json

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the CV function cell
for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def train_model_with_enhanced_cv' in source:
            print(f"Cell {idx} contains train_model_with_enhanced_cv")
            print("=" * 80)
            print(source)
            print("=" * 80)
            
            # Check for key components
            has_for_loop = 'for fold_idx' in source
            has_append = 'fold_results.append' in source
            has_return = 'return fold_results' in source
            
            print("\nKey components:")
            print(f"✓ Has for loop: {has_for_loop}")
            print(f"✓ Has append: {has_append}")
            print(f"✓ Has return: {has_return}")
            
            if has_append:
                # Find the append line
                for line in source.split('\n'):
                    if 'fold_results.append' in line:
                        print(f"\nAppend line: {repr(line)}")
            break
