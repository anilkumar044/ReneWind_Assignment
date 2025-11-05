#!/usr/bin/env python3
"""
Verify CV loop structure in the production notebook
"""

import json

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

print("=" * 80)
print("VERIFYING CV LOOP STRUCTURE")
print("=" * 80)
print()

# Find the CV function cell
for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def train_model_with_enhanced_cv' in source:
            print(f"Found train_model_with_enhanced_cv in cell {idx}")
            print()

            # Check the structure
            lines = source.split('\n')

            # Find the for loop
            for_loop_line = None
            append_line = None
            summary_line = None

            for i, line in enumerate(lines):
                if 'for fold_idx' in line and 'enumerate' in line:
                    for_loop_line = i
                    print(f"✓ Loop starts at line {i}:")
                    print(f"  {line.strip()}")
                
                if 'fold_results.append(fold_data)' in line:
                    append_line = i
                    # Check indentation
                    indent = len(line) - len(line.lstrip())
                    print(f"\n✓ fold_results.append at line {i} (indent={indent}):")
                    print(f"  {line.strip()}")
                
                if 'Summary statistics' in line or 'avg_' in line or 'mean(' in line or 'std(' in line:
                    if summary_line is None and i > (append_line or 0):
                        summary_line = i
                        indent = len(line) - len(line.lstrip())
                        print(f"\n✓ Summary statistics at line {i} (indent={indent}):")
                        print(f"  {line.strip()}")
                        break

            # Verify indentation
            if for_loop_line and append_line:
                for_indent = len(lines[for_loop_line]) - len(lines[for_loop_line].lstrip())
                append_indent = len(lines[append_line]) - len(lines[append_line].lstrip())
                
                print("\n" + "=" * 80)
                print("INDENTATION ANALYSIS")
                print("=" * 80)
                print(f"For loop indent: {for_indent}")
                print(f"Append indent: {append_indent}")
                
                if append_indent > for_indent:
                    print("✓ CORRECT: append is indented MORE than for loop (inside loop)")
                else:
                    print("✗ ERROR: append is NOT properly indented (outside loop)")
                
                if summary_line:
                    summary_indent = len(lines[summary_line]) - len(lines[summary_line].lstrip())
                    print(f"Summary indent: {summary_indent}")
                    
                    if summary_indent == for_indent:
                        print("✓ CORRECT: summary is at SAME level as for loop (outside loop)")
                    else:
                        print("⚠ WARNING: summary indent may be incorrect")

            break

print()
print("=" * 80)
