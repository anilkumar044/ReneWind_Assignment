#!/usr/bin/env python3
import json

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

print("=" * 80)
print("SEARCHING FOR CV FUNCTION PARTS")
print("=" * 80)

# Find all cells related to CV function
cv_cells = []
for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if ('train_model_with_enhanced_cv' in source or
            'Continue in next cell' in source or
            'fold_results.append' in source or
            'return fold_results' in source):
            cv_cells.append(idx)
            print(f"\nCell {idx}:")
            if 'def train_model_with_enhanced_cv' in source:
                print("  ✓ Contains function definition")
            if 'for fold_idx' in source:
                print("  ✓ Contains for loop start")
            if 'fold_results.append' in source:
                print("  ✓ Contains fold_results.append")
            if 'return fold_results' in source:
                print("  ✓ Contains return statement")
            if 'Continue in next cell' in source:
                print("  ⚠ Multi-cell function (continues)")

print(f"\n\nCV function spans cells: {cv_cells}")

# Check the second part
if len(cv_cells) >= 2:
    print("\n" + "=" * 80)
    print(f"CHECKING CELL {cv_cells[1]} (continuation)")
    print("=" * 80)
    
    cell = notebook['cells'][cv_cells[1]]
    source = ''.join(cell['source'])
    
    # Show last 50 lines to see the end of the loop
    lines = source.split('\n')
    print(f"\nTotal lines: {len(lines)}")
    
    # Find key lines
    for i, line in enumerate(lines):
        if 'fold_results.append' in line:
            print(f"\nLine {i} (append):")
            print(f"  {repr(line)}")
            # Show context
            indent = len(line) - len(line.lstrip())
            print(f"  Indent: {indent} spaces")
            
            # Check next few lines to see if loop ends
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith('#'):
                    next_indent = len(lines[j]) - len(lines[j].lstrip())
                    print(f"\nLine {j} (next statement):")
                    print(f"  {repr(lines[j])}")
                    print(f"  Indent: {next_indent} spaces")
                    
                    if next_indent <= indent - 4:
                        print("  ✓ Loop appears to close here (dedented)")
                    break
