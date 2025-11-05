#!/usr/bin/env python3
import json

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

cell = notebook['cells'][44]  # Second part of CV function
source = ''.join(cell['source'])
lines = source.split('\n')

print("=" * 80)
print("VERIFYING LOOP CLOSURE IN CELL 44")
print("=" * 80)

# Find the for loop line
for_line_idx = None
append_line_idx = None
return_line_idx = None

for i, line in enumerate(lines):
    if 'for fold_idx' in line and 'enumerate' in line:
        for_line_idx = i
    if 'fold_results.append' in line:
        append_line_idx = i
    if 'return fold_results' in line:
        return_line_idx = i

print(f"\nFor loop at line: {for_line_idx}")
print(f"Append at line: {append_line_idx}")
print(f"Return at line: {return_line_idx}")

if for_line_idx is not None:
    for_indent = len(lines[for_line_idx]) - len(lines[for_line_idx].lstrip())
    print(f"\nFor loop indent: {for_indent}")

if append_line_idx is not None:
    append_indent = len(lines[append_line_idx]) - len(lines[append_line_idx].lstrip())
    print(f"Append indent: {append_indent}")
    
    # Show lines around append
    print("\n" + "=" * 80)
    print("LINES AROUND APPEND (showing indent and content)")
    print("=" * 80)
    
    for i in range(max(0, append_line_idx - 2), min(len(lines), append_line_idx + 10)):
        line = lines[i]
        if line.strip():  # Skip empty lines for clarity
            indent = len(line) - len(line.lstrip())
            marker = "  <-- APPEND" if i == append_line_idx else ""
            marker += "  <-- RETURN" if i == return_line_idx else ""
            print(f"Line {i:3d} (indent={indent:2d}): {line[:70]}{marker}")

# Determine if loop closes properly
if for_line_idx is not None and append_line_idx is not None and return_line_idx is not None:
    for_indent = len(lines[for_line_idx]) - len(lines[for_line_idx].lstrip())
    append_indent = len(lines[append_line_idx]) - len(lines[append_line_idx].lstrip())
    return_indent = len(lines[return_line_idx]) - len(lines[return_line_idx].lstrip())
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print(f"For loop indent: {for_indent}")
    print(f"Append indent: {append_indent}")
    print(f"Return indent: {return_indent}")
    
    if append_indent > for_indent:
        print("\n✓ CORRECT: append is inside the loop (indented more than for)")
    else:
        print("\n✗ ERROR: append is NOT inside the loop")
    
    if return_indent == for_indent:
        print("✓ CORRECT: return is outside the loop (same indent as for)")
    else:
        print("⚠ WARNING: return indent may be incorrect")
    
    # Find first dedented line after append
    for i in range(append_line_idx + 1, len(lines)):
        if lines[i].strip() and not lines[i].strip().startswith('#'):
            line_indent = len(lines[i]) - len(lines[i].lstrip())
            if line_indent <= for_indent:
                print(f"\n✓ Loop closes at line {i}")
                print(f"  First dedented statement: {lines[i].strip()[:60]}")
                break

