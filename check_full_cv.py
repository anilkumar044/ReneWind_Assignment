#!/usr/bin/env python3
import json

with open('ReneWind_FINAL_PRODUCTION.ipynb', 'r') as f:
    notebook = json.load(f)

# Combine both cells
cell1_source = ''.join(notebook['cells'][43]['source'])
cell2_source = ''.join(notebook['cells'][44]['source'])

full_source = cell1_source + "\n" + cell2_source
lines = full_source.split('\n')

print("=" * 80)
print("COMPLETE CV FUNCTION STRUCTURE")
print("=" * 80)

# Find key lines
for_line = None
append_line = None
return_line = None

for i, line in enumerate(lines):
    if 'for fold_idx' in line and 'enumerate' in line:
        for_line = i
    if 'fold_results.append(fold_data)' in line:
        append_line = i
    if 'return fold_results' in line:
        return_line = i

print(f"\nFor loop starts at line: {for_line}")
print(f"Append at line: {append_line}")
print(f"Return at line: {return_line}")

if all(x is not None for x in [for_line, append_line, return_line]):
    for_indent = len(lines[for_line]) - len(lines[for_line].lstrip())
    append_indent = len(lines[append_line]) - len(lines[append_line].lstrip())
    return_indent = len(lines[return_line]) - len(lines[return_line].lstrip())
    
    print(f"\nFor loop indent: {for_indent}")
    print(f"Append indent: {append_indent}")
    print(f"Return indent: {return_indent}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    if append_indent > for_indent:
        print("✓ CORRECT: fold_results.append() is INSIDE the loop")
    else:
        print("✗ ERROR: fold_results.append() is OUTSIDE the loop")
    
    if return_indent == for_indent:
        print("✓ CORRECT: return statement is OUTSIDE the loop (at function level)")
    else:
        print("⚠ WARNING: return indent may be incorrect")
    
    # Show the region around loop closure
    print("\n" + "=" * 80)
    print("LOOP CLOSURE REGION")
    print("=" * 80)
    
    for i in range(append_line, min(append_line + 25, len(lines))):
        line = lines[i]
        if line.strip():
            indent = len(line) - len(line.lstrip())
            marker = ""
            if i == append_line:
                marker = "  <-- APPEND (inside loop)"
            elif indent <= for_indent and i > append_line:
                marker = "  <-- LOOP ENDS HERE (dedented to for level or less)"
                print(f"Line {i:3d} (indent={indent:2d}): {line.strip()[:70]}{marker}")
                break
            print(f"Line {i:3d} (indent={indent:2d}): {line.strip()[:70]}{marker}")

