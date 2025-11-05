#!/usr/bin/env python3
"""
Integrate enhancement cells into ReneWind_FINAL_Enhanced.ipynb

This script:
1. Reads the original notebook
2. Parses enhancement markdown files
3. Replaces/inserts cells as specified
4. Writes the enhanced notebook
"""

import json
import re
from typing import List, Dict, Tuple


def parse_markdown_file(filepath: str) -> List[Dict]:
    """
    Parse markdown file containing cell definitions.

    Returns:
        List of cell dicts with 'cell_type', 'source', 'metadata', etc.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    cells = []

    # Split by cell headers (## Cell X.Y or X.Y.Z)
    sections = re.split(r'(?=## Cell [\d\.]+ \[)', content)

    for section in sections:
        if not section.strip() or not section.startswith('## Cell'):
            continue

        # Extract cell type from header: ## Cell X.Y [TYPE]
        header_match = re.search(r'## Cell [^\[]+\[([A-Z]+)\]', section)
        if not header_match:
            continue

        cell_type_str = header_match.group(1).strip().upper()

        # Find the code block (```type ... ```)
        code_block_match = re.search(r'```(\w+)\n(.*?)\n```', section, re.DOTALL)
        if not code_block_match:
            continue

        code_type = code_block_match.group(1).lower()
        source_content = code_block_match.group(2)

        # Determine cell type
        if cell_type_str == 'MARKDOWN' or code_type == 'markdown':
            cell_type = 'markdown'
        elif cell_type_str == 'CODE' or code_type == 'python':
            cell_type = 'code'
        else:
            print(f"WARNING: Unknown cell type: {cell_type_str}, {code_type}")
            continue

        # Convert source to list of lines with \n
        source_lines = []
        for line in source_content.split('\n'):
            source_lines.append(line + '\n')

        # Remove trailing newline from last line
        if source_lines and source_lines[-1] == '\n':
            source_lines = source_lines[:-1]

        # Create cell structure
        cell = {
            'cell_type': cell_type,
            'metadata': {},
            'source': source_lines
        }

        # Add execution-specific fields for code cells
        if cell_type == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

        cells.append(cell)

    return cells


def create_enhanced_notebook():
    """
    Main function to create the enhanced notebook.
    """

    print("=" * 70)
    print("JUPYTER NOTEBOOK ENHANCEMENT INTEGRATION")
    print("=" * 70)

    # ========== STEP 1: Read original notebook ==========
    print("\n[1/6] Reading original notebook...")
    with open('/home/user/ReneWind_Assignment/ReneWind_FINAL_Enhanced.ipynb', 'r') as f:
        original_nb = json.load(f)

    original_cells = original_nb['cells']
    print(f"   Original cells: {len(original_cells)}")

    # ========== STEP 2: Parse enhancement files ==========
    print("\n[2/6] Parsing enhancement files...")

    section5_cells = parse_markdown_file(
        '/home/user/ReneWind_Assignment/CELLS_SECTION_5_COST_FRAMEWORK.md'
    )
    print(f"   Section 5: {len(section5_cells)} cells")

    section6_cells = parse_markdown_file(
        '/home/user/ReneWind_Assignment/CELLS_SECTION_6_ENHANCED_CV.md'
    )
    print(f"   Section 6: {len(section6_cells)} cells")

    section6_5_cells = parse_markdown_file(
        '/home/user/ReneWind_Assignment/CELLS_SECTION_6.5_VISUALIZATIONS.md'
    )
    print(f"   Section 6.5 (NEW): {len(section6_5_cells)} cells")

    section7_cells = parse_markdown_file(
        '/home/user/ReneWind_Assignment/CELLS_SECTION_7_MODEL_TRAINING.md'
    )
    print(f"   Section 7 & 7.5: {len(section7_cells)} cells total")

    # Split section 7 cells
    section7_training = section7_cells[:2]  # Cells 7.1, 7.2
    section7_5_viz = section7_cells[2:]     # Cells 7.3-7.6
    print(f"     - Section 7: {len(section7_training)} cells")
    print(f"     - Section 7.5 (NEW): {len(section7_5_viz)} cells")

    # ========== STEP 3: Identify section boundaries ==========
    print("\n[3/6] Identifying section boundaries in original notebook...")

    section_indices = {}
    for i, cell in enumerate(original_cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            # Look for section markers
            if '# **Section 5:' in source:
                section_indices['section5_start'] = i
            elif '# **Section 6:' in source:
                section_indices['section6_start'] = i
            elif '# **Section 7:' in source:
                section_indices['section7_start'] = i
            elif '# **Section 8:' in source:
                section_indices['section8_start'] = i

    print(f"   Section 5 starts at cell: {section_indices.get('section5_start', 'NOT FOUND')}")
    print(f"   Section 6 starts at cell: {section_indices.get('section6_start', 'NOT FOUND')}")
    print(f"   Section 7 starts at cell: {section_indices.get('section7_start', 'NOT FOUND')}")
    print(f"   Section 8 starts at cell: {section_indices.get('section8_start', 'NOT FOUND')}")

    # ========== STEP 4: Build new notebook ==========
    print("\n[4/6] Building enhanced notebook structure...")

    new_cells = []

    # Keep cells 0 to (Section 5 - 1) - Sections 1-4
    section5_idx = section_indices['section5_start']
    section6_idx = section_indices['section6_start']
    section7_idx = section_indices['section7_start']
    section8_idx = section_indices['section8_start']

    # Part 1: Keep Sections 1-4
    new_cells.extend(original_cells[:section5_idx])
    print(f"   Added cells 0-{section5_idx-1} (Sections 1-4): {section5_idx} cells")

    # Part 2: Replace Section 5
    new_cells.extend(section5_cells)
    print(f"   Replaced Section 5: {len(section5_cells)} cells")

    # Part 3: Replace Section 6
    new_cells.extend(section6_cells)
    print(f"   Replaced Section 6: {len(section6_cells)} cells")

    # Part 4: Insert NEW Section 6.5
    new_cells.extend(section6_5_cells)
    print(f"   Inserted Section 6.5 (NEW): {len(section6_5_cells)} cells")

    # Part 5: Replace Section 7
    new_cells.extend(section7_training)
    print(f"   Replaced Section 7: {len(section7_training)} cells")

    # Part 6: Insert NEW Section 7.5
    new_cells.extend(section7_5_viz)
    print(f"   Inserted Section 7.5 (NEW): {len(section7_5_viz)} cells")

    # Part 7: Keep Sections 8-11
    remaining_cells = original_cells[section8_idx:]
    new_cells.extend(remaining_cells)
    print(f"   Added cells {section8_idx}-{len(original_cells)-1} (Sections 8-11): {len(remaining_cells)} cells")

    print(f"\n   TOTAL NEW CELLS: {len(new_cells)}")

    # ========== STEP 5: Create new notebook ==========
    print("\n[5/6] Creating new notebook JSON...")

    new_notebook = {
        'cells': new_cells,
        'metadata': original_nb.get('metadata', {}),
        'nbformat': original_nb.get('nbformat', 4),
        'nbformat_minor': original_nb.get('nbformat_minor', 5)
    }

    # ========== STEP 6: Write to file ==========
    output_path = '/home/user/ReneWind_Assignment/ReneWind_FINAL_Enhanced_With_Visualizations.ipynb'
    print(f"\n[6/6] Writing enhanced notebook to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(new_notebook, f, indent=1)

    print(f"   File written successfully!")

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)
    print(f"Original notebook cells:  {len(original_cells)}")
    print(f"Enhanced notebook cells:  {len(new_cells)}")
    print(f"Cells added:              {len(new_cells) - len(original_cells)}")
    print()
    print("Sections modified:")
    print(f"  • Section 5:   REPLACED ({len(section5_cells)} cells)")
    print(f"  • Section 6:   REPLACED ({len(section6_cells)} cells)")
    print(f"  • Section 6.5: NEW      ({len(section6_5_cells)} cells)")
    print(f"  • Section 7:   REPLACED ({len(section7_training)} cells)")
    print(f"  • Section 7.5: NEW      ({len(section7_5_viz)} cells)")
    print()
    print("Sections preserved:")
    print(f"  • Sections 1-4 (Environment, Data, EDA, Preprocessing)")
    print(f"  • Sections 8-11 (Comparison, Test Eval, Insights, Conclusions)")
    print("=" * 70)

    return new_notebook, output_path


if __name__ == '__main__':
    notebook, path = create_enhanced_notebook()
    print(f"\n✅ SUCCESS! Enhanced notebook created at:\n   {path}")
