#!/usr/bin/env python3
"""Add a debug print at the start of __init__.py to verify it's being loaded."""

from pathlib import Path

def add_debug_print():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    init_file_path = project_root / "python" / "vulkan_forge" / "__init__.py"
    
    print(f"Adding debug print to: {init_file_path}")
    
    # Read current content
    with open(init_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add debug print after the docstring
    lines = content.split('\n')
    
    # Find where to insert (after the docstring)
    insert_index = 0
    for i, line in enumerate(lines):
        if i > 0 and lines[i-1].strip().endswith('"""'):
            insert_index = i
            break
    
    # Insert debug print
    debug_line = "\nprint('DEBUG: Loading vulkan_forge/__init__.py from:', __file__)\n"
    lines.insert(insert_index, debug_line)
    
    # Write back
    new_content = '\n'.join(lines)
    with open(init_file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)
    
    print("✓ Debug print added")

if __name__ == "__main__":
    add_debug_print()