#!/usr/bin/env python3
"""Compile all GLSL shaders to SPIR-V format."""

import os
import sys
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

# Shader stage mappings
SHADER_STAGES = {
    '.vert': 'vertex',
    '.frag': 'fragment',
    '.comp': 'compute',
    '.geom': 'geometry',
    '.tesc': 'tess_control',
    '.tese': 'tess_eval'
}

def get_file_hash(filepath: Path) -> str:
    """Get SHA256 hash of a file."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def find_glslc() -> Optional[str]:
    """Find glslc compiler in system PATH or common locations."""
    # Try system PATH first
    try:
        result = subprocess.run(['glslc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return 'glslc'
    except FileNotFoundError:
        pass
    
    # Common installation paths
    common_paths = [
        # Vulkan SDK paths
        os.path.join(os.environ.get('VULKAN_SDK', ''), 'bin', 'glslc'),
        os.path.join(os.environ.get('VULKAN_SDK', ''), 'bin', 'glslc.exe'),
        # Common Linux paths
        '/usr/bin/glslc',
        '/usr/local/bin/glslc',
        # Common Windows paths
        'C:\\VulkanSDK\\*\\Bin\\glslc.exe',
        'C:\\Program Files\\VulkanSDK\\*\\Bin\\glslc.exe',
    ]
    
    for path in common_paths:
        if '*' in path:
            # Handle wildcard paths
            import glob
            matches = glob.glob(path)
            if matches:
                return matches[0]
        elif os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    return None

def compile_shader(glsl_path: Path, spv_path: Path, 
                  stage: str, glslc_path: str) -> bool:
    """Compile a single GLSL shader to SPIR-V."""
    try:
        cmd = [
            glslc_path,
            '-fshader-stage=' + stage,
            '-o', str(spv_path),
            str(glsl_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error compiling {glsl_path}:")
            print(result.stderr)
            return False
        
        print(f"Successfully compiled: {glsl_path} -> {spv_path}")
        return True
        
    except Exception as e:
        print(f"Exception compiling {glsl_path}: {e}")
        return False

def get_shader_stage(filename: str) -> Optional[str]:
    """Determine shader stage from filename."""
    # Check explicit extensions first
    for ext, stage in SHADER_STAGES.items():
        if filename.endswith(ext + '.glsl') or filename.endswith(ext):
            return stage
    
    # Check generic .glsl files
    if filename.endswith('.glsl'):
        # Try to infer from filename
        lower = filename.lower()
        if 'vert' in lower:
            return 'vertex'
        elif 'frag' in lower:
            return 'fragment'
        elif 'comp' in lower:
            return 'compute'
    
    return None

def compile_all_shaders(shader_dir: Path, force: bool = False) -> Dict[str, bool]:
    """Compile all GLSL shaders in directory to SPIR-V."""
    glslc = find_glslc()
    if not glslc:
        print("ERROR: glslc compiler not found!")
        print("Please install Vulkan SDK or ensure glslc is in PATH")
        return {}
    
    print(f"Using glslc: {glslc}")
    
    results = {}
    
    # Find all GLSL files
    glsl_files = list(shader_dir.glob('*.glsl'))
    glsl_files.extend(shader_dir.glob('*.vert'))
    glsl_files.extend(shader_dir.glob('*.frag'))
    glsl_files.extend(shader_dir.glob('*.comp'))
    
    for glsl_file in glsl_files:
        # Skip if already has .spv in name
        if '.spv' in glsl_file.name:
            continue
            
        # Determine output name
        if glsl_file.suffix == '.glsl':
            spv_name = glsl_file.stem + '.spv'
        else:
            spv_name = glsl_file.name + '.spv'
        
        spv_file = glsl_file.parent / spv_name
        
        # Check if compilation needed
        compile_needed = force or not spv_file.exists()
        
        if not compile_needed and spv_file.exists():
            # Check if source is newer than compiled
            if glsl_file.stat().st_mtime > spv_file.stat().st_mtime:
                compile_needed = True
        
        if not compile_needed:
            print(f"Skipping {glsl_file.name} (up to date)")
            results[str(glsl_file)] = True
            continue
        
        # Determine shader stage
        stage = get_shader_stage(glsl_file.name)
        if not stage:
            print(f"WARNING: Cannot determine shader stage for {glsl_file.name}")
            continue
        
        # Compile shader
        success = compile_shader(glsl_file, spv_file, stage, glslc)
        results[str(glsl_file)] = success
    
    return results

def generate_embedded_spirv(shader_dir: Path, output_file: Path):
    """Generate Python file with embedded SPIR-V bytecode."""
    spv_files = list(shader_dir.glob('*.spv'))
    
    with open(output_file, 'w') as f:
        f.write('"""Embedded SPIR-V shader bytecode for fallback."""\n\n')
        f.write('# This file is auto-generated by compile_all_shaders.py\n')
        f.write('# DO NOT EDIT MANUALLY\n\n')
        f.write('from typing import Dict\n\n')
        f.write('EMBEDDED_SPIRV: Dict[str, bytes] = {\n')
        
        for spv_file in spv_files:
            # Read SPIR-V bytes
            with open(spv_file, 'rb') as spv:
                data = spv.read()
            
            # Write as Python bytes literal
            f.write(f'    "{spv_file.stem}": (\n')
            
            # Write bytes in chunks for readability
            for i in range(0, len(data), 16):
                chunk = data[i:i+16]
                hex_str = ' '.join(f'\\x{b:02x}' for b in chunk)
                f.write(f'        b"{hex_str}"\n')
            
            f.write('    ),\n')
        
        f.write('}\n')
    
    print(f"Generated embedded SPIR-V: {output_file}")

def main():
    """Main entry point."""
    # Get shader directory
    script_dir = Path(__file__).parent
    shader_dir = script_dir
    
    print(f"Compiling shaders in: {shader_dir}")
    
    # Parse arguments
    force = '--force' in sys.argv or '-f' in sys.argv
    
    # Compile all shaders
    results = compile_all_shaders(shader_dir, force=force)
    
    # Generate embedded SPIR-V file
    embedded_file = shader_dir / 'embedded_spirv.py'
    generate_embedded_spirv(shader_dir, embedded_file)
    
    # Print summary
    success_count = sum(1 for v in results.values() if v)
    print(f"\nCompilation summary: {success_count}/{len(results)} successful")
    
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())