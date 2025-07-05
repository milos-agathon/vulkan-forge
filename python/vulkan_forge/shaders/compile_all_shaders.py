#!/usr/bin/env python3
"""
Compile GLSL shaders to SPIR-V for Vulkan
Requires Vulkan SDK to be installed (provides glslc compiler)
"""

import subprocess
import os
import sys
from pathlib import Path


def find_glslc():
    """Find the glslc compiler from Vulkan SDK"""
    # Common locations for glslc
    possible_paths = [
        # Windows
        r"C:\VulkanSDK\*\Bin\glslc.exe",
        r"C:\VulkanSDK\*\Bin32\glslc.exe",
        # Linux
        "/usr/bin/glslc",
        "/usr/local/bin/glslc",
        # macOS
        "/usr/local/bin/glslc",
    ]
    
    # Check environment variable
    vulkan_sdk = os.environ.get('VULKAN_SDK')
    if vulkan_sdk:
        possible_paths.insert(0, os.path.join(vulkan_sdk, 'Bin', 'glslc.exe'))
        possible_paths.insert(1, os.path.join(vulkan_sdk, 'Bin', 'glslc'))
    
    # Try to find glslc
    for path_pattern in possible_paths:
        from glob import glob
        matches = glob(path_pattern)
        if matches and os.path.exists(matches[0]):
            return matches[0]
    
    # Try running glslc directly (if it's in PATH)
    try:
        subprocess.run(['glslc', '--version'], capture_output=True, check=True)
        return 'glslc'
    except:
        pass
    
    return None


def compile_shader(glslc_path, input_file, output_file):
    """Compile a single shader file"""
    print(f"Compiling {input_file} -> {output_file}")
    
    try:
        result = subprocess.run(
            [glslc_path, '-o', output_file, input_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed:")
        print(f"    {e.stderr}")
        return False


def convert_spirv_to_header(spirv_file, header_file, array_name):
    """Convert SPIR-V binary to C++ header file"""
    print(f"Converting {spirv_file} -> {header_file}")
    
    with open(spirv_file, 'rb') as f:
        spirv_data = f.read()
    
    # Convert to uint32_t array
    uint32_values = []
    for i in range(0, len(spirv_data), 4):
        # Little-endian uint32
        value = int.from_bytes(spirv_data[i:i+4], byteorder='little')
        uint32_values.append(value)
    
    # Write header file
    with open(header_file, 'w') as f:
        f.write(f"// Auto-generated from {spirv_file}\n")
        f.write(f"// Do not edit manually\n\n")
        f.write(f"static const uint32_t {array_name}[] = {{\n")
        
        # Write values in rows of 8
        for i in range(0, len(uint32_values), 8):
            row = uint32_values[i:i+8]
            hex_values = [f"0x{v:08x}" for v in row]
            f.write("    " + ", ".join(hex_values))
            if i + 8 < len(uint32_values):
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        f.write(f"static const size_t {array_name}_size = sizeof({array_name});\n")
        f.write(f"static const size_t {array_name}_count = sizeof({array_name}) / sizeof(uint32_t);\n")
    
    print(f"  ✓ Generated header with {len(uint32_values)} uint32_t values")


def main():
    # Find shader files
    shader_dir = Path("shaders")
    if not shader_dir.exists():
        shader_dir = Path("cpp/shaders")
    
    if not shader_dir.exists():
        print(f"Error: Shader directory not found")
        print(f"Please run this script from the project root")
        return 1
    
    # Find glslc compiler
    glslc = find_glslc()
    if not glslc:
        print("Error: Could not find glslc compiler")
        print("Please install Vulkan SDK and ensure glslc is in your PATH")
        print("Download from: https://vulkan.lunarg.com/")
        return 1
    
    print(f"Using glslc: {glslc}\n")
    
    # Create output directory
    output_dir = Path("build/shaders")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    include_dir = Path("cpp/include/shaders")
    include_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile shaders
    shaders = [
        ("height_field.vert", "vertexShaderSpirv"),
        ("height_field.frag", "fragmentShaderSpirv"),
        ("vertex.glsl", "vertexShaderSpirv"),
        ("fragment.glsl", "fragmentShaderSpirv"),
    ]

    # Additional point shaders (if they exist)
    point_shaders = [
        ("point_vertex.glsl", "pointVertexShaderSpirv"),
        ("point_fragment.glsl", "pointFragmentShaderSpirv"),
    ]
    
    
    success = True
    for shader_file, array_name in shaders:
        input_path = shader_dir / shader_file
        spirv_path = output_dir / f"{shader_file}.spv"
        header_path = include_dir / f"{shader_file}.h"
        
        if not input_path.exists():
            print(f"Error: Shader file not found: {input_path}")
            success = False
            continue
        
        # Compile to SPIR-V
        if compile_shader(glslc, str(input_path), str(spirv_path)):
            # Convert to header
            convert_spirv_to_header(str(spirv_path), str(header_path), array_name)
        else:
            success = False
           
    # Try to compile point shaders but don't fail if they don't exist
    for shader_file, array_name in point_shaders:
        input_path = shader_dir / shader_file
        if input_path.exists():
            spirv_path = output_dir / f"{shader_file}.spv"
            header_path = include_dir / f"{shader_file}.h"
            
            if compile_shader(glslc, str(input_path), str(spirv_path)):
                convert_spirv_to_header(str(spirv_path), str(header_path), array_name)
            else:
                print(f"Warning: Failed to compile optional shader: {shader_file}")
        else:
            print(f"Info: Optional shader not found: {shader_file}")
    
    if success:
        print("\n✓ All shaders compiled successfully!")
        print(f"  SPIR-V files in: {output_dir}")
        print(f"  Header files in: {include_dir}")
    else:
        print("\n✗ Some shaders failed to compile")
        return 1
    
    # Also create a combined header
    combined_header = include_dir / "all_shaders.h"
    with open(combined_header, 'w') as f:
        f.write("// Combined shader header\n")
        f.write("#pragma once\n\n")
        for shader_file, _ in shaders:
            f.write(f'#include "{shader_file}.h"\n')
    
    print(f"\n  Combined header: {combined_header}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())