#!/usr/bin/env python3
"""
Enhanced GLSL to SPIR-V compiler for Vulkan Forge mesh pipeline
Supports optimization, validation, and embedded fallbacks
Requires Vulkan SDK (provides glslc compiler)
"""

import subprocess
import os
import sys
import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ShaderConfig:
    """Configuration for a single shader"""
    source_file: str
    array_name: str
    shader_type: str  # vertex, fragment, compute
    optimization_level: str = "performance"  # none, size, performance
    
    
class ShaderCompiler:
    """Enhanced shader compiler with optimization and validation"""
    
    def __init__(self, vulkan_version: str = "1.2", debug: bool = False):
        self.vulkan_version = vulkan_version
        self.debug = debug
        self.glslc_path = self._find_glslc()
        self.stats = {
            'compiled': 0,
            'failed': 0,
            'cached': 0,
            'total_size': 0
        }
        
    def _find_glslc(self) -> Optional[str]:
        """Find the glslc compiler from Vulkan SDK"""
        # Check environment variable first
        vulkan_sdk = os.environ.get('VULKAN_SDK')
        if vulkan_sdk:
            candidates = [
                os.path.join(vulkan_sdk, 'Bin', 'glslc.exe'),
                os.path.join(vulkan_sdk, 'Bin', 'glslc'),
                os.path.join(vulkan_sdk, 'bin', 'glslc'),
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
        
        # Try to find in PATH
        try:
            result = subprocess.run(['glslc', '--version'], 
                                  capture_output=True, check=True)
            return 'glslc'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Common installation paths
        common_paths = [
            # Windows
            r"C:\VulkanSDK\*\Bin\glslc.exe",
            r"C:\VulkanSDK\*\bin\glslc.exe", 
            # Linux
            "/usr/bin/glslc",
            "/usr/local/bin/glslc",
            "/opt/vulkan-sdk/bin/glslc",
            # macOS
            "/usr/local/bin/glslc",
            "/opt/homebrew/bin/glslc",
        ]
        
        import glob
        for path_pattern in common_paths:
            matches = glob.glob(path_pattern)
            if matches and os.path.exists(matches[0]):
                return matches[0]
        
        return None
    
    def _validate_spirv(self, spirv_file: Path) -> bool:
        """Validate SPIR-V binary format and magic number"""
        try:
            with open(spirv_file, 'rb') as f:
                magic_bytes = f.read(4)
                
            # SPIR-V magic number: 0x07230203 (little-endian)
            expected_magic = b'\x03\x02\x23\x07'
            if magic_bytes != expected_magic:
                print(f"  ✗ Invalid SPIR-V magic number in {spirv_file}")
                return False
                
            # Basic size validation
            file_size = spirv_file.stat().st_size
            if file_size < 20:  # Minimum SPIR-V header size
                print(f"  ✗ SPIR-V file too small: {file_size} bytes")
                return False
                
            if file_size % 4 != 0:
                print(f"  ✗ SPIR-V file size not aligned to 4 bytes: {file_size}")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error validating SPIR-V: {e}")
            return False
    
    def _get_cache_key(self, shader_config: ShaderConfig, source_content: str) -> str:
        """Generate cache key for compiled shader"""
        content_hash = hashlib.sha256(source_content.encode()).hexdigest()[:16]
        config_str = f"{shader_config.shader_type}_{shader_config.optimization_level}_{self.vulkan_version}"
        return f"{config_str}_{content_hash}"
    
    def _compile_shader(self, 
                       input_file: Path, 
                       output_file: Path, 
                       config: ShaderConfig) -> bool:
        """Compile a single shader file to SPIR-V"""
        
        if not input_file.exists():
            print(f"  ✗ Shader source not found: {input_file}")
            return False
        
        # Read source for cache key
        try:
            source_content = input_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"  ✗ Error reading shader source: {e}")
            return False
        
        # Check cache
        cache_key = self._get_cache_key(config, source_content)
        cache_file = output_file.parent / f".{output_file.stem}_{cache_key}.spv"
        
        if cache_file.exists() and self._validate_spirv(cache_file):
            # Use cached version
            if cache_file != output_file:
                import shutil
                shutil.copy2(cache_file, output_file)
            print(f"  ↺ Using cached SPIR-V")
            self.stats['cached'] += 1
            return True
        
        # Build glslc command
        cmd = [self.glslc_path]
        
        # Output file
        cmd.extend(['-o', str(output_file)])
        
        # Vulkan target environment
        cmd.extend(['--target-env', f'vulkan{self.vulkan_version}'])
        
        # Optimization flags
        if config.optimization_level == 'performance':
            cmd.append('-O')
        elif config.optimization_level == 'size':
            cmd.extend(['-Os'])
        elif config.optimization_level != 'none':
            print(f"  ! Unknown optimization level: {config.optimization_level}")
        
        # Debug information
        if self.debug:
            cmd.extend(['-g', '-O0'])
        
        # Shader stage detection (if not auto-detected by extension)
        stage_map = {
            'vertex': '-fshader-stage=vertex',
            'fragment': '-fshader-stage=fragment', 
            'compute': '-fshader-stage=compute',
            'geometry': '-fshader-stage=geometry',
            'tesscontrol': '-fshader-stage=tesscontrol',
            'tessevaluation': '-fshader-stage=tessevaluation'
        }
        
        if config.shader_type in stage_map:
            cmd.append(stage_map[config.shader_type])
        
        # Additional useful flags
        cmd.extend([
            '-Werror',  # Treat warnings as errors
            '-fauto-map-locations',  # Auto-assign locations
        ])
        
        # Input file
        cmd.append(str(input_file))
        
        print(f"  Compiling {input_file.name} -> {output_file.name}")
        if self.debug:
            print(f"    Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30  # Prevent hanging
            )
            
            # Validate output
            if not self._validate_spirv(output_file):
                return False
            
            # Cache successful compilation
            if cache_file != output_file:
                import shutil
                shutil.copy2(output_file, cache_file)
            
            # Update stats
            file_size = output_file.stat().st_size
            self.stats['compiled'] += 1
            self.stats['total_size'] += file_size
            
            print(f"    ✓ Success ({file_size:,} bytes)")
            
            if result.stdout and self.debug:
                print(f"    Output: {result.stdout}")
                
            return True
            
        except subprocess.TimeoutExpired:
            print(f"    ✗ Compilation timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Compilation failed:")
            if e.stderr:
                for line in e.stderr.strip().split('\n'):
                    print(f"      {line}")
            return False
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            return False
    
    def _convert_spirv_to_header(self, 
                                spirv_file: Path, 
                                header_file: Path, 
                                array_name: str) -> bool:
        """Convert SPIR-V binary to C++ header file"""
        
        try:
            with open(spirv_file, 'rb') as f:
                spirv_data = f.read()
        except Exception as e:
            print(f"  ✗ Error reading SPIR-V file: {e}")
            return False
        
        if len(spirv_data) % 4 != 0:
            print(f"  ✗ SPIR-V data not aligned to 4 bytes")
            return False
        
        # Convert to uint32_t array
        uint32_values = []
        for i in range(0, len(spirv_data), 4):
            # Little-endian uint32
            value = int.from_bytes(spirv_data[i:i+4], byteorder='little')
            uint32_values.append(value)
        
        # Generate header content
        header_content = [
            f"// Auto-generated from {spirv_file.name}",
            f"// SPIR-V size: {len(spirv_data):,} bytes ({len(uint32_values):,} uint32_t values)",
            f"// Generated by vulkan_forge shader compiler",
            "",
            "#pragma once",
            "#include <cstdint>",
            "",
            f"static const uint32_t {array_name}[] = {{",
        ]
        
        # Write values in rows of 8 for readability
        for i in range(0, len(uint32_values), 8):
            row = uint32_values[i:i+8]
            hex_values = [f"0x{v:08x}" for v in row]
            line = "    " + ", ".join(hex_values)
            if i + 8 < len(uint32_values):
                line += ","
            header_content.append(line)