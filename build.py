#!/usr/bin/env python3
"""
Complete build script for vulkan-forge
Handles shader compilation and module building
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_requirements():
    """Check that all build requirements are met"""
    print("Checking build requirements...")
    
    # Check for Python
    print(f"  Python: {sys.version}")
    
    # Check for Vulkan SDK
    vulkan_sdk = os.environ.get('VULKAN_SDK')
    if vulkan_sdk and os.path.exists(vulkan_sdk):
        print(f"  Vulkan SDK: {vulkan_sdk}")
    else:
        print("  WARNING: VULKAN_SDK environment variable not set or invalid")
        print("  Download from: https://vulkan.lunarg.com/")
    
    # Check for CMake
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  CMake: {result.stdout.splitlines()[0]}")
    except:
        print("  ERROR: CMake not found. Please install CMake >= 3.24")
        return False
    
    # Check for a C++ compiler
    if sys.platform == 'win32':
        # Check for MSVC
        try:
            result = subprocess.run(['cl'], capture_output=True, text=True)
            print("  MSVC: Found")
        except:
            print("  WARNING: MSVC not found. Install Visual Studio 2019 or later")
    else:
        # Check for g++ or clang++
        for compiler in ['g++', 'clang++']:
            try:
                result = subprocess.run([compiler, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  C++ Compiler: {compiler}")
                    break
            except:
                continue
    
    return True


def setup_directory_structure():
    """Create necessary directories"""
    print("\nSetting up directory structure...")
    
    dirs = [
        'cpp/src',
        'cpp/include',
        'cpp/shaders',
        'python/vulkan_forge',
        'build',
        'tests',
        'examples',
        'docs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")


def write_file_if_not_exists(path, content):
    """Write file only if it doesn't exist"""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(content)
        print(f"  Created: {path}")
    else:
        print(f"  Exists: {path}")


def create_project_files():
    """Create all necessary project files"""
    print("\nCreating project files...")
    
    # Create README
    write_file_if_not_exists('README.md', """# Vulkan Forge

A high-performance 3D height field renderer for Python using Vulkan.

## Features

- GPU-accelerated rendering using Vulkan
- Support for large height fields
- Customizable lighting and shading
- Integration with matplotlib
- Cross-platform support

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from vulkan_forge import VulkanRenderer

# Create height data
terrain = np.sin(np.mgrid[0:10:0.1, 0:10:0.1].sum(axis=0))

# Create renderer
renderer = VulkanRenderer(800, 600)

# Render
image = renderer.render_heightfield(terrain, z_scale=2.0)
```

## License

MIT License
""")
    
    # Create .gitignore
    write_file_if_not_exists('.gitignore', """# Build files
build/
_skbuild/
*.egg-info/
dist/
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
*.dll
*.dylib

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Generated files
*.spv
generated/
""")
    
    # Create setup.py for compatibility
    write_file_if_not_exists('setup.py', """from setuptools import setup

# Minimal setup.py for compatibility
# The actual build is handled by scikit-build-core via pyproject.toml
setup()
""")


def compile_shaders():
    """Compile GLSL shaders to SPIR-V"""
    print("\nCompiling shaders...")
    
    # Run the shader compilation script
    shader_script = Path('compile_shaders.py')
    if shader_script.exists():
        result = subprocess.run([sys.executable, str(shader_script)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ Shaders compiled successfully")
        else:
            print("  ✗ Shader compilation failed:")
            print(result.stderr)
            return False
    else:
        print("  ⚠ Shader compilation script not found")
        print("  Creating dummy shader headers for now...")
        
        # Create dummy headers
        include_dir = Path('cpp/include/shaders')
        include_dir.mkdir(parents=True, exist_ok=True)
        
        for shader_name in ['height_field.vert', 'height_field.frag']:
            header_path = include_dir / f"{shader_name}.h"
            write_file_if_not_exists(str(header_path), f"""// Dummy shader header for {shader_name}
// Replace with actual compiled SPIR-V data
static const uint32_t {shader_name.replace('.', '_')}_spirv[] = {{
    0x07230203, 0x00010000, 0x00080001, 0x00000001
}};
static const size_t {shader_name.replace('.', '_')}_spirv_size = sizeof({shader_name.replace('.', '_')}_spirv);
""")
    
    return True


def build_module():
    """Build the Python module"""
    print("\nBuilding Python module...")
    
    # Clean previous builds
    for dir_name in ['build', '_skbuild', 'dist']:
        if os.path.exists(dir_name):
            print(f"  Cleaning {dir_name}/")
            shutil.rmtree(dir_name)
    
    # Install build dependencies
    print("\n  Installing build dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
                   'pip', 'setuptools', 'wheel', 'scikit-build-core', 'pybind11', 'ninja'],
                  check=True)
    
    # Build the module
    print("\n  Building module...")
    env = os.environ.copy()
    
    # Set build type
    env['CMAKE_BUILD_TYPE'] = 'Release'
    
    # Build
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '-v'],
                          env=env)
    
    if result.returncode == 0:
        print("\n  ✓ Build successful!")
        return True
    else:
        print("\n  ✗ Build failed!")
        return False


def run_tests():
    """Run basic tests to verify the build"""
    print("\nRunning tests...")
    
    test_script = """
import sys
try:
    import vulkan_forge
    print(f"  ✓ Module imported successfully")
    print(f"  Version: {vulkan_forge.__version__}")
    
    # Test basic functionality
    scene = vulkan_forge.HeightFieldScene()
    print(f"  ✓ Created HeightFieldScene")
    
    import numpy as np
    heights = np.ones((10, 10), dtype=np.float32)
    scene.build(heights, zscale=1.0)
    print(f"  ✓ Built scene with {scene.n_indices} indices")
    
    renderer = vulkan_forge.Renderer(100, 100)
    print(f"  ✓ Created Renderer")
    
    print("\\n  All basic tests passed!")
    
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    sys.exit(1)
"""
    
    result = subprocess.run([sys.executable, '-c', test_script])
    return result.returncode == 0


def create_examples():
    """Create example scripts"""
    print("\nCreating example scripts...")
    
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    
    # Basic example
    write_file_if_not_exists('examples/basic_example.py', '''#!/usr/bin/env python3
"""Basic example of using vulkan-forge"""

import numpy as np
import matplotlib.pyplot as plt
from vulkan_forge import VulkanRenderer

# Create test terrain
x = np.linspace(-5, 5, 128)
y = np.linspace(-5, 5, 128)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) + 0.5 * np.cos(X) * np.sin(Y)

# Create renderer
renderer = VulkanRenderer(800, 600)

# Render
image = renderer.render_heightfield(Z.astype(np.float32), z_scale=1.0)

# Display
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.title('Vulkan Rendered Terrain')
plt.axis('off')
plt.tight_layout()
plt.show()
''')
    
    print(f"  Created: examples/basic_example.py")


def print_summary():
    """Print build summary and next steps"""
    print("\n" + "="*60)
    print("BUILD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the installation:")
    print("   python -c 'import vulkan_forge; print(vulkan_forge.__version__)'")
    print("\n2. Run the basic example:")
    print("   python examples/basic_example.py")
    print("\n3. Run the full demo:")
    print("   python complete_example.py")
    print("\nFor more information, see README.md")


def main():
    """Main build process"""
    print("Vulkan Forge Build Script")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\nBuild aborted due to missing requirements.")
        return 1
    
    # Set up project
    setup_directory_structure()
    create_project_files()
    
    # Compile shaders
    if not compile_shaders():
        print("\nWarning: Shader compilation failed. Continuing with dummy shaders...")
    
    # Build module
    if not build_module():
        print("\nBuild failed!")
        return 1
    
    # Create examples
    create_examples()
    
    # Run tests
    if run_tests():
        print_summary()
        return 0
    else:
        print("\nBuild succeeded but tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())