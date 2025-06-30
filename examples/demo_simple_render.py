# File: examples/demo_simple_render.py
"""
Simple demo of vulkan-forge: Renders a rotating lit cube and saves the final frame.

Usage:
    python demo_simple_render.py [--frames N]
"""

import argparse
import numpy as np
import sys
import os

# Check if local vulkan_forge package exists in python/ directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_package_path = os.path.join(project_root, 'python', 'vulkan_forge')

if not os.path.exists(local_package_path):
    print(f"ERROR: Local vulkan_forge package not found at {local_package_path}")
    print("Please ensure the package structure exists at python/vulkan_forge/:")
    print("  ├── __init__.py")
    print("  ├── backend.py")
    print("  ├── renderer.py")
    print("  └── matrices.py")
    sys.exit(1)

# Check for required dependencies
try:
    import vulkan
except ImportError:
    print("ERROR: 'vulkan' module not found. Please install it:")
    print("  pip install vulkan")
    print("\nAlternatively, you can run this demo with CPU-only mode by setting:")
    print("  set VULKAN_FORGE_CPU_ONLY=1")
    print("  python examples/demo_simple_render.py --frames 120")
    
    if os.environ.get('VULKAN_FORGE_CPU_ONLY') != '1':
        sys.exit(1)

# Since there's a naming conflict with the system package, we'll import our modules directly
import importlib.util

# Helper function to load a module from file
def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# First load backend and matrices as they have no dependencies
vf_matrices = load_module_from_file("vf_matrices", os.path.join(local_package_path, "matrices.py"))
vf_backend = load_module_from_file("vf_backend", os.path.join(local_package_path, "backend.py"))

# Make them available for renderer to import
sys.modules['backend'] = vf_backend
sys.modules['matrices'] = vf_matrices

# Now load renderer which depends on backend and matrices
vf_renderer = load_module_from_file("vf_renderer", os.path.join(local_package_path, "renderer.py"))

# Extract the components we need
create_renderer = vf_renderer.create_renderer
Matrix4x4 = vf_matrices.Matrix4x4
RenderTarget = vf_renderer.RenderTarget
Mesh = vf_renderer.Mesh
Material = vf_renderer.Material
Light = vf_renderer.Light

print(f"Loaded local vulkan_forge modules from: {local_package_path}")

# Optional PIL for image saving
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not found. Will save raw RGBA data.")


def create_cube_mesh():
    """Create a unit cube mesh centered at origin."""
    # Cube vertices (8 corners)
    vertices = np.array([
        # Front face
        [-0.5, -0.5,  0.5],  # 0: bottom-left
        [ 0.5, -0.5,  0.5],  # 1: bottom-right
        [ 0.5,  0.5,  0.5],  # 2: top-right
        [-0.5,  0.5,  0.5],  # 3: top-left
        # Back face
        [-0.5, -0.5, -0.5],  # 4: bottom-left
        [ 0.5, -0.5, -0.5],  # 5: bottom-right
        [ 0.5,  0.5, -0.5],  # 6: top-right
        [-0.5,  0.5, -0.5],  # 7: top-left
    ], dtype=np.float32)
    
    # Face indices (12 triangles, 2 per face)
    indices = np.array([
        # Front face
        0, 1, 2,  2, 3, 0,
        # Back face
        5, 4, 7,  7, 6, 5,
        # Left face
        4, 0, 3,  3, 7, 4,
        # Right face
        1, 5, 6,  6, 2, 1,
        # Top face
        3, 2, 6,  6, 7, 3,
        # Bottom face
        4, 5, 1,  1, 0, 4,
    ], dtype=np.uint32)
    
    # Calculate face normals (6 faces × 4 vertices each)
    face_normals = np.array([
        [ 0,  0,  1],  # Front
        [ 0,  0, -1],  # Back
        [-1,  0,  0],  # Left
        [ 1,  0,  0],  # Right
        [ 0,  1,  0],  # Top
        [ 0, -1,  0],  # Bottom
    ], dtype=np.float32)
    
    # Expand normals to match vertex count (duplicate for shared vertices)
    normals = np.array([
        face_normals[0],  # Vertex 0 (front)
        face_normals[0],  # Vertex 1 (front)
        face_normals[0],  # Vertex 2 (front)
        face_normals[0],  # Vertex 3 (front)
        face_normals[1],  # Vertex 4 (back)
        face_normals[1],  # Vertex 5 (back)
        face_normals[1],  # Vertex 6 (back)
        face_normals[1],  # Vertex 7 (back)
    ], dtype=np.float32)
    
    # UV coordinates (simple 0-1 mapping per vertex)
    uvs = np.array([
        [0, 1], [1, 1], [1, 0], [0, 0],  # Front face UVs
        [0, 1], [1, 1], [1, 0], [0, 0],  # Back face UVs
    ], dtype=np.float32)
    
    return Mesh(vertices=vertices, normals=normals, uvs=uvs, indices=indices)


def save_framebuffer(framebuffer, filename):
    """Save framebuffer to image file."""
    if HAS_PIL:
        # Convert to PIL Image and save as PNG
        image = Image.fromarray(framebuffer, mode='RGBA')
        image.save(filename)
        print(f"Saved render to {filename}")
    else:
        # Save raw RGBA data
        raw_filename = filename.replace('.png', '.rgba')
        framebuffer.tofile(raw_filename)
        height, width = framebuffer.shape[:2]
        print(f"Saved raw RGBA data to {raw_filename}")
        print(f"To convert to PNG, use: ffmpeg -f rawvideo -pixel_format rgba -s {width}x{height} -i {raw_filename} {filename}")


def main():
    """Main demo function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vulkan-Forge simple rendering demo")
    parser.add_argument('--frames', type=int, default=120, help='Number of frames to render (default: 120)')
    args = parser.parse_args()
    
    print(f"Vulkan-Forge Demo: Rendering {args.frames} frames...")
    
    # Step 1: Create renderer with automatic GPU/CPU selection
    cpu_only = os.environ.get('VULKAN_FORGE_CPU_ONLY') == '1'
    
    try:
        if cpu_only:
            print("Running in CPU-only mode (VULKAN_FORGE_CPU_ONLY=1)")
            renderer = create_renderer(prefer_gpu=False)
        else:
            renderer = create_renderer(prefer_gpu=True, enable_validation=False)
            print("Initialized renderer (GPU acceleration may be active)")
    except Exception as e:
        print(f"Warning: GPU initialization failed ({e}), using CPU fallback")
        renderer = create_renderer(prefer_gpu=False)
    
    # Step 2: Setup render target (HD resolution)
    render_target = RenderTarget(width=1280, height=720, format="RGBA8")
    renderer.set_render_target(render_target)
    
    # Step 3: Create scene geometry
    cube_mesh = create_cube_mesh()
    
    # Step 4: Define material with PBR properties
    material = Material(
        base_color=(0.7, 0.3, 0.3, 1.0),  # Reddish color
        metallic=0.2,                      # Slightly metallic
        roughness=0.6,                     # Medium roughness
        emissive=(0.0, 0.0, 0.0)          # No emission
    )
    
    # Step 5: Setup lights
    lights = [
        Light(
            position=(5.0, 5.0, 5.0),
            color=(1.0, 1.0, 1.0),
            intensity=2.0,
            light_type="point"
        ),
        Light(
            position=(-3.0, 3.0, -3.0),
            color=(0.5, 0.5, 0.8),
            intensity=1.5,
            light_type="point"
        )
    ]
    
    # Step 6: Setup camera matrices
    view_matrix = Matrix4x4.look_at(
        eye=(3.0, 2.0, 3.0),     # Camera position
        target=(0.0, 0.0, 0.0),  # Look at origin
        up=(0.0, 1.0, 0.0)       # Up vector
    )
    
    projection_matrix = Matrix4x4.perspective(
        fov=np.pi / 4,           # 45 degrees field of view
        aspect=1280.0 / 720.0,   # HD aspect ratio
        near=0.1,
        far=100.0
    )
    
    # Step 7: Render loop with rotation animation
    framebuffer = None
    
    for frame in range(args.frames):
        # Calculate rotation angle (full 360° over all frames)
        angle = (frame / args.frames) * 2 * np.pi
        
        # Create rotation matrix around Y axis
        rotation = Matrix4x4.rotation_y(angle)
        
        # Apply rotation to view matrix (rotate the camera around the cube)
        eye_x = 3.0 * np.cos(angle)
        eye_z = 3.0 * np.sin(angle)
        animated_view = Matrix4x4.look_at(
            eye=(eye_x, 2.0, eye_z),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        
        # Render frame
        framebuffer = renderer.render(
            meshes=[cube_mesh],
            materials=[material],
            lights=lights,
            view_matrix=animated_view,
            projection_matrix=projection_matrix
        )
        
        # Progress indicator
        if (frame + 1) % 10 == 0:
            progress = (frame + 1) / args.frames * 100
            print(f"Progress: {progress:.0f}%")
    
    # Step 8: Save final frame
    if framebuffer is not None:
        save_framebuffer(framebuffer, "demo_output.png")
    else:
        print("Error: No frames were rendered")
        sys.exit(1)
    
    # Step 9: Cleanup
    renderer.cleanup()
    print("Demo complete!")


if __name__ == "__main__":
    main()