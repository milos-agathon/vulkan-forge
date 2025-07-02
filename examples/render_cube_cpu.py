#!/usr/bin/env python3
"""Render a 3D cube using CPU renderer to see actual 3D rendering."""

import numpy as np
from vulkan_forge import (
    create_renderer, RenderTarget, Mesh, Material, Light, Matrix4x4
)
import matplotlib.pyplot as plt

def create_cube_mesh():
    """Create a simple cube mesh with proper normals."""
    # Define the 8 vertices of a cube
    base_vertices = np.array([
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [ 1,  1, -1],  # 2
        [-1,  1, -1],  # 3
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1,  1],  # 6
        [-1,  1,  1],  # 7
    ], dtype=np.float32) * 0.5
    
    # Each face needs its own vertices with correct normals
    vertices = []
    normals = []
    
    # Front face (z = 1)
    vertices.extend([base_vertices[4], base_vertices[5], base_vertices[6], base_vertices[7]])
    normals.extend([[0, 0, 1]] * 4)
    
    # Back face (z = -1)
    vertices.extend([base_vertices[1], base_vertices[0], base_vertices[3], base_vertices[2]])
    normals.extend([[0, 0, -1]] * 4)
    
    # Top face (y = 1)
    vertices.extend([base_vertices[7], base_vertices[6], base_vertices[2], base_vertices[3]])
    normals.extend([[0, 1, 0]] * 4)
    
    # Bottom face (y = -1)
    vertices.extend([base_vertices[0], base_vertices[1], base_vertices[5], base_vertices[4]])
    normals.extend([[0, -1, 0]] * 4)
    
    # Right face (x = 1)
    vertices.extend([base_vertices[5], base_vertices[1], base_vertices[2], base_vertices[6]])
    normals.extend([[1, 0, 0]] * 4)
    
    # Left face (x = -1)
    vertices.extend([base_vertices[0], base_vertices[4], base_vertices[7], base_vertices[3]])
    normals.extend([[-1, 0, 0]] * 4)
    
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    
    # Create indices for triangles (2 triangles per face)
    indices = []
    for i in range(6):  # 6 faces
        base = i * 4
        # First triangle
        indices.extend([base, base + 1, base + 2])
        # Second triangle
        indices.extend([base, base + 2, base + 3])
    
    indices = np.array(indices, dtype=np.uint32)
    
    # Simple UV coordinates
    uvs = np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (6, 1)).astype(np.float32)
    
    return Mesh(vertices=vertices, normals=normals, uvs=uvs, indices=indices)

def render_rotating_cube():
    """Render cube from multiple angles."""
    print("=== VulkanForge 3D Cube Animation ===\n")
    
    # Force CPU renderer to see actual 3D rendering
    print("Creating CPU renderer for actual 3D visualization...")
    renderer = create_renderer(prefer_gpu=False)  # Force CPU
    print(f"Using: {type(renderer).__name__}")
    
    # Set up render target
    width, height = 600, 600
    target = RenderTarget(width=width, height=height)
    renderer.set_render_target(target)
    
    # Create cube
    cube = create_cube_mesh()
    print(f"Cube: {len(cube.vertices)} vertices, {len(cube.indices)//3} triangles")
    
    # Create materials - different colors to show rotation
    materials = [
        Material(base_color=(0.8, 0.2, 0.2, 1.0), roughness=0.5),  # Red
    ]
    
    # Create lights
    lights = [
        Light(position=(5.0, 5.0, 5.0), color=(1.0, 1.0, 1.0), intensity=1.0),
        Light(position=(-3.0, 3.0, -2.0), color=(0.5, 0.5, 0.8), intensity=0.5),
    ]
    
    # Render from multiple angles
    angles = [0, 45, 90, 135]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx, angle in enumerate(angles):
        print(f"\nRendering at {angle}° rotation...")
        
        # Camera position rotates around Y axis
        rad = np.radians(angle)
        eye_x = 3.0 * np.cos(rad)
        eye_z = 3.0 * np.sin(rad)
        eye_pos = (eye_x, 2.0, eye_z)
        
        view_matrix = Matrix4x4.look_at(eye_pos, (0, 0, 0), (0, 1, 0))
        projection_matrix = Matrix4x4.perspective(
            fov=np.radians(60),
            aspect=width/height,
            near=0.1,
            far=100.0
        )
        
        # Render
        framebuffer = renderer.render(
            meshes=[cube],
            materials=materials,
            lights=lights,
            view_matrix=view_matrix,
            projection_matrix=projection_matrix
        )
        
        # Display
        axes[idx].imshow(framebuffer)
        axes[idx].set_title(f'Rotation: {angle}°')
        axes[idx].axis('off')
    
    plt.suptitle('VulkanForge: 3D Cube from Different Angles (CPU Renderer)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Cleanup
    renderer.cleanup()
    print("\nDone!")

def main():
    """Run the cube rendering example."""
    render_rotating_cube()

if __name__ == "__main__":
    main()