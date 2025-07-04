#!/usr/bin/env python3
"""Test script for VulkanForge 3D rendering with proper cube geometry."""

import numpy as np
import matplotlib.pyplot as plt
from vulkan_forge import (
    create_renderer, RenderTarget, Mesh, Material, Light, Matrix4x4
)
import logging

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_cube_mesh():
    """Create a proper cube mesh with correct normals for each face."""
    # We need separate vertices for each face to have correct normals
    vertices = []
    normals = []
    uvs = []
    indices = []
    
    # Define the 8 corner positions of a unit cube
    positions = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # Front
    ], dtype=np.float32)
    
    # Define faces (indices into positions array)
    faces = [
        # Front face (z = 1)
        ([4, 5, 6, 7], [0, 0, 1]),
        # Back face (z = -1)  
        ([1, 0, 3, 2], [0, 0, -1]),
        # Top face (y = 1)
        ([7, 6, 2, 3], [0, 1, 0]),
        # Bottom face (y = -1)
        ([0, 1, 5, 4], [0, -1, 0]),
        # Right face (x = 1)
        ([5, 1, 2, 6], [1, 0, 0]),
        # Left face (x = -1)
        ([0, 4, 7, 3], [-1, 0, 0]),
    ]
    
    vertex_index = 0
    
    for face_indices, normal in faces:
        # Add 4 vertices for this face
        for i in face_indices:
            vertices.append(positions[i])
            normals.append(normal)
            uvs.append([0, 0])  # Simple UVs
        
        # Add 2 triangles for this face
        indices.extend([
            vertex_index, vertex_index + 1, vertex_index + 2,
            vertex_index, vertex_index + 2, vertex_index + 3
        ])
        vertex_index += 4
    
    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        normals=np.array(normals, dtype=np.float32),
        uvs=np.array(uvs, dtype=np.float32),
        indices=np.array(indices, dtype=np.uint32)
    )

def test_3d_rendering():
    """Test 3D rendering with better camera setup."""
    print("=== VulkanForge 3D Cube Rendering Test ===\n")
    
    # Create renderer
    print("1. Creating renderer...")
    renderer = create_renderer(prefer_gpu=True, enable_validation=False)
    print(f"   Renderer type: {type(renderer).__name__}")
    
    if hasattr(renderer, 'gpu_active'):
        print(f"   GPU active: {renderer.gpu_active}")
    
    # Set render target
    print("\n2. Setting render target...")
    target = RenderTarget(width=800, height=600)
    renderer.set_render_target(target)
    
    # Create cube
    print("\n3. Creating 3D cube...")
    cube = create_cube_mesh()
    print(f"   Cube has {len(cube.vertices)} vertices, {len(cube.indices)//3} triangles")
    
    # Create material
    material = Material(
        base_color=(0.8, 0.3, 0.3, 1.0),  # Red
        roughness=0.5
    )
    
    # Create lights - position them to show 3D shape better
    lights = [
        Light(position=(5.0, 5.0, 5.0), color=(1.0, 1.0, 1.0), intensity=1.0),
        Light(position=(-3.0, 2.0, 4.0), color=(0.7, 0.7, 1.0), intensity=0.5),
    ]
    
    # Render from different angles
    angles = [30, 45, 60, 120]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    pixel_counts = []
    for idx, angle in enumerate(angles):
        print(f"\n4. Rendering at {angle}° angle...")
        
        # Camera position - closer to the cube
        rad = np.radians(angle)
        distance = 5.0  # Camera distance from origin
        eye_x = distance * np.cos(rad)
        eye_z = distance * np.sin(rad)
        eye_y = 3.0  # Slight elevation
        
        view_matrix = Matrix4x4.look_at(
            eye=(eye_x, eye_y, eye_z),
            target=(0, 0, 0),
            up=(0, 1, 0)
        )
        
        projection_matrix = Matrix4x4.perspective(
            fov=np.radians(45),  # 45 degree field of view
            aspect=target.width / target.height,
            near=0.1,
            far=100.0
        )
        
        # Render
        framebuffer = renderer.render(
            meshes=[cube],
            materials=[material],
            lights=lights,
            view_matrix=view_matrix,
            projection_matrix=projection_matrix
        )
        
        # Check if we got any rendered pixels
        non_black_pixels = np.any(framebuffer[:, :, :3] > 0, axis=2)
        rendered_pixel_count = int(np.sum(non_black_pixels))
        pixel_counts.append(rendered_pixel_count)
        print(f"   Rendered {rendered_pixel_count} non-black pixels")
        
        # Display
        axes[idx].imshow(framebuffer)
        axes[idx].set_title(f'View angle: {angle}°\n({rendered_pixel_count} pixels rendered)')
        axes[idx].axis('off')

    threshold = 30000 if getattr(renderer, 'gpu_active', False) else 15000
    assert max(pixel_counts) > threshold
    
    plt.suptitle(f'VulkanForge 3D Cube Rendering ({type(renderer).__name__})', fontsize=16)
    plt.tight_layout()
    plt.savefig('cube_render_debug.png', dpi=150)
    plt.show()
    
    # Save the last rendered frame
    print("\n5. Saving output...")
    from PIL import Image
    img = Image.fromarray(framebuffer)
    img.save('test_3d_cube.png')
    print("   Saved to test_3d_cube.png")
    
    # Let's also render a single large view for debugging
    print("\n6. Rendering close-up view...")
    view_matrix = Matrix4x4.look_at(
        eye=(4, 3, 4),
        target=(0, 0, 0),
        up=(0, 1, 0)
    )
    
    framebuffer = renderer.render(
        meshes=[cube],
        materials=[material],
        lights=lights,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix
    )
    
    plt.figure(figsize=(10, 8))
    plt.imshow(framebuffer)
    plt.title('Close-up view of 3D cube')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('cube_closeup.png', dpi=150)
    plt.show()
    
    # Cleanup
    renderer.cleanup()
    print("\n✓ Test complete!")

if __name__ == "__main__":
    test_3d_rendering()