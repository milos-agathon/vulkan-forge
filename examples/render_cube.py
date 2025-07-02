#!/usr/bin/env python3
"""Minimal example: Render a 3D cube using VulkanForge."""

import numpy as np
from vulkan_forge import (
    create_renderer, RenderTarget, Mesh, Material, Light, Matrix4x4
)
import matplotlib.pyplot as plt

def create_cube_mesh():
    """Create a simple cube mesh."""
    # Cube vertices (8 corners)
    vertices = np.array([
        # Front face
        [-1, -1,  1],  # 0: bottom-left-front
        [ 1, -1,  1],  # 1: bottom-right-front
        [ 1,  1,  1],  # 2: top-right-front
        [-1,  1,  1],  # 3: top-left-front
        # Back face
        [-1, -1, -1],  # 4: bottom-left-back
        [ 1, -1, -1],  # 5: bottom-right-back
        [ 1,  1, -1],  # 6: top-right-back
        [-1,  1, -1],  # 7: top-left-back
    ], dtype=np.float32) * 0.5  # Scale down to unit cube
    
    # Face normals (pointing outward)
    normals = np.array([
        # Front face vertices get front normal
        [ 0,  0,  1],  # 0
        [ 0,  0,  1],  # 1
        [ 0,  0,  1],  # 2
        [ 0,  0,  1],  # 3
        # Back face vertices get back normal
        [ 0,  0, -1],  # 4
        [ 0,  0, -1],  # 5
        [ 0,  0, -1],  # 6
        [ 0,  0, -1],  # 7
    ], dtype=np.float32)
    
    # Simple UV coordinates
    uvs = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],  # Front
        [0, 0], [1, 0], [1, 1], [0, 1],  # Back
    ], dtype=np.float32)
    
    # Triangle indices (12 triangles, 2 per face)
    indices = np.array([
        # Front face
        0, 1, 2,  0, 2, 3,
        # Back face
        5, 4, 7,  5, 7, 6,
        # Top face
        3, 2, 6,  3, 6, 7,
        # Bottom face
        4, 5, 1,  4, 1, 0,
        # Right face
        1, 5, 6,  1, 6, 2,
        # Left face
        4, 0, 3,  4, 3, 7,
    ], dtype=np.uint32)
    
    # We need to expand vertices/normals/uvs to match the indices
    # This is because each vertex can have different normals on different faces
    expanded_vertices = []
    expanded_normals = []
    expanded_uvs = []
    
    # Define proper normals for each face
    face_normals = [
        [ 0,  0,  1],  # Front
        [ 0,  0, -1],  # Back
        [ 0,  1,  0],  # Top
        [ 0, -1,  0],  # Bottom
        [ 1,  0,  0],  # Right
        [-1,  0,  0],  # Left
    ]
    
    # Define vertex indices for each face
    face_indices = [
        [0, 1, 2, 3],  # Front
        [5, 4, 7, 6],  # Back
        [3, 2, 6, 7],  # Top
        [4, 5, 1, 0],  # Bottom
        [1, 5, 6, 2],  # Right
        [4, 0, 3, 7],  # Left
    ]
    
    # Build expanded arrays
    new_indices = []
    vertex_counter = 0
    
    for face_idx, face_verts in enumerate(face_indices):
        face_normal = face_normals[face_idx]
        for vert_idx in face_verts:
            expanded_vertices.append(vertices[vert_idx])
            expanded_normals.append(face_normal)
            expanded_uvs.append([0, 0])  # Simple UVs
        
        # Add two triangles for this face
        base = vertex_counter
        new_indices.extend([base, base+1, base+2])
        new_indices.extend([base, base+2, base+3])
        vertex_counter += 4
    
    # Convert to numpy arrays
    expanded_vertices = np.array(expanded_vertices, dtype=np.float32)
    expanded_normals = np.array(expanded_normals, dtype=np.float32)
    expanded_uvs = np.array(expanded_uvs, dtype=np.float32)
    new_indices = np.array(new_indices, dtype=np.uint32)
    
    return Mesh(
        vertices=expanded_vertices,
        normals=expanded_normals,
        uvs=expanded_uvs,
        indices=new_indices
    )

def main():
    print("=== VulkanForge 3D Cube Rendering Example ===\n")
    
    # Create renderer
    print("Creating renderer...")
    renderer = create_renderer(prefer_gpu=True, enable_validation=False)
    print(f"Using: {type(renderer).__name__}")
    
    # Set up render target
    width, height = 800, 600
    target = RenderTarget(width=width, height=height)
    renderer.set_render_target(target)
    print(f"Render target: {width}x{height}")
    
    # Create cube mesh
    print("\nCreating cube mesh...")
    cube = create_cube_mesh()
    print(f"Vertices: {len(cube.vertices)}, Triangles: {len(cube.indices)//3}")
    
    # Create material (red cube)
    material = Material(
        base_color=(0.8, 0.2, 0.2, 1.0),  # Red
        metallic=0.0,
        roughness=0.7
    )
    
    # Create lights
    lights = [
        Light(
            position=(5.0, 5.0, 5.0),
            color=(1.0, 1.0, 1.0),
            intensity=1.0
        ),
        Light(
            position=(-3.0, 3.0, 2.0),
            color=(0.5, 0.5, 0.8),
            intensity=0.5
        )
    ]
    
    # Set up camera matrices
    eye_pos = (3.0, 3.0, 3.0)
    target_pos = (0.0, 0.0, 0.0)
    up = (0.0, 1.0, 0.0)
    
    view_matrix = Matrix4x4.look_at(eye_pos, target_pos, up)
    projection_matrix = Matrix4x4.perspective(
        fov=np.radians(60),  # 60 degree field of view
        aspect=width/height,
        near=0.1,
        far=100.0
    )
    
    # Render the scene
    print("\nRendering...")
    framebuffer = renderer.render(
        meshes=[cube],
        materials=[material],
        lights=lights,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix
    )
    
    print(f"Rendered frame: {framebuffer.shape}, dtype={framebuffer.dtype}")
    
    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(framebuffer)
    plt.title('VulkanForge: 3D Cube Render')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save the image
    from PIL import Image
    img = Image.fromarray(framebuffer)
    img.save('cube_render.png')
    print("\nSaved render to: cube_render.png")
    
    # Cleanup
    renderer.cleanup()
    print("Done!")

if __name__ == "__main__":
    main()