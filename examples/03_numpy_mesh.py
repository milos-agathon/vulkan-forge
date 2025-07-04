"""Example: Load mesh data from NumPy arrays with structured vertices."""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import vulkan_forge as vf
import time

logger = logging.getLogger(__name__)

def create_sphere_mesh(subdivisions=4):
    """Create a UV sphere mesh using NumPy."""
    # Create initial icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Base vertices
    base_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float32)
    
    # Normalize to unit sphere
    base_verts = base_verts / np.linalg.norm(base_verts, axis=1, keepdims=True)
    
    # Base triangles
    base_tris = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.uint32)
    
    # Subdivide
    vertices = base_verts
    triangles = base_tris
    
    for _ in range(subdivisions):
        vertices, triangles = subdivide_mesh(vertices, triangles)
    
    # Calculate normals (for sphere, normal = position)
    normals = vertices.copy()
    
    # Calculate texture coordinates (spherical mapping)
    texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
    texcoords[:, 0] = 0.5 + np.arctan2(vertices[:, 2], vertices[:, 0]) / (2 * np.pi)
    texcoords[:, 1] = 0.5 + np.arcsin(vertices[:, 1]) / np.pi
    
    return vertices, normals, texcoords, triangles


def subdivide_mesh(vertices, triangles):
    """Subdivide a triangular mesh."""
    edge_map = {}
    new_vertices = list(vertices)
    
    def get_edge_point(v0, v1):
        """Get or create midpoint of edge."""
        edge = tuple(sorted([v0, v1]))
        if edge not in edge_map:
            # Create new vertex at midpoint
            mid = (vertices[v0] + vertices[v1]) / 2
            mid = mid / np.linalg.norm(mid)  # Project to sphere
            edge_map[edge] = len(new_vertices)
            new_vertices.append(mid)
        return edge_map[edge]
    
    # Subdivide each triangle into 4
    new_triangles = []
    for tri in triangles:
        v0, v1, v2 = tri
        
        # Get edge midpoints
        v01 = get_edge_point(v0, v1)
        v12 = get_edge_point(v1, v2)
        v20 = get_edge_point(v2, v0)
        
        # Create 4 new triangles
        new_triangles.extend([
            [v0, v01, v20],
            [v1, v12, v01],
            [v2, v20, v12],
            [v01, v12, v20]
        ])
    
    return np.array(new_vertices, dtype=np.float32), np.array(new_triangles, dtype=np.uint32)


def main():
    """Main example function."""
    print("NumPy Mesh Example - Subdivided Sphere")
    print("=" * 50)
    
    # Initialize
    renderer = vf.Renderer(1920, 1080)
    # Try to create allocator
    try:
        allocator = vf.create_allocator() if hasattr(vf, 'create_allocator') else None
    except Exception as e:
        logger.warning(f"Failed to create allocator: {e}")
        allocator = None
    
    # Create sphere mesh
    print("Creating subdivided sphere mesh...")
    start_time = time.perf_counter()
    vertices, normals, texcoords, indices = create_sphere_mesh(subdivisions=5)
    creation_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Mesh stats:")
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Triangles: {len(indices):,}")
    print(f"  Creation time: {creation_time:.2f}ms")

    # Create mesh object for renderer
    mesh = vf.Mesh(
        vertices=vertices,
        normals=normals,
        uvs=texcoords,
        indices=indices
    )
    
    # Create material
    material = vf.Material(
        base_color=(0.8, 0.8, 0.8, 1.0),
        metallic=0.0,
        roughness=0.5
    )
    
    # Create light
    light = vf.Light(
        position=(5.0, 5.0, 5.0),
        color=(1.0, 1.0, 1.0),
        intensity=1.0
    )
    
    
    # Set up matrices
    view_matrix = vf.Matrix4x4.look_at(
        (3.0, 3.0, 3.0),  # eye - expects tuple
        (0.0, 0.0, 0.0),  # target - expects tuple  
        (0.0, 1.0, 0.0)   # up - expects tuple
    )

    projection_matrix = vf.Matrix4x4.perspective(np.radians(45), 1920/1080, 0.1, 100)
    
    # Render loop
    print("\nRendering...")
    frame_times = []
    
    for frame in range(30):  # 1 second at 30 FPS
        frame_start = time.perf_counter()
        
        # Animate rotation - create rotated vertices
        angle = frame * 0.02
        rotation = vf.Matrix4x4.rotation_y(angle)

        # Transform vertices
        rotated_vertices = np.empty_like(vertices)
        for i, vertex in enumerate(vertices):
            rotated_vertices[i] = rotation.transform_point(tuple(vertex))[:3]
         
        # Update mesh with rotated vertices
        mesh.vertices = rotated_vertices
        
        # Render
        image = renderer.render(
            meshes=[mesh],
            materials=[material],
            lights=[light],
            view_matrix=view_matrix,
            projection_matrix=projection_matrix
        )
        
        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)
        
        if frame % 30 == 0:
            avg_time = np.mean(frame_times[-30:])
            print(f"Frame {frame}: {avg_time:.2f}ms ({1000/avg_time:.0f} FPS)")
    
    # Save final frame
    vf.save_image(image, "sphere_mesh.png")
    
    print("\nDone!")

if __name__ == "__main__":
    main()