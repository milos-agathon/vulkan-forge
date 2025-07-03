"""Example: Load mesh data from NumPy arrays with structured vertices."""

import numpy as np
import vulkan_forge as vf
from vulkan_forge.numpy_buffer import StructuredBuffer, create_index_buffer
from vulkan_forge.vertex_input import VertexInputDescription
import time


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
    allocator = vf.create_allocator()
    
    # Create sphere mesh
    print("Creating subdivided sphere mesh...")
    start_time = time.perf_counter()
    vertices, normals, texcoords, indices = create_sphere_mesh(subdivisions=5)
    creation_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Mesh stats:")
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Triangles: {len(indices):,}")
    print(f"  Creation time: {creation_time:.2f}ms")
    
    # Create structured vertex buffer
    print("\nCreating structured vertex buffer...")
    vertex_dtype = np.dtype([
        ('position', np.float32, 3),
        ('normal', np.float32, 3),
        ('texcoord', np.float32, 2)
    ])
    
    vertex_buffer = StructuredBuffer(allocator, vertex_dtype, len(vertices))
    
    # Fill vertex data
    start_time = time.perf_counter()
    vertex_buffer['position'] = vertices
    vertex_buffer['normal'] = normals
    vertex_buffer['texcoord'] = texcoords
    fill_time = (time.perf_counter() - start_time) * 1000
    print(f"Fill time: {fill_time:.2f}ms")
    
    # Create index buffer
    index_buffer = create_index_buffer(allocator, indices)
    
    # Set up vertex input description
    vertex_desc = VertexInputDescription.from_dtype(vertex_dtype)
    
    # Create uniform buffer for transformation matrix
    uniform_dtype = np.dtype([
        ('model', np.float32, (4, 4)),
        ('view', np.float32, (4, 4)),
        ('proj', np.float32, (4, 4)),
        ('light_dir', np.float32, 4)
    ])
    
    uniforms = np.zeros(1, dtype=uniform_dtype)
    
    # Set up matrices
    uniforms['model'] = np.eye(4, dtype=np.float32)
    uniforms['view'] = look_at(
        np.array([3, 3, 3], dtype=np.float32),
        np.array([0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32)
    )
    uniforms['proj'] = perspective(45, 1920/1080, 0.1, 100)
    uniforms['light_dir'] = [0.577, 0.577, 0.577, 0]  # Normalized
    
    uniform_buffer = vf.create_uniform_buffer(allocator, uniforms)
    
    # Render loop
    print("\nRendering...")
    frame_times = []
    
    for frame in range(300):  # 10 seconds at 30 FPS
        frame_start = time.perf_counter()
        
        # Animate rotation
        angle = frame * 0.02
        uniforms['model'] = rotation_matrix_y(angle)
        uniform_buffer.sync_to_gpu()
        
        # Render
        image = renderer.render_mesh(
            vertex_buffer.buffer,
            index_buffer,
            uniform_buffer,
            vertex_desc,
            len(indices)
        )
        
        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)
        
        if frame % 30 == 0:
            avg_time = np.mean(frame_times[-30:])
            print(f"Frame {frame}: {avg_time:.2f}ms ({1000/avg_time:.0f} FPS)")
    
    # Save final frame
    vf.save_image(image, "sphere_mesh.png")
    
    print("\nDone!")


# Helper functions for matrices
def look_at(eye, center, up):
    """Create view matrix."""
    f = center - eye
    f = f / np.linalg.norm(f)
    
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    
    u = np.cross(s, f)
    
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[3, :3] = -np.dot(m[:3, :3], eye)
    
    return m


def perspective(fovy, aspect, near, far):
    """Create perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fovy) / 2)
    
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = -1
    m[3, 2] = (2 * far * near) / (near - far)
    
    return m


def rotation_matrix_y(angle):
    """Create Y-axis rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


if __name__ == "__main__":
    main()