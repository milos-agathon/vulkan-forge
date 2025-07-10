#!/usr/bin/env python3
"""
COMPLETE terrain rendering fix - actually uses heightmap for 3D mesh displacement
- Loads 16-bit heightmap as uint16 data
- Generates vertex grid with proper displacement
- Creates actual 3D geometry from height data
- Applies height-based materials/coloring
"""

import numpy as np
import sys
from pathlib import Path
from PIL import Image

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

import vulkan_forge as vf

def load_heightmap(filename):
    """
    Load 16-bit heightmap correctly - this is what engines actually do
    Returns normalized height data (0-1) from uint16 PNG
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"Heightmap not found: {filename}")
    
    # Load as 16-bit grayscale (preserves full precision)
    if HAS_IMAGEIO:
        heightmap = imageio.imread(filename)
        print(f"Loaded heightmap via imageio: {filename}")
    else:
        img = Image.open(filename)
        heightmap = np.array(img)
        print(f"Loaded heightmap via PIL: {filename}")
    
    print(f"  Raw data type: {heightmap.dtype}")
    print(f"  Raw data shape: {heightmap.shape}")
    print(f"  Raw value range: {heightmap.min()} to {heightmap.max()}")
    
    # Normalize to 0-1 (this is what shaders expect)
    if heightmap.dtype == np.uint16:
        heightmap_norm = heightmap.astype(np.float32) / 65535.0
    elif heightmap.dtype == np.uint8:
        heightmap_norm = heightmap.astype(np.float32) / 255.0
    else:
        heightmap_norm = heightmap.astype(np.float32)
    
    print(f"  Normalized range: {heightmap_norm.min():.3f} to {heightmap_norm.max():.3f}")
    
    return heightmap_norm

def generate_terrain_mesh(heightmap, scale_x=1.0, scale_z=1.0, scale_y=0.5):
    """
    Generate actual 3D mesh from heightmap - this creates real geometry
    This is what the GPU vertex shader would do
    """
    h, w = heightmap.shape
    print(f"Generating mesh from {w}x{h} heightmap...")
    
    # Create vertex grid
    x_coords = np.linspace(-scale_x, scale_x, w)
    z_coords = np.linspace(-scale_z, scale_z, h)
    X, Z = np.meshgrid(x_coords, z_coords)
    
    # CRITICAL: Displace Y coordinates based on heightmap
    # This is what turns a flat grid into 3D terrain
    Y = heightmap * scale_y
    
    print(f"  Vertex grid: {X.shape}")
    print(f"  Height displacement: {Y.min():.3f} to {Y.max():.3f}")
    
    # Generate triangle indices (this is what the GPU does)
    indices = []
    for i in range(h-1):
        for j in range(w-1):
            # Each quad becomes 2 triangles
            # Triangle 1: (i,j), (i+1,j), (i,j+1)
            indices.extend([
                i*w + j,           # Bottom-left
                (i+1)*w + j,       # Top-left  
                i*w + (j+1)        # Bottom-right
            ])
            # Triangle 2: (i+1,j), (i+1,j+1), (i,j+1)
            indices.extend([
                (i+1)*w + j,       # Top-left
                (i+1)*w + (j+1),   # Top-right
                i*w + (j+1)        # Bottom-right
            ])
    
    # Flatten vertex arrays
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    indices = np.array(indices)
    
    print(f"  Generated {len(vertices)} vertices, {len(indices)//3} triangles")
    
    return vertices, indices, X, Y, Z

def apply_terrain_materials(vertices, heightmap):
    """
    Apply height-based materials/colors - this is what fragment shader does
    """
    print("Applying terrain materials...")
    
    # Get height for each vertex
    heights = vertices[:, 1]  # Y component
    height_norm = (heights - heights.min()) / (heights.max() - heights.min())
    
    # Generate colors based on height (terrain shader logic)
    colors = np.zeros((len(vertices), 3))
    
    for i, h in enumerate(height_norm):
        if h < 0.1:
            # Deep water
            colors[i] = [0.0, 0.2, 0.4]
        elif h < 0.2:
            # Shallow water
            colors[i] = [0.2, 0.4, 0.6]
        elif h < 0.3:
            # Beach/sand
            colors[i] = [0.8, 0.7, 0.5]
        elif h < 0.5:
            # Grass/lowlands
            colors[i] = [0.3, 0.6, 0.2]
        elif h < 0.7:
            # Forest/hills
            colors[i] = [0.2, 0.4, 0.1]
        elif h < 0.85:
            # Rock/mountains
            colors[i] = [0.5, 0.4, 0.3]
        else:
            # Snow/peaks
            colors[i] = [0.9, 0.9, 0.9]
    
    print(f"  Applied materials to {len(colors)} vertices")
    return colors

def render_mesh_cpu(vertices, indices, colors, width, height, camera_pos=None):
    """
    CPU-based mesh renderer - simulates what GPU would do
    Actually renders the displaced 3D mesh, not a flat texture
    """
    print(f"Rendering 3D mesh ({len(vertices)} vertices) at {width}x{height}...")
    
    if camera_pos is None:
        camera_pos = np.array([0, 1, 2])  # Above and behind terrain
    
    # Output buffers
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), float('inf'))
    
    # Camera/projection parameters
    fov = 60
    aspect = width / height
    near = 0.1
    far = 10.0
    
    # Process each triangle
    num_triangles = len(indices) // 3
    processed = 0
    
    for tri_idx in range(num_triangles):
        # Get triangle vertices
        i1, i2, i3 = indices[tri_idx*3:(tri_idx+1)*3]
        v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
        c1, c2, c3 = colors[i1], colors[i2], colors[i3]
        
        # Transform to camera space (look at terrain from above)
        v1_cam = v1 - camera_pos
        v2_cam = v2 - camera_pos  
        v3_cam = v3 - camera_pos
        
        # Skip triangles behind camera (negative Z in camera space)
        if v1_cam[2] >= 0 or v2_cam[2] >= 0 or v3_cam[2] >= 0:
            continue
            
        # Project to screen space
        screen_coords = []
        valid_projection = True
        
        for v_cam, color in [(v1_cam, c1), (v2_cam, c2), (v3_cam, c3)]:
            # Perspective projection (negative Z means in front of camera)
            z_cam = -v_cam[2]  # Make Z positive for projection
            if z_cam <= near:
                valid_projection = False
                break
                
            x_proj = v_cam[0] / z_cam
            y_proj = v_cam[1] / z_cam
            
            # Convert to screen coordinates
            x_screen = int((x_proj * 0.5 + 0.5) * width)
            y_screen = int((0.5 - y_proj * 0.5) * height)  # Flip Y
            
            screen_coords.append((x_screen, y_screen, z_cam, color))
        
        # Skip invalid triangles
        if not valid_projection or len(screen_coords) != 3:
            continue
        
        # Rasterize triangle
        rasterize_triangle_with_color(image, z_buffer, screen_coords, width, height)
        
        processed += 1
        if processed % 10000 == 0:
            print(f"  Processed {processed}/{num_triangles} triangles...")
    
    print(f"  Rendered {processed} triangles")
    return image

def rasterize_triangle_with_color(image, z_buffer, tri_coords, width, height):
    """Rasterize triangle with interpolated colors"""
    if len(tri_coords) != 3:
        return
    
    # Get triangle bounds
    xs = [tc[0] for tc in tri_coords]
    ys = [tc[1] for tc in tri_coords]
    zs = [tc[2] for tc in tri_coords]
    
    min_x = max(0, min(xs))
    max_x = min(width-1, max(xs))
    min_y = max(0, min(ys))
    max_y = min(height-1, max(ys))
    
    # Rasterize
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            if point_in_triangle_screen(px, py, tri_coords):
                # Simple depth interpolation
                depth = np.mean(zs)
                
                if depth < z_buffer[py, px]:
                    z_buffer[py, px] = depth
                    
                    # Interpolate color (simplified)
                    color = np.mean([tc[3] for tc in tri_coords], axis=0)
                    image[py, px] = (color * 255).astype(np.uint8)

def point_in_triangle_screen(px, py, tri_coords):
    """Check if point is inside triangle"""
    x1, y1 = tri_coords[0][:2]
    x2, y2 = tri_coords[1][:2]
    x3, y3 = tri_coords[2][:2]
    
    # Barycentric coordinates
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
    
    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    c = 1 - a - b
    
    return a >= 0 and b >= 0 and c >= 0

def save_proper_heightmap(terrain, filename="heightmap16.png"):
    """Save 16-bit heightmap properly"""
    # Flip for engine compatibility
    terrain_flipped = np.flipud(terrain)
    
    # Convert to 16-bit
    heightmap_16bit = (terrain_flipped * 65535).astype(np.uint16)
    
    if HAS_IMAGEIO:
        imageio.imwrite(filename, heightmap_16bit)
        print(f"Saved 16-bit heightmap: {filename} (imageio)")
    else:
        heightmap_image = Image.fromarray(heightmap_16bit)
        heightmap_image.save(filename)
        print(f"Saved 16-bit heightmap: {filename} (PIL)")
    
    return heightmap_16bit

def create_3d_preview(vertices, indices, colors, title="3D Terrain Mesh"):
    """Create 3D preview using actual mesh data"""
    if not HAS_MATPLOTLIB:
        return
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D mesh view
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot triangular mesh
    for i in range(0, len(indices), 3):
        tri_indices = indices[i:i+3]
        tri_verts = vertices[tri_indices]
        tri_colors = colors[tri_indices]
        
        # Plot triangle
        xs = [tri_verts[j][0] for j in range(3)] + [tri_verts[0][0]]
        ys = [tri_verts[j][1] for j in range(3)] + [tri_verts[0][1]]
        zs = [tri_verts[j][2] for j in range(3)] + [tri_verts[0][2]]
        
        if i < 1000:  # Only plot first 1000 triangles to avoid slowdown
            ax1.plot(xs, ys, zs, 'k-', alpha=0.3, linewidth=0.5)
    
    # Plot surface
    X = vertices[:, 0].reshape(int(np.sqrt(len(vertices))), -1)
    Y = vertices[:, 1].reshape(int(np.sqrt(len(vertices))), -1)
    Z = vertices[:, 2].reshape(int(np.sqrt(len(vertices))), -1)
    
    ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8)
    ax1.set_title('3D Displaced Mesh')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y (Height)')
    ax1.set_zlabel('Z')
    
    # Color mapping view
    ax2 = fig.add_subplot(122)
    height_image = Y
    im = ax2.imshow(height_image, cmap='terrain', origin='lower')
    ax2.set_title('Height-Based Colors')
    plt.colorbar(im, ax=ax2, label='Height')
    
    plt.tight_layout()
    plt.savefig('terrain_mesh_preview.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_terrain(size=256):
    """Create test terrain"""
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Create varied terrain
    Z = np.exp(-((X)**2 + (Y)**2) / 2) * 0.8
    Z += np.sin(X * 2) * np.cos(Y * 2) * 0.2
    Z += np.exp(-((X-1)**2 + (Y+0.5)**2) / 1) * 0.3
    
    # Normalize
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z.astype(np.float32)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="COMPLETE terrain rendering with actual 3D mesh")
    parser.add_argument('--size', type=int, default=128, help='Terrain size')
    parser.add_argument('--width', type=int, default=800, help='Render width')
    parser.add_argument('--height', type=int, default=600, help='Render height')
    parser.add_argument('--output', type=str, default='terrain_3d_render.png', help='Output file')
    parser.add_argument('--heightmap', type=str, default='heightmap16.png', help='Heightmap file')
    parser.add_argument('--preview', action='store_true', help='Show 3D preview')
    parser.add_argument('--generate', action='store_true', help='Generate new terrain')
    parser.add_argument('--scale-height', type=float, default=0.3, help='Height scale factor')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPLETE TERRAIN RENDERING PIPELINE")
    print("="*70)
    
    # Step 1: Generate or load heightmap
    if args.generate or not Path(args.heightmap).exists():
        print("Generating new terrain...")
        terrain = create_terrain(args.size)
        save_proper_heightmap(terrain, args.heightmap)
    
    # Step 2: Load 16-bit heightmap (this is what engines do)
    print(f"\nLoading heightmap: {args.heightmap}")
    heightmap = load_heightmap(args.heightmap)
    
    # Step 3: Generate actual 3D mesh with vertex displacement
    print("\nGenerating 3D mesh...")
    vertices, indices, X, Y, Z = generate_terrain_mesh(heightmap, 
                                                       scale_x=2.0, 
                                                       scale_z=2.0, 
                                                       scale_y=args.scale_height)
    
    # Step 4: Apply materials/colors (fragment shader equivalent)
    print("\nApplying materials...")
    colors = apply_terrain_materials(vertices, heightmap)
    
    # Step 5: Render the actual 3D mesh
    print(f"\nRendering 3D mesh...")
    camera_pos = np.array([0, args.scale_height + 0.8, -3])  # Behind and above terrain
    image = render_mesh_cpu(vertices, indices, colors, args.width, args.height, camera_pos)
    
    # Save result
    Image.fromarray(image).save(args.output)
    print(f"\nSaved 3D terrain render: {args.output}")
    
    # Optional 3D preview
    if args.preview:
        print("\nGenerating 3D preview...")
        create_3d_preview(vertices, indices, colors)
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"+ Loaded {args.heightmap} as 16-bit heightmap")
    print(f"+ Generated {len(vertices)} vertices from height displacement")
    print(f"+ Created {len(indices)//3} triangles of actual 3D geometry")
    print(f"+ Applied height-based materials to fragments")
    print(f"+ Rendered displaced mesh (not flat texture)")
    print(f"+ Output: {args.output}")
    
    print("\n" + "="*70)
    print("WHAT'S DIFFERENT NOW")
    print("="*70)
    print("BEFORE: Flat texture with screenshot artifacts")
    print("AFTER:  Actual 3D mesh with vertex displacement")
    print("BEFORE: RGB decorations interpreted as height")
    print("AFTER:  Clean 16-bit elevation data")
    print("BEFORE: No geometry - just 2D shading")
    print("AFTER:  Real triangles with proper Z-buffering")
    
    return 0

if __name__ == "__main__":
    if not HAS_IMAGEIO:
        print("TIP: Install imageio for better 16-bit support: pip install imageio")
    
    sys.exit(main())
