#!/usr/bin/env python3
"""
WORKING terrain rendering - complete pipeline that actually works
- Loads 16-bit heightmap correctly
- Generates real 3D mesh with vertex displacement
- Renders actual 3D geometry (not flat sprites)
- Height-based materials and proper Z-buffering
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

def save_proper_heightmap(terrain, filename="heightmap16.png"):
    """Save 16-bit heightmap (engine-compatible format)"""
    # Flip for engine compatibility (0,0 = bottom-left)
    terrain_flipped = np.flipud(terrain)
    
    # Convert to 16-bit for maximum precision
    heightmap_16bit = (terrain_flipped * 65535).astype(np.uint16)
    
    if HAS_IMAGEIO:
        imageio.imwrite(filename, heightmap_16bit)
        print(f"+ Saved 16-bit heightmap: {filename} (imageio)")
    else:
        heightmap_image = Image.fromarray(heightmap_16bit)
        heightmap_image.save(filename)
        print(f"+ Saved 16-bit heightmap: {filename} (PIL)")
    
    return heightmap_16bit

def load_heightmap(filename):
    """Load 16-bit heightmap properly (what engines actually do)"""
    if not Path(filename).exists():
        raise FileNotFoundError(f"Heightmap not found: {filename}")
    
    # Load preserving bit depth
    if HAS_IMAGEIO:
        heightmap = imageio.imread(filename)
        print(f"+ Loaded heightmap via imageio: {filename}")
    else:
        img = Image.open(filename)
        heightmap = np.array(img)
        print(f"+ Loaded heightmap via PIL: {filename}")
    
    print(f"  Data type: {heightmap.dtype}")
    print(f"  Shape: {heightmap.shape}")
    print(f"  Range: {heightmap.min()} to {heightmap.max()}")
    
    # Normalize to 0-1 (shader-friendly)
    if heightmap.dtype == np.uint16:
        heightmap_norm = heightmap.astype(np.float32) / 65535.0
    elif heightmap.dtype == np.uint8:
        heightmap_norm = heightmap.astype(np.float32) / 255.0
    else:
        heightmap_norm = heightmap.astype(np.float32)
    
    print(f"  Normalized: {heightmap_norm.min():.3f} to {heightmap_norm.max():.3f}")
    return heightmap_norm

def generate_terrain_mesh(heightmap, scale=2.0, height_scale=0.5):
    """Generate 3D mesh with vertex displacement (what GPU vertex shader does)"""
    h, w = heightmap.shape
    print(f"+ Generating mesh from {w}x{h} heightmap...")
    
    # Create vertex grid coordinates
    x_coords = np.linspace(-scale, scale, w)
    z_coords = np.linspace(-scale, scale, h)
    X, Z = np.meshgrid(x_coords, z_coords)
    
    # CRITICAL: Height displacement from heightmap
    Y = heightmap * height_scale
    
    print(f"  Vertex bounds: X[{X.min():.2f}, {X.max():.2f}], Y[{Y.min():.2f}, {Y.max():.2f}], Z[{Z.min():.2f}, {Z.max():.2f}]")
    
    # Flatten for vertex array
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Generate triangle indices
    indices = []
    for i in range(h-1):
        for j in range(w-1):
            # Each quad = 2 triangles
            base = i * w + j
            # Triangle 1: bottom-left, top-left, bottom-right
            indices.extend([base, base + w, base + 1])
            # Triangle 2: top-left, top-right, bottom-right
            indices.extend([base + w, base + w + 1, base + 1])
    
    indices = np.array(indices)
    print(f"  Generated {len(vertices)} vertices, {len(indices)//3} triangles")
    
    return vertices, indices, X, Y, Z

def apply_terrain_materials(vertices):
    """Apply height-based colors (fragment shader equivalent)"""
    print("+ Applying terrain materials...")
    
    heights = vertices[:, 1]  # Y component (height)
    height_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-10)
    
    colors = np.zeros((len(vertices), 3))
    
    # Terrain color mapping
    for i, h in enumerate(height_norm):
        if h < 0.15:
            colors[i] = [0.2, 0.4, 0.8]      # Deep blue (water)
        elif h < 0.25:
            colors[i] = [0.4, 0.6, 0.9]      # Light blue (shallow)
        elif h < 0.35:
            colors[i] = [0.9, 0.8, 0.6]      # Sand/beach
        elif h < 0.55:
            colors[i] = [0.4, 0.7, 0.3]      # Grass/plains
        elif h < 0.75:
            colors[i] = [0.3, 0.5, 0.2]      # Forest
        elif h < 0.9:
            colors[i] = [0.6, 0.5, 0.4]      # Rock/stone
        else:
            colors[i] = [0.95, 0.95, 0.95]   # Snow/peaks
    
    print(f"  Applied materials to {len(colors)} vertices")
    return colors

def render_terrain_mesh(vertices, indices, colors, width, height, camera_height=1.5):
    """Render 3D mesh with proper orthographic projection"""
    print(f"+ Rendering 3D mesh at {width}x{height}...")
    
    # Output buffers
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), float('inf'))
    
    # Camera setup (orthographic projection looking down)
    terrain_bounds = {
        'x_min': vertices[:, 0].min(),
        'x_max': vertices[:, 0].max(), 
        'z_min': vertices[:, 2].min(),
        'z_max': vertices[:, 2].max()
    }
    
    print(f"  Terrain bounds: X[{terrain_bounds['x_min']:.2f}, {terrain_bounds['x_max']:.2f}], Z[{terrain_bounds['z_min']:.2f}, {terrain_bounds['z_max']:.2f}]")
    
    triangles_rendered = 0
    triangles_total = len(indices) // 3
    
    # Render each triangle
    for tri_idx in range(triangles_total):
        i1, i2, i3 = indices[tri_idx*3:(tri_idx+1)*3]
        v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
        c1, c2, c3 = colors[i1], colors[i2], colors[i3]
        
        # Project vertices to screen (orthographic top-down view)
        screen_verts = []
        for v, c in [(v1, c1), (v2, c2), (v3, c3)]:
            # Map world coordinates to screen coordinates
            x_norm = (v[0] - terrain_bounds['x_min']) / (terrain_bounds['x_max'] - terrain_bounds['x_min'])
            z_norm = (v[2] - terrain_bounds['z_min']) / (terrain_bounds['z_max'] - terrain_bounds['z_min'])
            
            x_screen = int(x_norm * (width - 1))
            y_screen = int((1 - z_norm) * (height - 1))  # Flip Z for screen Y
            
            # Use height for depth testing
            depth = camera_height - v[1]  # Higher terrain = closer to camera
            
            screen_verts.append((x_screen, y_screen, depth, c))
        
        # Rasterize triangle
        rasterize_triangle(image, z_buffer, screen_verts, width, height)
        triangles_rendered += 1
        
        if triangles_rendered % 5000 == 0:
            print(f"  Progress: {triangles_rendered}/{triangles_total} triangles")
    
    print(f"  Rendered {triangles_rendered} triangles")
    return image

def rasterize_triangle(image, z_buffer, verts, width, height):
    """Rasterize triangle with depth testing"""
    if len(verts) != 3:
        return
    
    # Get bounding box
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    
    min_x = max(0, min(xs))
    max_x = min(width-1, max(xs))
    min_y = max(0, min(ys))
    max_y = min(height-1, max(ys))
    
    # Average color and depth
    avg_color = np.mean([v[3] for v in verts], axis=0)
    avg_depth = np.mean(zs)
    
    # Fill triangle
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            if point_in_triangle(px, py, verts):
                if avg_depth < z_buffer[py, px]:
                    z_buffer[py, px] = avg_depth
                    image[py, px] = (avg_color * 255).astype(np.uint8)

def point_in_triangle(px, py, verts):
    """Point-in-triangle test using barycentric coordinates"""
    x1, y1 = verts[0][:2]
    x2, y2 = verts[1][:2]
    x3, y3 = verts[2][:2]
    
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
    
    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    c = 1 - a - b
    
    return a >= 0 and b >= 0 and c >= 0

def create_3d_preview(X, Y, Z, title="3D Terrain Mesh"):
    """Create 3D mesh preview"""
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available - skipping 3D preview")
        return
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.9)
    ax1.set_title('3D Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Height')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=45, azim=45)
    
    # Wireframe
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_wireframe(X, Y, Z, color='black', alpha=0.7, linewidth=0.5)
    ax2.set_title('3D Wireframe')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Height')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=45, azim=45)
    
    # Top-down view
    ax3 = fig.add_subplot(133)
    im = ax3.imshow(Y, cmap='terrain', origin='lower', extent=[X.min(), X.max(), Z.min(), Z.max()])
    ax3.set_title('Top-Down (Heightmap)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    plt.colorbar(im, ax=ax3, label='Height')
    
    plt.tight_layout()
    plt.savefig('terrain_3d_preview.png', dpi=150, bbox_inches='tight')
    print("+ Saved 3D preview: terrain_3d_preview.png")
    plt.show()

def create_terrain(size=128):
    """Create realistic test terrain"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # Multi-scale terrain features
    Z = np.zeros_like(X)
    
    # Major mountain
    Z += np.exp(-((X-0.5)**2 + (Y+0.3)**2) / 2.5) * 0.8
    
    # Secondary peaks
    Z += np.exp(-((X+1.2)**2 + (Y-1.0)**2) / 1.5) * 0.5
    Z += np.exp(-((X-1.8)**2 + (Y+1.5)**2) / 1.0) * 0.4
    
    # Valley
    valley = -np.exp(-((X+0.5)**2 + (Y-0.8)**2) / 2.0) * 0.3
    Z += valley
    
    # Rolling hills
    Z += np.sin(X * 1.5) * np.cos(Y * 1.2) * 0.15
    Z += np.sin(X * 3.2) * np.cos(Y * 2.8) * 0.08
    
    # Add noise
    noise = np.random.random((size, size)) * 0.05
    Z += noise
    
    # Ensure non-negative and normalize
    Z = np.maximum(Z, 0)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    return Z.astype(np.float32)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WORKING terrain rendering pipeline")
    parser.add_argument('--size', type=int, default=128, help='Terrain size')
    parser.add_argument('--width', type=int, default=800, help='Render width')
    parser.add_argument('--height', type=int, default=600, help='Render height')
    parser.add_argument('--output', type=str, default='terrain_3d_WORKING.png', help='Output file')
    parser.add_argument('--heightmap', type=str, default='heightmap16.png', help='Heightmap file')
    parser.add_argument('--preview', action='store_true', help='Show 3D preview')
    parser.add_argument('--generate', action='store_true', help='Generate new terrain')
    parser.add_argument('--height-scale', type=float, default=0.4, help='Height scale factor')
    parser.add_argument('--terrain-scale', type=float, default=2.0, help='Terrain X/Z scale')
    
    args = parser.parse_args()
    
    print("="*60)
    print("WORKING TERRAIN RENDERING PIPELINE")
    print("="*60)
    
    # Step 1: Generate or use existing heightmap
    if args.generate or not Path(args.heightmap).exists():
        print("Step 1: Generating terrain...")
        terrain = create_terrain(args.size)
        save_proper_heightmap(terrain, args.heightmap)
    else:
        print(f"Step 1: Using existing heightmap: {args.heightmap}")
    
    # Step 2: Load 16-bit heightmap
    print("\nStep 2: Loading heightmap...")
    heightmap = load_heightmap(args.heightmap)
    
    # Step 3: Generate 3D mesh with vertex displacement
    print("\nStep 3: Generating 3D mesh...")
    vertices, indices, X, Y, Z = generate_terrain_mesh(heightmap, 
                                                       scale=args.terrain_scale,
                                                       height_scale=args.height_scale)
    
    # Step 4: Apply materials
    print("\nStep 4: Applying materials...")
    colors = apply_terrain_materials(vertices)
    
    # Step 5: Render 3D mesh
    print("\nStep 5: Rendering 3D mesh...")
    image = render_terrain_mesh(vertices, indices, colors, args.width, args.height)
    
    # Save result
    Image.fromarray(image).save(args.output)
    print(f"+ Saved final render: {args.output}")
    
    # Optional 3D preview
    if args.preview:
        print("\nStep 6: Generating 3D preview...")
        create_3d_preview(X, Y, Z)
    
    print("\n" + "="*60)
    print("SUCCESS - COMPLETE PIPELINE")
    print("="*60)
    print(f"+ 16-bit heightmap: {args.heightmap}")
    print(f"+ 3D mesh: {len(vertices)} vertices, {len(indices)//3} triangles")
    print(f"+ Height displacement: vertex shader simulation")
    print(f"+ Material mapping: fragment shader simulation")
    print(f"+ 3D rendering: actual geometry, not flat texture")
    print(f"+ Final output: {args.output}")
    
    return 0

if __name__ == "__main__":
    if not HAS_IMAGEIO:
        print("TIP: Install imageio for better 16-bit support: pip install imageio")
    
    sys.exit(main())
