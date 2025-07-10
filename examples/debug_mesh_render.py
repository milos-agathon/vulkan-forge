#!/usr/bin/env python3
"""
Debug mesh renderer - simplified to identify projection issues
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_simple_terrain(size=32):
    """Create simple test terrain"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2)) * 0.5  # Simple gaussian hill
    return X, Y, Z

def simple_mesh_render(X, Y, Z, width=400, height=300):
    """Simplified mesh renderer"""
    h, w = X.shape
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), float('inf'))
    
    # Camera looking down from above
    camera_pos = np.array([0, 1, 0])
    camera_target = np.array([0, 0, 0])
    
    print(f"Terrain bounds: X=[{X.min():.2f}, {X.max():.2f}], Y=[{Y.min():.2f}, {Y.max():.2f}], Z=[{Z.min():.2f}, {Z.max():.2f}]")
    print(f"Camera at: {camera_pos}")
    
    triangles_rendered = 0
    
    # Process each quad as 2 triangles
    for i in range(h-1):
        for j in range(w-1):
            # Get quad corners
            v1 = np.array([X[i, j], Z[i, j], Y[i, j]])        # Bottom-left
            v2 = np.array([X[i, j+1], Z[i, j+1], Y[i, j+1]])  # Bottom-right
            v3 = np.array([X[i+1, j], Z[i+1, j], Y[i+1, j]])  # Top-left
            v4 = np.array([X[i+1, j+1], Z[i+1, j+1], Y[i+1, j+1]])  # Top-right
            
            # Two triangles per quad
            triangles = [(v1, v2, v3), (v2, v4, v3)]
            
            for tri in triangles:
                # Project vertices to screen
                screen_verts = []
                for v in tri:
                    # Simple orthographic projection (looking down Y axis)
                    x_screen = int((v[0] + 1) * width / 2)
                    z_screen = int((v[2] + 1) * height / 2)
                    
                    # Clip to screen bounds
                    if 0 <= x_screen < width and 0 <= z_screen < height:
                        screen_verts.append((x_screen, z_screen, v[1]))  # v[1] is height (Y)
                
                if len(screen_verts) == 3:
                    # Simple triangle fill
                    fill_triangle(image, z_buffer, screen_verts, width, height)
                    triangles_rendered += 1
    
    print(f"Rendered {triangles_rendered} triangles")
    return image

def fill_triangle(image, z_buffer, verts, width, height):
    """Simple triangle fill"""
    # Get triangle bounds
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    
    min_x = max(0, min(xs))
    max_x = min(width-1, max(xs))
    min_y = max(0, min(ys))
    max_y = min(height-1, max(ys))
    
    # Get height for coloring
    avg_height = np.mean(zs)
    
    # Height-based color
    if avg_height < 0.1:
        color = [100, 150, 200]  # Blue (low)
    elif avg_height < 0.3:
        color = [150, 200, 100]  # Green (medium)
    else:
        color = [200, 150, 100]  # Brown (high)
    
    # Fill triangle (simplified)
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            # Simple point-in-triangle test
            if point_in_triangle_simple(px, py, verts):
                depth = avg_height
                if depth < z_buffer[py, px]:
                    z_buffer[py, px] = depth
                    image[py, px] = color

def point_in_triangle_simple(px, py, verts):
    """Simple point-in-triangle test"""
    x1, y1 = verts[0][:2]
    x2, y2 = verts[1][:2]
    x3, y3 = verts[2][:2]
    
    # Barycentric coordinates
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
    
    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    c = 1 - a - b
    
    return a >= 0 and b >= 0 and c >= 0

def main():
    print("Creating simple terrain...")
    X, Y, Z = create_simple_terrain(32)
    
    print("Rendering mesh...")
    image = simple_mesh_render(X, Y, Z, 400, 300)
    
    # Save result
    Image.fromarray(image).save('debug_terrain.png')
    print("Saved debug_terrain.png")
    
    # Show comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original terrain
    ax1.imshow(Z, cmap='terrain', origin='lower')
    ax1.set_title('Original Terrain Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Rendered result
    ax2.imshow(image, origin='upper')
    ax2.set_title('Rendered 3D Mesh')
    ax2.set_xlabel('Screen X')
    ax2.set_ylabel('Screen Y')
    
    plt.tight_layout()
    plt.savefig('debug_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
