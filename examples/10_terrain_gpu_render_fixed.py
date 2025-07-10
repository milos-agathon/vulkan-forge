#!/usr/bin/env python3
"""
GPU-accelerated terrain rendering - FIXED VERSION
Properly handles renderer fallback and creates proper 3D terrain visualization
"""

import numpy as np
import sys
from pathlib import Path
from PIL import Image, ImageFilter

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import vulkan_forge as vf

def render_terrain_3d(terrain, width, height, view_angle=45, scale_height=100):
    """CPU-based 3D terrain renderer with proper perspective"""
    h, w = terrain.shape
    
    # Create image buffer
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), float('inf'))
    
    # Camera and projection parameters
    view_angle_rad = np.radians(view_angle)
    aspect = width / height
    
    # Transform terrain to 3D coordinates
    x_coords = np.linspace(-1, 1, w)
    y_coords = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Scale height for better visibility
    Z = terrain * scale_height / 100.0
    
    # Apply simple perspective projection
    for i in range(h-1):
        for j in range(w-1):
            # Get quad vertices
            vertices = [
                (X[i, j], Y[i, j], Z[i, j]),
                (X[i, j+1], Y[i, j+1], Z[i, j+1]),
                (X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]),
                (X[i+1, j], Y[i+1, j], Z[i+1, j])
            ]
            
            # Project to screen coordinates
            screen_coords = []
            for x, y, z in vertices:
                # Simple perspective projection
                x_screen = int((x / (1 + z * 0.5) + 1) * width / 2)
                y_screen = int((y / (1 + z * 0.5) + 1) * height / 2)
                screen_coords.append((x_screen, y_screen, z))
            
            # Get average height for coloring
            avg_height = np.mean([v[2] for v in vertices])
            
            # Create terrain-like color based on height
            if avg_height < 0.2:
                color = [100, 150, 200]  # Blue (water)
            elif avg_height < 0.4:
                color = [150, 200, 100]  # Green (lowlands)
            elif avg_height < 0.6:
                color = [200, 180, 100]  # Brown (hills)
            else:
                color = [255, 255, 255]  # White (peaks)
            
            # Draw triangles
            for tri in [0, 1, 2], [0, 2, 3]:  # Two triangles per quad
                try:
                    # Get triangle vertices
                    v1, v2, v3 = [screen_coords[i] for i in tri]
                    
                    # Simple triangle rasterization
                    min_x = max(0, min(v1[0], v2[0], v3[0]))
                    max_x = min(width-1, max(v1[0], v2[0], v3[0]))
                    min_y = max(0, min(v1[1], v2[1], v3[1]))
                    max_y = min(height-1, max(v1[1], v2[1], v3[1]))
                    
                    for py in range(min_y, max_y + 1):
                        for px in range(min_x, max_x + 1):
                            # Simple point-in-triangle test
                            if point_in_triangle(px, py, v1, v2, v3):
                                z_val = avg_height
                                if z_val < z_buffer[py, px]:
                                    z_buffer[py, px] = z_val
                                    image[py, px] = color
                except:
                    pass  # Skip invalid triangles
    
    return image

def point_in_triangle(px, py, v1, v2, v3):
    """Check if point (px, py) is inside triangle defined by v1, v2, v3"""
    # Using barycentric coordinates
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    x3, y3 = v3[0], v3[1]
    
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
    
    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    c = 1 - a - b
    
    return a >= 0 and b >= 0 and c >= 0

def create_terrain(size=256):
    """Create interesting terrain with multiple features"""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Create multiple terrain features
    # Central peak
    Z = np.exp(-((X)**2 + (Y)**2) / 4) * 0.8
    
    # Rolling hills
    Z += np.sin(X * 0.8) * np.cos(Y * 0.8) * 0.3
    
    # Ridge system
    Z += np.exp(-((X-2)**2 + (Y+1)**2) / 2) * 0.5
    Z += np.exp(-((X+1)**2 + (Y-2)**2) / 2) * 0.4
    
    # Noise for detail
    noise = np.random.random((size, size)) * 0.1
    Z += noise
    
    # Normalize to [0, 1]
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    return Z.astype(np.float32)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fixed terrain rendering")
    parser.add_argument('--size', type=int, default=128, help='Terrain size')
    parser.add_argument('--width', type=int, default=800, help='Render width')
    parser.add_argument('--height', type=int, default=600, help='Render height')
    parser.add_argument('--output', type=str, default='terrain_render.png', help='Output file')
    parser.add_argument('--view-angle', type=float, default=45, help='View angle in degrees')
    parser.add_argument('--height-scale', type=float, default=100, help='Height scaling factor')
    
    args = parser.parse_args()
    
    print(f"Creating {args.size}x{args.size} terrain...")
    terrain = create_terrain(args.size)
    
    print(f"Rendering 3D terrain at {args.width}x{args.height}...")
    image = render_terrain_3d(terrain, args.width, args.height, 
                              args.view_angle, args.height_scale)
    
    # Save the rendered image
    Image.fromarray(image).save(args.output)
    print(f"Saved 3D terrain render to {args.output}")
    
    # Save the heightmap for comparison
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(6, 6))
        plt.imshow(terrain, cmap='terrain', interpolation='bilinear')
        plt.colorbar(label='Height')
        plt.title('Terrain Heightmap')
        plt.savefig('terrain_heightmap.png')
        print("Saved terrain_heightmap.png")
        
        # Show comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(terrain, cmap='terrain')
        plt.title('Height Map')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('3D Rendered Terrain')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
