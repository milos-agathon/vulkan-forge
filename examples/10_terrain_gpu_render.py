#!/usr/bin/env python3
"""
GPU-accelerated terrain rendering - WORKING VERSION
Handles the wireframe rendering properly
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

# Import the correct renderer
try:
    # Try to import from core.renderer (the real implementation)
    from vulkan_forge.core.renderer import Renderer, create_renderer
    from vulkan_forge.core import HeightFieldScene
    print("Using core renderer implementation")
except ImportError:
    try:
        # Fallback to renderer module
        from vulkan_forge.renderer import Renderer, create_renderer
        from vulkan_forge import HeightFieldScene
        print("Using renderer module")
    except ImportError:
        # Last resort - use the stub
        import vulkan_forge as vf
        Renderer = vf.Renderer
        HeightFieldScene = vf.HeightFieldScene
        print("Using stub renderer (limited functionality)")

def render_terrain_cpu_wireframe(terrain, width, height):
    """CPU-based wireframe renderer for terrain"""
    h, w = terrain.shape
    
    # Create image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate scaling
    scale_x = (width - 1) / (w - 1)
    scale_y = (height - 1) / (h - 1)
    
    # Draw horizontal lines
    for y in range(h):
        for x in range(w - 1):
            # Get heights
            h1 = terrain[y, x]
            h2 = terrain[y, x + 1]
            
            # Convert to screen coordinates
            x1 = int(x * scale_x)
            x2 = int((x + 1) * scale_x)
            y_screen = int(y * scale_y)
            
            # Height-based color
            color = int(min(255, (h1 + h2) * 0.5 * 255))
            
            # Draw line (simple)
            for xi in range(x1, min(x2 + 1, width)):
                if 0 <= y_screen < height:
                    image[y_screen, xi] = [color, color // 2, color // 4]
    
    # Draw vertical lines
    for x in range(w):
        for y in range(h - 1):
            # Get heights
            h1 = terrain[y, x]
            h2 = terrain[y + 1, x]
            
            # Convert to screen coordinates
            y1 = int(y * scale_y)
            y2 = int((y + 1) * scale_y)
            x_screen = int(x * scale_x)
            
            # Height-based color
            color = int(min(255, (h1 + h2) * 0.5 * 255))
            
            # Draw line (simple)
            for yi in range(y1, min(y2 + 1, height)):
                if 0 <= x_screen < width:
                    image[yi, x_screen] = [color, color // 2, color // 4]
    
    return image


def enhance_wireframe(image):
    """Convert wireframe render to filled terrain"""
    # Extract RGB
    rgb = image[:, :, :3]
    
    # Amplify wireframe pixels (they're dim)
    rgb = rgb.astype(np.float32) * 10
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    
    # Blur to fill gaps
    img = Image.fromarray(rgb)
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    
    # Enhance
    final = np.array(img).astype(np.float32) * 1.5
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    return final


def create_terrain(size=256):
    """Create interesting terrain"""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Composite terrain
    Z = np.sin(np.sqrt(X**2 + Y**2)) * 0.3
    Z += np.exp(-((X-1)**2 + (Y-1)**2) / 3) * 2
    Z += np.sin(X * 0.5) * np.cos(Y * 0.5) * 0.2
    
    # Normalize to optimal range
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z = Z * 0.6 + 0.2  # 0.2-0.8 range works best
    
    return Z.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU terrain rendering")
    parser.add_argument('--size', type=int, default=128, help='Terrain size')
    parser.add_argument('--width', type=int, default=800, help='Render width')
    parser.add_argument('--height', type=int, default=600, help='Render height')
    parser.add_argument('--output', type=str, default='terrain_render.png', help='Output file')
    parser.add_argument('--no-enhance', action='store_true', help='Skip enhancement')
    parser.add_argument('--cpu-fallback', action='store_true', help='Force CPU rendering')
    
    args = parser.parse_args()
    
    print(f"Creating {args.size}x{args.size} terrain...")
    terrain = create_terrain(args.size)
    
    if args.cpu_fallback:
        print(f"Using CPU wireframe renderer...")
        image = render_terrain_cpu_wireframe(terrain, args.width, args.height)
    else:
        try:
            print(f"Building scene...")
            scene = HeightFieldScene()
            scene.build(terrain)
            print(f"  {scene.n_indices} indices ({scene.n_indices // 3} triangles)")
            
            print(f"Rendering at {args.width}x{args.height}...")
            renderer = Renderer(args.width, args.height)
            image = renderer.render(scene)
            
            # Check wireframe coverage
            visible = np.sum(image[:,:,:3].max(axis=2) > 0)
            print(f"  Wireframe pixels: {visible} ({100*visible/(args.width*args.height):.1f}%)")
        except Exception as e:
            print(f"GPU rendering failed: {e}")
            print("Falling back to CPU renderer...")
            image = render_terrain_cpu_wireframe(terrain, args.width, args.height)
    
    if args.no_enhance:
        # Just amplify
        output = image[:, :, :3].astype(np.float32) * 2
        output = np.clip(output, 0, 255).astype(np.uint8)
    else:
        # Enhance to filled terrain
        output = enhance_wireframe(image)
    
    # Save
    Image.fromarray(output).save(args.output)
    print(f"Saved to {args.output}")
    
    # Also save the raw heightmap for comparison
    plt.figure(figsize=(6, 6))
    plt.imshow(terrain, cmap='terrain', interpolation='bilinear')
    plt.colorbar(label='Height')
    plt.title('Terrain Heightmap')
    plt.savefig('terrain_heightmap.png')
    print("Saved terrain_heightmap.png")
    
    # Display if available
    if HAS_MATPLOTLIB and not args.output.endswith('._noshow'):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(terrain, cmap='terrain')
        plt.title('Height Map')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(output)
        plt.title('Rendered')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())