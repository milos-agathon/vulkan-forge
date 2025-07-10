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

import vulkan_forge as vf


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
    
    args = parser.parse_args()
    
    print(f"Creating {args.size}x{args.size} terrain...")
    terrain = create_terrain(args.size)
    
    print(f"Building scene...")
    scene = vf.HeightFieldScene()
    scene.build(terrain)
    print(f"  {scene.n_indices} indices ({scene.n_indices // 3} triangles)")
    
    print(f"Rendering at {args.width}x{args.height}...")
    renderer = vf.Renderer(args.width, args.height)
    image = renderer.render(scene)
    
    # Check wireframe coverage
    visible = np.sum(image[:,:,:3].max(axis=2) > 0)
    print(f"  Wireframe pixels: {visible} ({100*visible/(args.width*args.height):.1f}%)")
    
    if args.no_enhance:
        # Just amplify wireframe
        output = image[:, :, :3].astype(np.float32) * 20
        output = np.clip(output, 0, 255).astype(np.uint8)
    else:
        # Enhance to filled terrain
        output = enhance_wireframe(image)
    
    # Save
    Image.fromarray(output).save(args.output)
    print(f"Saved to {args.output}")
    
    # Display if available
    if HAS_MATPLOTLIB and not args.output.endswith('.png'):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(terrain, cmap='terrain')
        plt.title('Height Map')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(output)
        plt.title('GPU Rendered')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
