#!/usr/bin/env python3
"""
Compare the old broken heightmap approach vs the new correct approach
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_terrain(size=256):
    """Create test terrain"""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Create terrain features
    Z = np.exp(-((X)**2 + (Y)**2) / 4) * 0.8
    Z += np.sin(X * 0.8) * np.cos(Y * 0.8) * 0.3
    Z += np.exp(-((X-2)**2 + (Y+1)**2) / 2) * 0.5
    
    # Normalize
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z.astype(np.float32)

def save_broken_heightmap(terrain, filename):
    """The OLD BROKEN way - saves matplotlib visualization"""
    plt.figure(figsize=(6, 6))
    plt.imshow(terrain, cmap='terrain', interpolation='bilinear')
    plt.colorbar(label='Height')
    plt.title('Terrain Heightmap')
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved BROKEN heightmap to {filename}")

def save_correct_heightmap(terrain, filename):
    """The NEW CORRECT way - saves raw grayscale data"""
    heightmap_16bit = (terrain * 65535).astype(np.uint16)
    heightmap_image = Image.fromarray(heightmap_16bit)
    heightmap_image.save(filename)
    print(f"Saved CORRECT heightmap to {filename}")

def analyze_heightmap(filename):
    """Analyze what's actually in a heightmap file"""
    img = Image.open(filename)
    data = np.array(img)
    
    print(f"\nAnalyzing {filename}:")
    print(f"  Image size: {img.size}")
    print(f"  Image mode: {img.mode}")
    print(f"  Data shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Value range: {data.min()} to {data.max()}")
    
    if len(data.shape) == 3:
        print(f"  RGB channels - NOT suitable for terrain!")
        print(f"  Contains decorative elements (title, colorbar, etc.)")
    else:
        print(f"  Grayscale - suitable for terrain data")
    
    return data

def main():
    print("Creating test terrain...")
    terrain = create_terrain(256)
    
    print("\n" + "="*50)
    print("DEMONSTRATING THE PROBLEM")
    print("="*50)
    
    # Save both versions
    save_broken_heightmap(terrain, 'heightmap_broken.png')
    save_correct_heightmap(terrain, 'heightmap_correct.png')
    
    # Analyze both
    print("\n" + "-"*30)
    broken_data = analyze_heightmap('heightmap_broken.png')
    print("-"*30)
    correct_data = analyze_heightmap('heightmap_correct.png')
    
    # Show the comparison
    plt.figure(figsize=(15, 10))
    
    # Original terrain
    plt.subplot(2, 3, 1)
    plt.imshow(terrain, cmap='terrain')
    plt.title('Original Terrain Data\n(256x256 float32)')
    plt.colorbar()
    
    # Broken heightmap
    plt.subplot(2, 3, 2)
    plt.imshow(broken_data)
    plt.title('BROKEN Heightmap\n(Contains decorations!)')
    plt.axis('off')
    
    # Correct heightmap
    plt.subplot(2, 3, 3)
    plt.imshow(correct_data, cmap='gray')
    plt.title('CORRECT Heightmap\n(Pure grayscale data)')
    plt.axis('off')
    
    # Show what happens when we try to use the broken one
    plt.subplot(2, 3, 4)
    if len(broken_data.shape) == 3:
        # Convert RGB to grayscale (what the renderer would do)
        broken_gray = np.dot(broken_data[...,:3], [0.299, 0.587, 0.114])
        plt.imshow(broken_gray, cmap='gray')
        plt.title('Broken → Grayscale\n(Decorations become terrain!)')
    plt.axis('off')
    
    # Show the correct data as terrain
    plt.subplot(2, 3, 5)
    plt.imshow(correct_data, cmap='terrain')
    plt.title('Correct → Terrain\n(Clean data)')
    plt.colorbar()
    
    # Show size comparison
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f"Original: {terrain.shape}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Broken: {broken_data.shape}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Correct: {correct_data.shape}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, "Size mismatch causes\nstretching artifacts!", fontsize=12, transform=plt.gca().transAxes, color='red')
    plt.title('Size Comparison')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('heightmap_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("X BROKEN: Saves matplotlib plot with decorations")
    print("V CORRECT: Saves raw grayscale data")
    print("\nThe broken version creates terrain from:")
    print("- Plot titles and labels")
    print("- Colorbar gradients")
    print("- Axis ticks and frames")
    print("- Wrong aspect ratios")
    print("\nThe correct version gives the renderer:")
    print("- Pure elevation data")
    print("- Proper grayscale format")
    print("- Correct dimensions")
    print("- No decorative elements")

if __name__ == "__main__":
    main()
