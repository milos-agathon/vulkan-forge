# Save as test_direct_import.py
import sys
import os

# Don't run from the source directory
if 'vulkan-forge' in os.getcwd():
    print("ERROR: Don't run this from the vulkan-forge directory!")
    print("Run it from somewhere else, like: cd .. && python vulkan-forge/test_direct_import.py")
    sys.exit(1)

# Clear any vulkan_forge modules
for key in list(sys.modules.keys()):
    if 'vulkan' in key.lower():
        del sys.modules[key]

# Add ONLY the .pyd directory
sys.path = [r"C:\Users\milos\AppData\Local\r-miniconda\Lib\site-packages\lib\vulkan_forge"] + sys.path

# Now try to import
try:
    import _vulkan_forge_native as vf
    print("Successfully imported _vulkan_forge_native!")
except ImportError:
    print("_vulkan_forge_native not found, trying vulkan_forge...")
    import vulkan_forge as vf

print(f"Module: {vf}")
print(f"Module file: {vf.__file__}")
print(f"Module contents: {[x for x in dir(vf) if not x.startswith('_')]}")

# Only test if it's the real native module
if vf.__file__.endswith('.pyd'):
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("\nTesting native module...")
    scene = vf.HeightFieldScene()
    print("✓ Created HeightFieldScene")
    
    heights = np.random.rand(64, 64).astype(np.float32)
    scene.build(heights, None, 0.3)
    print("✓ Built scene")
    
    renderer = vf.Renderer(400, 400)
    print("✓ Created Renderer")
    
    img = renderer.render(scene)
    print(f"✓ Rendered image with shape: {img.shape}")
    
    plt.imshow(img)
    plt.title("Success! Native Vulkan Render")
    plt.axis('off')
    plt.show()
else:
    print("\nERROR: Still getting Python module, not native .pyd!")