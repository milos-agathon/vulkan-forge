"""
Command-line tools for vulkan-forge
"""

def main():
    """Entry point for vf-test command"""
    print("vulkan-forge tools - test installation successful!")
    
    try:
        from . import HeightFieldScene, Renderer
        print("✅ Core classes imported successfully")
        
        # Quick test
        scene = HeightFieldScene()
        renderer = Renderer(32, 32)
        print("✅ Objects created successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())