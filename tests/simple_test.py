from vulkan_forge import create_renderer, RenderTarget, Matrix4x4

# Create a renderer (will use CPU fallback if Vulkan is not available)
renderer = create_renderer(prefer_gpu=True)

# Set up a render target
target = RenderTarget(width=800, height=600)
renderer.set_render_target(target)

print(f"Renderer created: {type(renderer).__name__}")