try:
    import vulkan_forge.core
    print('Core module available')
    print(dir(vulkan_forge.core))
except ImportError as e:
    print('Core module not available:', e)

try:
    from vulkan_forge.core.renderer import Renderer
    print('Core renderer available')
    r = Renderer(800, 600)
    print(dir(r))
except ImportError as e:
    print('Core renderer not available:', e)
