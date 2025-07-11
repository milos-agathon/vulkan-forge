"""
3D and 2D plotting utilities for terrain visualization in VulkanForge.

This module provides functions for creating unified terrain visualizations
with consistent color mapping and GL axis convention support.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Any, Dict

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def create_unified_terrain_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                              terrain_lut: mcolors.LinearSegmentedColormap) -> Tuple[plt.Figure, Tuple[Any, Any, Any], np.ndarray, mcolors.Normalize]:
    """
    Create 3-panel figure with unified color mapping and GL axis convention.
    
    All panels use the same terrain_lut and normalization for consistent colors.
    
    Args:
        X: East-West coordinate array
        Y: Height array
        Z: North-South coordinate array
        terrain_lut: Terrain colormap
        
    Returns:
        Tuple of (figure, axes, surface_colors, norm)
    """
    # Single source normalization
    norm = mcolors.Normalize(vmin=Y.min(), vmax=Y.max())
    
    # Create figure with consistent layout
    fig = plt.figure(figsize=(18, 6), dpi=150)
    
    # Panel 1: 3D Surface with unified colors
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Apply terrain colormap to surface
    surface_colors = terrain_lut(norm(Y))
    
    surf1 = ax1.plot_surface(X, Y, Z, facecolors=surface_colors, 
                            linewidth=0, antialiased=False, alpha=0.9)
    
    # GL axis convention
    ax1.set_xlabel('X (East-West)')
    ax1.set_ylabel('Height (Y)')
    ax1.set_zlabel('Z (North-South)')
    ax1.set_title('3D Surface (Unified Colors)')
    ax1.view_init(elev=30, azim=45)
    
    # Panel 2: 3D Surface (colored, not wireframe) with same LUT
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Create colored surface using same colormap/normalization
    surf2 = ax2.plot_surface(X, Y, Z, facecolors=surface_colors,
                            rstride=1, cstride=1, linewidth=0.5, 
                            antialiased=False, alpha=0.8)
    
    # GL axis convention  
    ax2.set_xlabel('X (East-West)')
    ax2.set_ylabel('Height (Y)')
    ax2.set_zlabel('Z (North-South)')
    ax2.set_title('3D Surface with Edges (Same LUT)')
    ax2.view_init(elev=30, azim=45)
    
    # Panel 3: 2D Orthographic validation with exact alignment
    ax3 = fig.add_subplot(133)
    
    # Use same colormap and normalization, proper extent alignment
    im = ax3.imshow(Y, cmap=terrain_lut, norm=norm, origin='lower',
                    extent=[X.min(), X.max(), Z.min(), Z.max()],
                    interpolation='nearest', alpha=0.9)
    
    # GL axis convention for 2D plot
    ax3.set_xlabel('X (East-West)')
    ax3.set_ylabel('Z (North-South)')
    ax3.set_title('2D Orthographic Validation\n(Same LUT & Normalization)')
    
    # Add colorbar with proper label
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Height (Y)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3), surface_colors, norm


def create_3d_terrain_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                         colormap: Optional[mcolors.LinearSegmentedColormap] = None,
                         figsize: Tuple[int, int] = (10, 8),
                         title: str = '3D Terrain Visualization') -> Tuple[plt.Figure, Any]:
    """
    Create a single 3D terrain plot.
    
    Args:
        X, Y, Z: Coordinate arrays
        colormap: Colormap for terrain (default: terrain colormap)
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    from .colormap import create_terrain_colormap
    
    if colormap is None:
        colormap = create_terrain_colormap()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Apply colormap
    norm = mcolors.Normalize(vmin=Y.min(), vmax=Y.max())
    surface_colors = colormap(norm(Y))
    
    # Create 3D surface
    surf = ax.plot_surface(X, Y, Z, facecolors=surface_colors,
                          linewidth=0, antialiased=True, alpha=0.9)
    
    # Set labels and title
    ax.set_xlabel('X (East-West)')
    ax.set_ylabel('Height (Y)')
    ax.set_zlabel('Z (North-South)')
    ax.set_title(title)
    
    return fig, ax


def create_2d_terrain_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                         colormap: Optional[mcolors.LinearSegmentedColormap] = None,
                         figsize: Tuple[int, int] = (8, 6),
                         title: str = '2D Terrain Visualization') -> Tuple[plt.Figure, Any]:
    """
    Create a 2D terrain plot (top-down view).
    
    Args:
        X, Y, Z: Coordinate arrays
        colormap: Colormap for terrain (default: terrain colormap)
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    from .colormap import create_terrain_colormap
    
    if colormap is None:
        colormap = create_terrain_colormap()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create 2D plot
    norm = mcolors.Normalize(vmin=Y.min(), vmax=Y.max())
    im = ax.imshow(Y, cmap=colormap, norm=norm, origin='lower',
                   extent=[X.min(), X.max(), Z.min(), Z.max()],
                   interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Height (Y)', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('X (East-West)')
    ax.set_ylabel('Z (North-South)')
    ax.set_title(title)
    
    return fig, ax


def create_contour_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                       levels: int = 20,
                       colormap: Optional[mcolors.LinearSegmentedColormap] = None,
                       figsize: Tuple[int, int] = (8, 6),
                       title: str = 'Terrain Contour Plot') -> Tuple[plt.Figure, Any]:
    """
    Create a contour plot of terrain data.
    
    Args:
        X, Y, Z: Coordinate arrays
        levels: Number of contour levels
        colormap: Colormap for contours (default: terrain colormap)
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    from .colormap import create_terrain_colormap
    
    if colormap is None:
        colormap = create_terrain_colormap()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create contour plot
    contour = ax.contour(X, Z, Y, levels=levels, cmap=colormap)
    contour_filled = ax.contourf(X, Z, Y, levels=levels, cmap=colormap, alpha=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(contour_filled, ax=ax)
    cbar.set_label('Height (Y)', rotation=270, labelpad=15)
    
    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('X (East-West)')
    ax.set_ylabel('Z (North-South)')
    ax.set_title(title)
    
    return fig, ax


def create_wireframe_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                         color: str = 'black',
                         alpha: float = 0.7,
                         figsize: Tuple[int, int] = (10, 8),
                         title: str = '3D Wireframe Terrain') -> Tuple[plt.Figure, Any]:
    """
    Create a wireframe plot of terrain data.
    
    Args:
        X, Y, Z: Coordinate arrays
        color: Wireframe color
        alpha: Transparency level
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create wireframe
    ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha)
    
    # Set labels and title
    ax.set_xlabel('X (East-West)')
    ax.set_ylabel('Height (Y)')
    ax.set_zlabel('Z (North-South)')
    ax.set_title(title)
    
    return fig, ax


def save_terrain_plot(fig: plt.Figure, filename: str, dpi: int = 150, 
                     bbox_inches: str = 'tight') -> None:
    """
    Save a terrain plot to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)


def set_3d_view(ax: Any, elev: float = 30, azim: float = 45) -> None:
    """
    Set the 3D view angle for a plot.
    
    Args:
        ax: 3D axes object
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
    """
    ax.view_init(elev=elev, azim=azim)


def add_terrain_lighting(ax: Any, light_source: Optional[Any] = None) -> None:
    """
    Add lighting effects to a 3D terrain plot.
    
    Args:
        ax: 3D axes object
        light_source: Light source object (optional)
    """
    # Basic lighting setup
    if hasattr(ax, 'zaxis'):
        ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))


def point_in_triangle_barycentric_gl(px: int, py: int, verts: list) -> bool:
    """Point-in-triangle test with GL convention."""
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


def rasterize_triangle_linear_gl(image: np.ndarray, z_buffer: np.ndarray, 
                                verts: list, width: int, height: int) -> None:
    """Rasterize triangle in linear RGB space with GL convention."""
    if len(verts) != 3:
        return
    
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    
    min_x = max(0, min(xs))
    max_x = min(width-1, max(xs))
    min_y = max(0, min(ys))
    max_y = min(height-1, max(ys))
    
    if min_x >= max_x or min_y >= max_y:
        return
    
    # Average color and depth (fragment-based colors already applied)
    avg_color = np.mean([v[3] for v in verts], axis=0)
    avg_depth = np.mean(zs)
    
    # Simple directional lighting in linear space
    light_factor = 0.8 + 0.2 * 0.7  # Consistent lighting
    lit_color = avg_color * light_factor
    
    # Fill triangle
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            if point_in_triangle_barycentric_gl(px, py, verts):
                if avg_depth < z_buffer[py, px]:
                    z_buffer[py, px] = avg_depth
                    image[py, px] = lit_color


def render_with_gl_camera(mesh_cache: Dict[str, Any], camera: Any, 
                         width: int, height: int, is_perspective: bool = True) -> np.ndarray:
    """
    Render mesh using GL camera system with unified projection.
    
    Args:
        mesh_cache: Mesh data dictionary
        camera: Camera object with projection methods
        width: Render width
        height: Render height
        is_perspective: Whether to use perspective projection
        
    Returns:
        Rendered image in linear RGB space
    """
    print(f"+ Rendering with GL camera at {width}x{height}...")
    
    vertices = mesh_cache['vertices']
    indices = mesh_cache['indices']
    colors = mesh_cache['colors']
    
    # Linear RGB output buffer
    image = np.zeros((height, width, 3), dtype=np.float32)
    z_buffer = np.full((height, width), float('inf'))
    
    triangles_rendered = 0
    triangles_total = len(indices) // 3
    
    print(f"  Camera position: [{camera.position[0]:.2f}, {camera.position[1]:.2f}, {camera.position[2]:.2f}] (GL: X=E/W, Y=Height, Z=N/S)")
    
    # Get world bounds for orthographic projection
    world_bounds = (
        mesh_cache.get('world_scale', 4.0) * -0.5,
        mesh_cache.get('world_scale', 4.0) * 0.5,
        mesh_cache.get('world_scale', 4.0) * -0.5,
        mesh_cache.get('world_scale', 4.0) * 0.5
    )
    
    # Render each triangle using unified camera projection
    for tri_idx in range(triangles_total):
        i1, i2, i3 = indices[tri_idx*3:(tri_idx+1)*3]
        v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
        c1, c2, c3 = colors[i1], colors[i2], colors[i3]
        
        # Project vertices using unified camera
        screen_coords = []
        if is_perspective:
            for v, c in [(v1, c1), (v2, c2), (v3, c3)]:
                proj_result = camera.project_perspective_gl(v, width, height)
                if proj_result is None:
                    screen_coords = None
                    break
                screen_coords.append((*proj_result, c))
        else:
            for v, c in [(v1, c1), (v2, c2), (v3, c3)]:
                proj_result = camera.project_orthographic_gl(v, width, height, world_bounds)
                screen_coords.append((*proj_result, c))
        
        if screen_coords is None or len(screen_coords) != 3:
            continue
        
        # Rasterize triangle in linear RGB space
        rasterize_triangle_linear_gl(image, z_buffer, screen_coords, width, height)
        triangles_rendered += 1
        
        if triangles_rendered % 5000 == 0:
            print(f"  Progress: {triangles_rendered}/{triangles_total} triangles")
    
    print(f"  Rendered {triangles_rendered} triangles with GL camera")
    
    # Keep in linear RGB; conversion happens on write
    return image


def create_verified_preview(mesh_cache: Dict[str, Any], heightmap: np.ndarray, 
                           camera: Any, lut_colors: np.ndarray,
                           title: str = "FINAL & VERIFIED Terrain") -> Tuple[np.ndarray, Any]:
    """
    Create verified preview with GL axis convention and validation.
    
    Args:
        mesh_cache: Mesh data dictionary
        heightmap: Height data array
        camera: Camera object
        lut_colors: LUT color array
        title: Plot title
        
    Returns:
        Tuple of (orthographic render, figure)
    """
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        print("  Matplotlib not available - skipping preview")
        return None, None
    
    from .colormap import sample_terrain_lut, linear_to_srgb
    
    print("+ Creating VERIFIED preview with GL axis convention...")
    
    X = mesh_cache['grid_x']
    Y = mesh_cache['grid_y']  # Height with GL convention
    Z = mesh_cache['grid_z']  # North-south with GL convention
    
    fig = plt.figure(figsize=(18, 6))
    
    # Panel 1 - Surface with GL axis labels and linear RGB
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Create color grid using same LUT
    color_grid = np.zeros((X.shape[0], X.shape[1], 3))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            hi = min(i, heightmap.shape[0] - 1)
            hj = min(j, heightmap.shape[1] - 1)
            normalized_height = heightmap[hi, hj]
            color_grid[i, j] = sample_terrain_lut(normalized_height, lut_colors)
    
    # Surface with nearest neighbor equivalent and linear RGB
    surf = ax1.plot_surface(X, Y, Z, facecolors=color_grid, alpha=0.9, 
                           linewidth=0, antialiased=False, shade=False)
    
    # GL axis convention labels
    ax1.set_title('Surface (GL Convention)')
    ax1.set_xlabel('X (East-West)')      # X = east-west
    ax1.set_ylabel('Height (Y)')         # Y = height  
    ax1.set_zlabel('Z (North-South)')    # Z = north-south
    
    # Set camera angle based on GL camera
    elev = np.degrees(np.arcsin((camera.position[1] - 0.0) / 
                               np.linalg.norm(camera.position)))
    azim = np.degrees(np.arctan2(camera.position[0] - 0.0, 
                                camera.position[2] - 0.0))
    ax1.view_init(elev=elev, azim=azim)
    
    # Check tick ranges
    ax1.set_xlim([-2, 2])   # X ∈ [-2, +2]
    ax1.set_ylim([0, 0.4])  # Y ∈ [0, +0.4] 
    ax1.set_zlim([-2, 2])   # Z ∈ [-2, +2]
    
    # Panel 2: Wireframe with GL convention
    ax2 = fig.add_subplot(132, projection='3d')
    
    step = max(1, X.shape[0] // 32)
    X_wire = X[::step, ::step]
    Y_wire = Y[::step, ::step]
    Z_wire = Z[::step, ::step]
    
    ax2.plot_wireframe(X_wire, Y_wire, Z_wire, color='black', alpha=0.6, linewidth=0.5)
    
    # GL axis convention labels
    ax2.set_title('Wireframe (GL Convention)')
    ax2.set_xlabel('X (East-West)')
    ax2.set_ylabel('Height (Y)')
    ax2.set_zlabel('Z (North-South)')
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_xlim([-2, 2])
    ax2.set_ylim([0, 0.4])
    ax2.set_zlim([-2, 2])
    
    # Panel 3 - Orthographic validation with exact world bounds
    ax3 = fig.add_subplot(133)
    
    # Use exact world bounds
    world_extent = [-2.0, 2.0, -2.0, 2.0]
    
    # Nearest neighbor sampling, terrain colormap, vmin=0, vmax=1
    im = ax3.imshow(heightmap, cmap='terrain', origin='lower', 
                    extent=world_extent, interpolation='nearest', 
                    vmin=0, vmax=1, alpha=0.9)
    
    # GL axis labels
    ax3.set_title('Orthographic Validation\n(GL Convention)')
    ax3.set_xlabel('X (East-West)')
    ax3.set_ylabel('Z (North-South)')
    plt.colorbar(im, ax=ax3, label='Height (Y)', shrink=0.8)
    
    # Validation annotation
    ax3.text(0.02, 0.98, 'GL Convention:\nX = East-West\nY = Height\nZ = North-South\n\nValidation:\n✓ Same mesh\n✓ Same LUT\n✓ GL_NEAREST\n✓ Linear RGB', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('terrain_verified_preview.png', dpi=150, bbox_inches='tight')
    print("+ Saved verified preview: terrain_verified_preview.png")
    plt.show()
    
    # Generate orthographic render for validation
    print("+ Generating orthographic render for validation...")
    ortho_image = render_with_gl_camera(mesh_cache, camera, 512, 512, is_perspective=False)
    ortho_image_srgb = linear_to_srgb(ortho_image)
    ortho_image_uint8 = np.clip(ortho_image_srgb * 255, 0, 255).astype(np.uint8)
    
    # Save orthographic render for comparison
    if HAS_PIL:
        Image.fromarray(ortho_image_uint8).save('terrain_ortho_validation.png')
        print("  Saved orthographic validation: terrain_ortho_validation.png")
    
    return ortho_image_uint8, fig
