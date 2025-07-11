"""
3D and 2D plotting utilities for terrain visualization in VulkanForge.

This module provides functions for creating unified terrain visualizations
with consistent color mapping and GL axis convention support.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Any


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
