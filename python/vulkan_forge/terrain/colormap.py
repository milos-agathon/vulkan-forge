"""
Terrain colormap utilities for VulkanForge.

This module provides functions for creating and managing terrain colormaps
with consistent color mappings across different visualization contexts.
"""

import numpy as np
import matplotlib.colors as mcolors
from typing import List, Optional, Tuple, Dict, Any

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def create_terrain_colormap(N: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Create single source of truth terrain colormap with proper normalization.
    
    Args:
        N: Number of colors in the colormap (default: 256)
        
    Returns:
        LinearSegmentedColormap for terrain visualization
    """
    # Define terrain color gradient: water -> sand -> grass -> rock -> snow
    colors = [
        '#1e3a8a',  # Deep blue (water)
        '#3b82f6',  # Light blue (shallow water)  
        '#fbbf24',  # Sand/beach
        '#22c55e',  # Grass/plains
        '#166534',  # Forest/dark green
        '#78716c',  # Rock/stone gray
        '#f8fafc'   # Snow/peaks white
    ]
    
    # Create LinearSegmentedColormap
    terrain_lut = mcolors.LinearSegmentedColormap.from_list(
        'terrain_lut', colors, N=N
    )
    
    return terrain_lut


def create_elevation_colormap(N: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Create a colormap specifically for elevation data.
    
    Args:
        N: Number of colors in the colormap
        
    Returns:
        LinearSegmentedColormap for elevation visualization
    """
    colors = [
        '#000080',  # Deep blue (low elevation)
        '#0040FF',  # Blue
        '#00FFFF',  # Cyan
        '#40FF40',  # Green
        '#FFFF00',  # Yellow
        '#FF8000',  # Orange
        '#FF0000',  # Red
        '#FFFFFF'   # White (high elevation)
    ]
    
    return mcolors.LinearSegmentedColormap.from_list(
        'elevation_lut', colors, N=N
    )


def create_geological_colormap(N: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Create a colormap for geological visualization.
    
    Args:
        N: Number of colors in the colormap
        
    Returns:
        LinearSegmentedColormap for geological features
    """
    colors = [
        '#8B4513',  # Brown (sedimentary)
        '#A0522D',  # Sienna
        '#D2691E',  # Chocolate
        '#F4A460',  # Sandy brown
        '#DEB887',  # Burlywood
        '#D3D3D3',  # Light gray (metamorphic)
        '#696969',  # Dim gray (igneous)
        '#2F4F4F'   # Dark slate gray
    ]
    
    return mcolors.LinearSegmentedColormap.from_list(
        'geological_lut', colors, N=N
    )


def create_custom_colormap(colors: List[str], name: str = 'custom_terrain', N: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Create a custom terrain colormap from a list of colors.
    
    Args:
        colors: List of color strings (hex or named colors)
        name: Name for the colormap
        N: Number of colors in the colormap
        
    Returns:
        LinearSegmentedColormap with custom colors
    """
    return mcolors.LinearSegmentedColormap.from_list(name, colors, N=N)


def get_terrain_color_at_height(height: float, height_range: Tuple[float, float], 
                               colormap: Optional[mcolors.LinearSegmentedColormap] = None) -> np.ndarray:
    """
    Get the terrain color for a specific height value.
    
    Args:
        height: Height value
        height_range: Tuple of (min_height, max_height)
        colormap: Colormap to use (default: terrain colormap)
        
    Returns:
        RGBA color array
    """
    if colormap is None:
        colormap = create_terrain_colormap()
    
    min_height, max_height = height_range
    normalized_height = (height - min_height) / (max_height - min_height)
    normalized_height = np.clip(normalized_height, 0.0, 1.0)
    
    return colormap(normalized_height)


def apply_colormap_to_terrain(height_data: np.ndarray, 
                            colormap: Optional[mcolors.LinearSegmentedColormap] = None,
                            normalize: bool = True) -> np.ndarray:
    """
    Apply colormap to terrain height data.
    
    Args:
        height_data: 2D array of height values
        colormap: Colormap to apply (default: terrain colormap)
        normalize: Whether to normalize height data to [0, 1] range
        
    Returns:
        RGBA color array with shape (height, width, 4)
    """
    if colormap is None:
        colormap = create_terrain_colormap()
    
    if normalize:
        norm = mcolors.Normalize(vmin=height_data.min(), vmax=height_data.max())
        normalized_heights = norm(height_data)
    else:
        normalized_heights = height_data
    
    return colormap(normalized_heights)


def create_colormap_legend(colormap: mcolors.LinearSegmentedColormap, 
                         height_range: Tuple[float, float],
                         num_ticks: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create legend data for a terrain colormap.
    
    Args:
        colormap: Colormap to create legend for
        height_range: Tuple of (min_height, max_height)
        num_ticks: Number of tick marks in legend
        
    Returns:
        Tuple of (tick_values, tick_colors)
    """
    min_height, max_height = height_range
    tick_values = np.linspace(min_height, max_height, num_ticks)
    
    # Normalize tick values to [0, 1] range for colormap
    normalized_ticks = (tick_values - min_height) / (max_height - min_height)
    tick_colors = colormap(normalized_ticks)
    
    return tick_values, tick_colors


def blend_colormaps(colormap1: mcolors.LinearSegmentedColormap,
                   colormap2: mcolors.LinearSegmentedColormap,
                   blend_factor: float = 0.5,
                   N: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Blend two colormaps together.
    
    Args:
        colormap1: First colormap
        colormap2: Second colormap
        blend_factor: Blending factor (0.0 = only colormap1, 1.0 = only colormap2)
        N: Number of colors in resulting colormap
        
    Returns:
        Blended colormap
    """
    # Sample colors from both colormaps
    x = np.linspace(0, 1, N)
    colors1 = colormap1(x)
    colors2 = colormap2(x)
    
    # Blend the colors
    blended_colors = (1 - blend_factor) * colors1 + blend_factor * colors2
    
    # Create new colormap from blended colors
    return mcolors.ListedColormap(blended_colors, name='blended_terrain')


def get_available_colormaps() -> List[str]:
    """
    Get list of available terrain colormap names.
    
    Returns:
        List of colormap names
    """
    return [
        'terrain_lut',
        'elevation_lut', 
        'geological_lut'
    ]


def create_colormap_by_name(name: str, N: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Create a colormap by name.
    
    Args:
        name: Name of the colormap
        N: Number of colors in the colormap
        
    Returns:
        LinearSegmentedColormap
        
    Raises:
        ValueError: If colormap name is not recognized
    """
    if name == 'terrain_lut':
        return create_terrain_colormap(N)
    elif name == 'elevation_lut':
        return create_elevation_colormap(N)
    elif name == 'geological_lut':
        return create_geological_colormap(N)
    else:
        raise ValueError(f"Unknown colormap name: {name}")


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB color space."""
    srgb = np.where(linear_rgb <= 0.0031308,
                    12.92 * linear_rgb,
                    1.055 * np.power(linear_rgb, 1.0/2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def srgb_to_linear(srgb_rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB color space."""
    linear = np.where(srgb_rgb <= 0.04045,
                      srgb_rgb / 12.92,
                      np.power((srgb_rgb + 0.055) / 1.055, 2.4))
    return np.clip(linear, 0, 1)


def create_terrain_lut_png(filename: str = 'terrain_lut.png') -> np.ndarray:
    """
    Create terrain color LUT as PNG for consistent color mapping.
    
    Args:
        filename: Output filename for LUT image
        
    Returns:
        LUT colors as uint8 array
        
    Raises:
        ImportError: If PIL is not available
    """
    if not HAS_PIL:
        raise ImportError("PIL is required for PNG LUT creation. Install with: pip install Pillow")
    
    print("+ Creating terrain color LUT...")
    
    # Create 256-entry gradient
    lut_colors = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        h = i / 255.0  # Normalized height 0-1
        
        # Apply terrain gradient
        if h < 0.15:
            color = np.array([0.1, 0.3, 0.6])
        elif h < 0.25:
            t = (h - 0.15) / 0.1
            color = np.array([0.1 + t*0.3, 0.3 + t*0.3, 0.6 + t*0.2])
        elif h < 0.35:
            t = (h - 0.25) / 0.1
            color = np.array([0.8, 0.7 + t*0.1, 0.4 + t*0.2])
        elif h < 0.55:
            t = (h - 0.35) / 0.2
            color = np.array([0.2 + t*0.2, 0.6 + t*0.1, 0.2 + t*0.1])
        elif h < 0.75:
            t = (h - 0.55) / 0.2
            color = np.array([0.1 + t*0.1, 0.4 + t*0.1, 0.1 + t*0.1])
        elif h < 0.9:
            t = (h - 0.75) / 0.15
            color = np.array([0.5 + t*0.2, 0.5 + t*0.2, 0.5 + t*0.2])
        else:
            t = (h - 0.9) / 0.1
            color = np.array([0.8 + t*0.2, 0.8 + t*0.2, 0.8 + t*0.2])
        
        # Convert to sRGB uint8 for LUT storage
        lut_colors[i] = (linear_to_srgb(color) * 255).astype(np.uint8)
    
    # Save as 256×1 PNG
    lut_image = lut_colors.reshape(1, 256, 3)
    Image.fromarray(lut_image).save(filename)
    print(f"  Saved terrain LUT: {filename}")
    
    return lut_colors


def sample_terrain_lut(normalized_height: float, lut_colors: np.ndarray) -> np.ndarray:
    """
    Sample LUT with GL_NEAREST equivalent.
    
    Args:
        normalized_height: Height value in [0, 1]
        lut_colors: LUT color array
        
    Returns:
        Linear RGB color
    """
    # Clamp and scale to LUT index
    h = np.clip(normalized_height, 0, 1)
    index = int(h * 255)
    
    # Return linear RGB color
    srgb_color = lut_colors[index].astype(np.float32) / 255.0
    return srgb_to_linear(srgb_color)


def apply_fragment_colors(mesh_cache: Dict[str, Any], heightmap: np.ndarray, 
                         lut_colors: np.ndarray) -> Dict[str, Any]:
    """
    Apply colors per-fragment using heightmap re-sampling.
    
    Args:
        mesh_cache: Mesh data dictionary
        heightmap: Height data array
        lut_colors: LUT color array
        
    Returns:
        Updated mesh cache with colors
    """
    print("+ Applying fragment-based colors (GL_NEAREST sampling)...")
    
    vertices = mesh_cache['vertices']
    texcoords = mesh_cache['texcoords']
    h, w = heightmap.shape
    
    # Re-sample height texture with GL_NEAREST for each vertex
    colors = np.zeros((len(vertices), 3), dtype=np.float32)
    
    for i, (vertex, texcoord) in enumerate(zip(vertices, texcoords)):
        # Sample heightmap using texture coordinates
        u, v = texcoord
        
        # Convert UV to heightmap indices with GL_NEAREST sampling
        x_idx = int(np.clip(u * w, 0, w - 1))
        y_idx = int(np.clip(v * h, 0, h - 1))
        
        # Sample normalized height from heightmap
        normalized_height = heightmap[y_idx, x_idx]
        
        # Sample terrain LUT with GL_NEAREST
        colors[i] = sample_terrain_lut(normalized_height, lut_colors)
    
    print(f"  Applied fragment colors to {len(colors)} vertices")
    mesh_cache['colors'] = colors
    
    return mesh_cache
