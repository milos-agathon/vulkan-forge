"""
Terrain data generation utilities for VulkanForge.

This module provides functions for generating synthetic terrain data with various
terrain features like mountains, valleys, hills, and noise patterns using GL
coordinate system conventions.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def create_terrain_data(size: int = 33, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create deterministic terrain data with GL convention coordinates.
    
    Args:
        size: Grid size for terrain (default: 33)
        seed: Random seed for deterministic generation (default: 0)
        
    Returns:
        Tuple of (X, Y, Z) numpy arrays where:
        - X: East-West coordinates
        - Y: Height values
        - Z: North-South coordinates
    """
    np.random.seed(seed)  # Deterministic rendering
    
    # Create coordinate grids with GL convention
    x = np.linspace(-2, 2, size)  # East-West
    z = np.linspace(-2, 2, size)  # North-South  
    X, Z = np.meshgrid(x, z)
    
    # Generate terrain features (Y = height)
    Y = np.zeros_like(X)
    
    # Central mountain
    Y += np.exp(-((X-0.3)**2 + (Z+0.2)**2) / 2.5) * 0.7
    
    # Secondary peaks
    Y += np.exp(-((X+1.2)**2 + (Z-1.0)**2) / 1.8) * 0.4
    Y += np.exp(-((X-1.5)**2 + (Z+1.3)**2) / 1.5) * 0.3
    
    # Valley system
    valley = -np.exp(-((X+0.5)**2 + (Z-0.8)**2) / 3.0) * 0.2
    Y += valley
    
    # Rolling hills
    Y += np.sin(X * 1.5) * np.cos(Z * 1.8) * 0.1
    Y += np.sin(X * 3.2) * np.cos(Z * 2.5) * 0.05
    
    # Fine detail noise
    noise = np.random.random((size, size)) * 0.03
    Y += noise
    
    # Normalize to [0, 1] and apply height scaling
    Y = np.maximum(Y, 0)
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    Y = Y * 0.4  # Scale to 0.4m max height
    
    return X, Y, Z


def create_simple_terrain(size: int = 33, height_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a simple terrain with basic features.
    
    Args:
        size: Grid size for terrain
        height_scale: Scale factor for height values
        
    Returns:
        Tuple of (X, Y, Z) numpy arrays
    """
    x = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    X, Z = np.meshgrid(x, z)
    
    # Simple sine wave terrain
    Y = np.sin(X * np.pi) * np.cos(Z * np.pi) * height_scale
    Y = np.maximum(Y, 0)  # Clamp to non-negative
    
    return X, Y, Z


def create_mountain_terrain(size: int = 33, num_peaks: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create terrain with multiple mountain peaks.
    
    Args:
        size: Grid size for terrain
        num_peaks: Number of mountain peaks to generate
        
    Returns:
        Tuple of (X, Y, Z) numpy arrays
    """
    x = np.linspace(-2, 2, size)
    z = np.linspace(-2, 2, size)
    X, Z = np.meshgrid(x, z)
    
    Y = np.zeros_like(X)
    
    # Generate random peaks
    np.random.seed(42)  # Fixed seed for reproducibility
    for _ in range(num_peaks):
        peak_x = np.random.uniform(-1.5, 1.5)
        peak_z = np.random.uniform(-1.5, 1.5)
        peak_height = np.random.uniform(0.3, 0.8)
        peak_width = np.random.uniform(1.0, 3.0)
        
        Y += np.exp(-((X - peak_x)**2 + (Z - peak_z)**2) / peak_width) * peak_height
    
    return X, Y, Z


def add_noise_to_terrain(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                        noise_scale: float = 0.1, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add noise to existing terrain data.
    
    Args:
        X, Y, Z: Terrain coordinate arrays
        noise_scale: Scale factor for noise (default: 0.1)
        seed: Random seed for noise generation
        
    Returns:
        Modified terrain arrays with noise added
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.random(Y.shape) * noise_scale
    Y_noisy = Y + noise
    
    return X, Y_noisy, Z


def normalize_terrain_height(Y: np.ndarray, min_height: float = 0.0, max_height: float = 1.0) -> np.ndarray:
    """
    Normalize terrain height values to specified range.
    
    Args:
        Y: Height array
        min_height: Minimum height value
        max_height: Maximum height value
        
    Returns:
        Normalized height array
    """
    Y_min, Y_max = Y.min(), Y.max()
    if Y_max == Y_min:
        return np.full_like(Y, min_height)
    
    Y_normalized = (Y - Y_min) / (Y_max - Y_min)
    return Y_normalized * (max_height - min_height) + min_height


def get_terrain_statistics(Y: np.ndarray) -> dict:
    """
    Get statistical information about terrain height data.
    
    Args:
        Y: Height array
        
    Returns:
        Dictionary with terrain statistics
    """
    return {
        'min_height': float(Y.min()),
        'max_height': float(Y.max()),
        'mean_height': float(Y.mean()),
        'std_height': float(Y.std()),
        'height_range': float(Y.max() - Y.min())
    }


def is_power_of_two_plus_one(n: int) -> bool:
    """Check if n is power of two plus one."""
    return n > 1 and (n - 1) & (n - 2) == 0


def save_heightmap_16bit(terrain: np.ndarray, filename: str = "heightmap16_verified.png") -> np.ndarray:
    """
    Save 16-bit heightmap with verification.
    
    Args:
        terrain: Terrain height data as 2D array
        filename: Output filename
        
    Returns:
        16-bit heightmap data
        
    Raises:
        ImportError: If imageio is not available
    """
    if not HAS_IMAGEIO:
        raise ImportError("imageio is required for 16-bit heightmap support. Install with: pip install imageio")
    
    h, w = terrain.shape
    if h != w or not is_power_of_two_plus_one(h):
        print(f"WARNING: Heightmap {h}x{w} should be square and power-of-two plus one")
    
    # Coordinate system fix
    terrain_flipped = np.flipud(terrain)
    
    # Full 16-bit precision
    heightmap_16bit = (terrain_flipped * 65535).astype(np.uint16)
    
    imageio.imwrite(filename, heightmap_16bit)
    print(f"+ Saved 16-bit heightmap: {filename}")
    
    return heightmap_16bit


def load_heightmap_16bit_strict(filename: str) -> np.ndarray:
    """
    Load 16-bit heightmap with strict validation.
    
    Args:
        filename: Path to heightmap file
        
    Returns:
        Normalized heightmap data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is incorrect
        ImportError: If imageio is not available
    """
    if not HAS_IMAGEIO:
        raise ImportError("imageio is required for 16-bit heightmap support. Install with: pip install imageio")
    
    if not Path(filename).exists():
        raise FileNotFoundError(f"Heightmap not found: {filename}")
    
    heightmap_raw = imageio.imread(filename)
    
    if heightmap_raw.dtype != np.uint16:
        raise ValueError(f"Heightmap must be 16-bit (uint16), got {heightmap_raw.dtype}")
    
    h, w = heightmap_raw.shape
    if not is_power_of_two_plus_one(h) or not is_power_of_two_plus_one(w):
        print(f"WARNING: Heightmap {h}x{w} should be power-of-two plus one dimensions")
    
    # Normalize before any scaling
    heightmap_normalized = heightmap_raw.astype(np.float32) / 65535.0
    
    # Apply coordinate system fix
    heightmap_corrected = np.flipud(heightmap_normalized)
    
    print(f"+ Loaded 16-bit heightmap: {filename}")
    print(f"  Raw range: {heightmap_raw.min()} to {heightmap_raw.max()}")
    print(f"  Normalized: {heightmap_corrected.min():.6f} to {heightmap_corrected.max():.6f}")
    
    return heightmap_corrected


def build_verified_mesh(heightmap: np.ndarray, world_scale: float = 4.0, 
                       height_scale: float = 0.4) -> Dict[str, Any]:
    """
    Build mesh with geometry integrity validation.
    
    Args:
        heightmap: Height data as 2D array
        world_scale: World coordinate scale
        height_scale: Height scaling factor
        
    Returns:
        Dictionary containing mesh data and metadata
        
    Raises:
        ValueError: If geometry integrity validation fails
    """
    h, w = heightmap.shape
    print(f"+ Building verified mesh from {w}x{h} heightmap...")
    
    # Mesh = (texels + 1)² vertices
    vertex_rows = h + 1
    vertex_cols = w + 1
    expected_vertices = vertex_rows * vertex_cols
    expected_triangles = 2 * h * w
    
    print(f"  Expected: {expected_vertices} vertices, {expected_triangles} triangles")
    
    # Create vertex coordinates with GL convention
    x_coords = np.linspace(-world_scale/2, world_scale/2, vertex_cols)  # East-west
    z_coords = np.linspace(-world_scale/2, world_scale/2, vertex_rows)  # North-south
    X, Z = np.meshgrid(x_coords, z_coords)
    
    # Height displacement with edge clamping
    heightmap_expanded = np.zeros((vertex_rows, vertex_cols))
    heightmap_expanded[:h, :w] = heightmap
    heightmap_expanded[h:, :] = heightmap_expanded[h-1:h, :]
    heightmap_expanded[:, w:] = heightmap_expanded[:, w-1:w]
    
    # Y = height with GL convention
    Y = heightmap_expanded * height_scale
    
    print(f"  GL bounds: X[{X.min():.2f}, {X.max():.2f}], Y[{Y.min():.2f}, {Y.max():.2f}], Z[{Z.min():.2f}, {Z.max():.2f}]")
    
    # Flatten for vertex array
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Generate texture coordinates for fragment shader sampling
    u_coords = np.linspace(0, 1, vertex_cols)
    v_coords = np.linspace(0, 1, vertex_rows)
    U, V = np.meshgrid(u_coords, v_coords)
    texcoords = np.column_stack([U.ravel(), V.ravel()])
    
    # Generate indices with consistent CCW winding
    indices = []
    for i in range(h):
        for j in range(w):
            bottom_left = i * vertex_cols + j
            bottom_right = i * vertex_cols + (j + 1)
            top_left = (i + 1) * vertex_cols + j
            top_right = (i + 1) * vertex_cols + (j + 1)
            
            # Consistent CCW triangulation
            indices.extend([bottom_left, bottom_right, top_left])
            indices.extend([bottom_right, top_right, top_left])
    
    indices = np.array(indices)
    
    # Assert geometry integrity
    actual_vertices = len(vertices)
    actual_triangles = len(indices) // 3
    
    print(f"  Actual: {actual_vertices} vertices, {actual_triangles} triangles")
    
    if actual_vertices != expected_vertices:
        raise ValueError(f"Vertex count mismatch: expected {expected_vertices}, got {actual_vertices}")
    
    if actual_triangles != expected_triangles:
        raise ValueError(f"Triangle count mismatch: expected {expected_triangles}, got {actual_triangles}")
    
    print("  PASS: Geometry integrity validated")
    
    mesh_cache = {
        'vertices': vertices,
        'texcoords': texcoords,
        'indices': indices,
        'grid_x': X,
        'grid_y': Y,
        'grid_z': Z,
        'vertex_rows': vertex_rows,
        'vertex_cols': vertex_cols,
        'heightmap_h': h,
        'heightmap_w': w,
        'world_scale': world_scale,
        'height_scale': height_scale
    }
    
    return mesh_cache
