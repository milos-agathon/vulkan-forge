"""
Terrain data generation utilities for VulkanForge.

This module provides functions for generating synthetic terrain data with various
terrain features like mountains, valleys, hills, and noise patterns using GL
coordinate system conventions.
"""

import numpy as np
from typing import Tuple, Optional


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
