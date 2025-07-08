#!/usr/bin/env python3
"""
Interactive Terrain Explorer for Vulkan-Forge

A comprehensive interactive viewer for exploring terrain data with real-time
performance monitoring, camera controls, and visual debugging features.

Requirements:
    pip install vulkan-forge rasterio numpy pygame OpenGL-accelerate pillow

Controls:
    WASD - Move camera
    Mouse - Look around
    Scroll - Zoom in/out
    F - Toggle fullscreen
    P - Toggle performance overlay
    T - Toggle tessellation wireframe
    L - Toggle LOD visualization
    ESC - Exit

Usage:
    python terrain_viewer.py path/to/heightmap.tif
    python terrain_viewer.py --synthetic --size 1024
"""

import sys
import argparse
import time
import math
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("Error: pygame required for interactive viewer")
    print("Install with: pip install pygame")
    sys.exit(1)

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import OpenGL.GL.shaders as shaders
except ImportError:
    print("Error: PyOpenGL required for 3D rendering")
    print("Install with: pip install PyOpenGL PyOpenGL-accelerate")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None

# Import vulkan-forge terrain system
from vulkan_forge.terrain import TerrainRenderer, TerrainStreamer
from vulkan_forge.terrain_config import TerrainConfig, TessellationMode, LODAlgorithm


class Camera:
    """3D camera with FPS-style controls"""
    
    def __init__(self, position: np.ndarray = None, target: np.ndarray = None):
        self.position = position if position is not None else np.array([0.0, 0.0, 100.0])
        self.target = target if target is not None else np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, 1.0])
        
        # Camera parameters
        self.yaw = 0.0      # Horizontal rotation
        self.pitch = 0.0    # Vertical rotation
        self.speed = 500.0  # Movement speed
        self.sensitivity = 0.1  # Mouse sensitivity
        self.fov = 60.0     # Field of view
        self.near = 1.0     # Near clipping plane
        self.far = 50000.0  # Far clipping plane
        
        self._update_vectors()
    
    def _update_vectors(self):
        """Update camera direction vectors from yaw and pitch"""
        # Calculate forward vector
        forward = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch))
        ])
        
        self.forward = forward / np.linalg.norm(forward)
        self.right = np.cross(self.forward, np.array([0, 0, 1]))
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        
        self.target = self.position + self.forward
    
    def process_mouse(self, dx: float, dy: float):
        """Process mouse movement"""
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        
        # Constrain pitch
        self.pitch = max(-89.0, min(89.0, self.pitch))
        
        self._update_vectors()
    
    def process_keyboard(self, keys, dt: float):
        """Process keyboard input"""
        velocity = self.speed * dt
        
        if keys[pygame.K_w]:
            self.position += self.forward * velocity
        if keys[pygame.K_s]:
            self.position -= self.forward * velocity
        if keys[pygame.K_a]:
            self.position -= self.right * velocity
        if keys[pygame.K_d]:
            self.position += self.right * velocity
        if keys[pygame.K_SPACE]:
            self.position[2] += velocity
        if keys[pygame.K_LSHIFT]:
            self.position[2] -= velocity
        
        self._update_vectors()
    
    def process_scroll(self, dy: float):
        """Process mouse scroll for zoom"""
        self.fov -= dy * 2.0
        self.fov = max(10.0, min(120.0, self.fov))
    
    def get_view_matrix(self) -> np.ndarray:
        """Get view matrix for OpenGL"""
        # Create look-at matrix
        f = self.forward
        r = self.right
        u = self.up
        
        view = np.array([
            [r[0], u[0], -f[0], 0],
            [r[1], u[1], -f[1], 0],
            [r[2], u[2], -f[2], 0],
            [-np.dot(r, self.position), -np.dot(u, self.position), np.dot(f, self.position), 1]
        ], dtype=np.float32)
        
        return view
    
    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get projection matrix for OpenGL"""
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        
        return proj


class PerformanceOverlay:
    """Performance monitoring overlay"""
    
    def __init__(self, font_size: int = 24):
        pygame.font.init()
        self.font = pygame.font.Font(None, font_size)
        self.small_font = pygame.font.Font(None, font_size - 6)
        
        self.fps_history = []
        self.frame_time_history = []
        self.max_history = 100
        
        self.visible = True
        self.last_update = time.perf_counter()
        
    def update(self, renderer: TerrainRenderer, camera: Camera):
        """Update performance metrics"""
        current_time = time.perf_counter()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
            self.frame_time_history.append(dt * 1000)
            
            # Trim history
            if len(self.fps_history) > self.max_history:
                self.fps_history.pop(0)
                self.frame_time_history.pop(0)
    
    def render(self, screen: pygame.Surface, renderer: TerrainRenderer, camera: Camera):
        """Render performance overlay"""
        if not self.visible:
            return
        
        width, height = screen.get_size()
        
        # Calculate metrics
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            min_fps = min(self.fps_history)
            max_fps = max(self.fps_history)
            avg_frame_time = sum(self.frame_time_history) / len(self.frame_time_history)
        else:
            avg_fps = min_fps = max_fps = avg_frame_time = 0
        
        # Performance text
        perf_lines = [
            f"FPS: {avg_fps:.1f} (min: {min_fps:.1f}, max: {max_fps:.1f})",
            f"Frame Time: {avg_frame_time:.2f}ms",
            f"Triangles: {renderer.frame_stats.get('triangles_rendered', 0):,}",
            f"Tiles: {renderer.frame_stats.get('tiles_rendered', 0)} visible, "
            f"{renderer.frame_stats.get('culled_tiles', 0)} culled",
        ]
        
        # Camera info
        pos = camera.position
        camera_lines = [
            f"Camera: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
            f"Yaw: {camera.yaw:.1f}°, Pitch: {camera.pitch:.1f}°",
            f"FOV: {camera.fov:.1f}°, Speed: {camera.speed:.0f}",
        ]
        
        # Terrain info
        terrain_lines = []
        if renderer.bounds:
            terrain_lines = [
                f"Terrain: {len(renderer.tiles)} tiles",
                f"Bounds: {renderer.bounds.max_x - renderer.bounds.min_x:.0f} x "
                f"{renderer.bounds.max_y - renderer.bounds.min_y:.0f}",
                f"Elevation: {renderer.bounds.min_elevation:.0f}m - {renderer.bounds.max_elevation:.0f}m",
            ]
        
        # Configuration info
        config = renderer.config
        config_lines = [
            f"Tessellation: {config.tessellation.mode.value} (level {config.tessellation.base_level})",
            f"LOD: {len(config.lod.distances)} levels, max distance {config.max_render_distance:.0f}m",
            f"Tile size: {config.tile_size}",
        ]
        
        # Render text sections
        y_offset = 10
        sections = [
            ("Performance", perf_lines),
            ("Camera", camera_lines),
            ("Terrain", terrain_lines),
            ("Configuration", config_lines)
        ]
        
        for section_name, lines in sections:
            if lines:
                # Section header
                header_surface = self.font.render(section_name, True, (255, 255, 0))
                screen.blit(header_surface, (10, y_offset))
                y_offset += 30
                
                # Section content
                for line in lines:
                    text_surface = self.small_font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (20, y_offset))
                    y_offset += 20
                
                y_offset += 10
        
        # FPS graph
        self._render_fps_graph(screen, width - 250, 10, 240, 100)
        
        # Controls help
        if height > 600:  # Only show if screen is tall enough
            self._render_controls_help(screen, 10, height - 200)
    
    def _render_fps_graph(self, screen: pygame.Surface, x: int, y: int, width: int, height: int):
        """Render FPS history graph"""
        if len(self.fps_history) < 2:
            return
        
        # Background
        pygame.draw.rect(screen, (0, 0, 0, 128), (x, y, width, height))
        pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height), 2)
        
        # Calculate scale
        max_fps = max(self.fps_history)
        min_fps = min(self.fps_history)
        fps_range = max_fps - min_fps
        if fps_range == 0:
            fps_range = 1
        
        # Draw grid lines
        for i in range(0, 5):
            grid_y = y + (i * height // 4)
            pygame.draw.line(screen, (64, 64, 64), (x, grid_y), (x + width, grid_y))
        
        # Draw FPS line
        points = []
        for i, fps in enumerate(self.fps_history):
            px = x + (i * width // len(self.fps_history))
            py = y + height - int(((fps - min_fps) / fps_range) * height)
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(screen, (0, 255, 0), False, points, 2)
        
        # Labels
        label = self.small_font.render(f"FPS: {max_fps:.0f}", True, (255, 255, 255))
        screen.blit(label, (x + 5, y + 5))
        
        label = self.small_font.render(f"{min_fps:.0f}", True, (255, 255, 255))
        screen.blit(label, (x + 5, y + height - 20))
    
    def _render_controls_help(self, screen: pygame.Surface, x: int, y: int):
        """Render controls help text"""
        controls = [
            "Controls:",
            "WASD - Move camera",
            "Mouse - Look around",
            "Scroll - Zoom",
            "Space/Shift - Up/Down",
            "P - Toggle overlay",
            "F - Fullscreen",
            "ESC - Exit"
        ]
        
        for i, control in enumerate(controls):
            color = (255, 255, 0) if i == 0 else (200, 200, 200)
            text = self.small_font.render(control, True, color)
            screen.blit(text, (x, y + i * 18))


class TerrainViewer:
    """Interactive terrain viewer application"""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.running = True
        self.fullscreen = False
        self.mouse_captured = True
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Vulkan-Forge Terrain Viewer")
        
        # Create display
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        # Initialize OpenGL context (mock for this example)
        # In real implementation, this would set up Vulkan or OpenGL
        self.gl_context = None
        
        # Initialize components
        self.camera = Camera()
        self.overlay = PerformanceOverlay()
        self.renderer: Optional[TerrainRenderer] = None
        self.streamer: Optional[TerrainStreamer] = None
        
        # Input state
        self.keys = pygame.key.get_pressed()
        self.mouse_pos = pygame.mouse.get_pos()
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        # Rendering options
        self.wireframe_mode = False
        self.lod_visualization = False
        self.show_tile_bounds = False
        
    def load_terrain(self, terrain_path: Optional[str] = None, synthetic_size: int = 1024):
        """Load terrain data"""
        # Create mock Vulkan context
        class MockVulkanContext:
            def create_buffer(self, data): return 1
            def destroy_buffer(self, buffer_id): pass
        
        vulkan_context = MockVulkanContext()
        
        # Create terrain configuration
        config = TerrainConfig.from_preset('balanced')
        config.performance.target_fps = 60  # Reasonable for interactive viewing
        
        # Create renderer
        self.renderer = TerrainRenderer(vulkan_context, config)
        
        if terrain_path:
            # Load from GeoTIFF
            print(f"Loading terrain from: {terrain_path}")
            success = self.renderer.load_geotiff(terrain_path)
            if not success:
                raise RuntimeError(f"Failed to load terrain from {terrain_path}")
        else:
            # Generate synthetic terrain
            print(f"Generating synthetic terrain: {synthetic_size}x{synthetic_size}")
            heightmap = self._generate_synthetic_terrain(synthetic_size)
            
            # Manually set up terrain data
            self.renderer.bounds = type('TerrainBounds', (), {
                'min_x': 0, 'max_x': synthetic_size * 10, 
                'min_y': 0, 'max_y': synthetic_size * 10,
                'min_elevation': np.min(heightmap),
                'max_elevation': np.max(heightmap)
            })()
            
            # Generate tiles
            self.renderer._generate_tiles(heightmap, np.eye(3))
        
        # Initialize camera position
        if self.renderer.bounds:
            bounds = self.renderer.bounds
            center_x = (bounds.min_x + bounds.max_x) / 2
            center_y = (bounds.min_y + bounds.max_y) / 2
            height = bounds.max_elevation + 500  # Start 500m above terrain
            
            self.camera.position = np.array([center_x, center_y, height])
            self.camera.target = np.array([center_x, center_y, bounds.max_elevation])
            self.camera._update_vectors()
        
        # Create terrain streamer
        self.streamer = TerrainStreamer(self.renderer, max_loaded_tiles=128)
        
        print(f"Terrain loaded: {len(self.renderer.tiles)} tiles")
        if self.renderer.bounds:
            bounds = self.renderer.bounds
            print(f"Bounds: ({bounds.min_x:.1f}, {bounds.min_y:.1f}) to "
                  f"({bounds.max_x:.1f}, {bounds.max_y:.1f})")
            print(f"Elevation: {bounds.min_elevation:.1f}m to {bounds.max_elevation:.1f}m")
    
    def _generate_synthetic_terrain(self, size: int) -> np.ndarray:
        """Generate synthetic terrain for demonstration"""
        # Create coordinate grids
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate terrain using multiple noise layers
        heightmap = np.zeros((size, size))
        
        # Base terrain - rolling hills
        heightmap += 100 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        
        # Add mountain ridges
        heightmap += 200 * np.exp(-(X**2 + Y**2) / 0.5)
        
        # Add detail layers
        for octave in range(1, 6):
            frequency = 2 ** octave
            amplitude = 50.0 / (2 ** octave)
            
            noise = amplitude * np.sin(frequency * np.pi * X) * np.cos(frequency * np.pi * Y)
            heightmap += noise
        
        # Add random detail
        heightmap += 10 * np.random.random((size, size))
        
        # Ensure non-negative heights
        heightmap = np.maximum(heightmap, 0)
        
        return heightmap.astype(np.float32)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_f:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_p:
                    self.overlay.visible = not self.overlay.visible
                elif event.key == pygame.K_t:
                    self.wireframe_mode = not self.wireframe_mode
                elif event.key == pygame.K_l:
                    self.lod_visualization = not self.lod_visualization
                elif event.key == pygame.K_b:
                    self.show_tile_bounds = not self.show_tile_bounds
                elif event.key == pygame.K_r:
                    self._reset_camera()
                elif event.key == pygame.K_1:
                    self._set_preset('high_performance')
                elif event.key == pygame.K_2:
                    self._set_preset('balanced')
                elif event.key == pygame.K_3:
                    self._set_preset('high_quality')
            
            elif event.type == pygame.MOUSEMOTION and self.mouse_captured:
                dx, dy = event.rel
                self.camera.process_mouse(dx, dy)
            
            elif event.type == pygame.MOUSEWHEEL:
                self.camera.process_scroll(event.y)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # Right click
                    self._toggle_mouse_capture()
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
    
    def _toggle_mouse_capture(self):
        """Toggle mouse capture"""
        self.mouse_captured = not self.mouse_captured
        pygame.mouse.set_visible(not self.mouse_captured)
        pygame.event.set_grab(self.mouse_captured)
    
    def _reset_camera(self):
        """Reset camera to default position"""
        if self.renderer and self.renderer.bounds:
            bounds = self.renderer.bounds
            center_x = (bounds.min_x + bounds.max_x) / 2
            center_y = (bounds.min_y + bounds.max_y) / 2
            height = bounds.max_elevation + 500
            
            self.camera.position = np.array([center_x, center_y, height])
            self.camera.yaw = 0
            self.camera.pitch = 0
            self.camera._update_vectors()
    
    def _set_preset(self, preset_name: str):
        """Change terrain rendering preset"""
        if self.renderer:
            self.renderer.config = TerrainConfig.from_preset(preset_name)
            print(f"Switched to {preset_name} preset")
    
    def update(self, dt: float):
        """Update game state"""
        # Update input
        self.keys = pygame.key.get_pressed()
        self.camera.process_keyboard(self.keys, dt)
        
        # Update terrain streaming
        if self.streamer:
            self.streamer.update(self.camera.position)
        
        # Update renderer
        if self.renderer:
            view_matrix = self.camera.get_view_matrix()
            aspect_ratio = self.screen.get_width() / self.screen.get_height()
            proj_matrix = self.camera.get_projection_matrix(aspect_ratio)
            
            self.renderer.update_camera(view_matrix, proj_matrix, self.camera.position)
            
            # Simulate frame rendering
            mock_command_buffer = None
            self.renderer.render_frame(mock_command_buffer)
        
        # Update overlay
        if self.renderer:
            self.overlay.update(self.renderer, self.camera)
    
    def render(self):
        """Render frame"""
        # Clear screen
        self.screen.fill((135, 206, 235))  # Sky blue
        
        # Render 3D terrain (simulated)
        self._render_terrain_2d()
        
        # Render overlay
        if self.renderer:
            self.overlay.render(self.screen, self.renderer, self.camera)
        
        pygame.display.flip()
    
    def _render_terrain_2d(self):
        """Render 2D representation of terrain for demonstration"""
        if not self.renderer or not self.renderer.bounds:
            return
        
        bounds = self.renderer.bounds
        screen_width, screen_height = self.screen.get_size()
        
        # Calculate 2D projection
        terrain_width = bounds.max_x - bounds.min_x
        terrain_height = bounds.max_y - bounds.min_y
        
        # Camera position in terrain space
        cam_x = (self.camera.position[0] - bounds.min_x) / terrain_width
        cam_y = (self.camera.position[1] - bounds.min_y) / terrain_height
        
        # Zoom factor based on camera height
        camera_height = self.camera.position[2] - bounds.min_elevation
        max_height = bounds.max_elevation - bounds.min_elevation + 1000
        zoom = max(0.1, min(2.0, (max_height - camera_height) / max_height + 0.1))
        
        # Render tiles
        for tile in self.renderer.tiles:
            if not tile.is_loaded and not self.show_tile_bounds:
                continue
            
            # Calculate tile screen position
            tile_x = (tile.bounds.min_x - bounds.min_x) / terrain_width
            tile_y = (tile.bounds.min_y - bounds.min_y) / terrain_height
            tile_w = (tile.bounds.max_x - tile.bounds.min_x) / terrain_width
            tile_h = (tile.bounds.max_y - tile.bounds.min_y) / terrain_height
            
            # Apply camera transform
            screen_x = (tile_x - cam_x) * zoom + 0.5
            screen_y = (tile_y - cam_y) * zoom + 0.5
            screen_w = tile_w * zoom
            screen_h = tile_h * zoom
            
            # Convert to screen coordinates
            sx = int(screen_x * screen_width)
            sy = int(screen_y * screen_height)
            sw = max(1, int(screen_w * screen_width))
            sh = max(1, int(screen_h * screen_height))
            
            # Skip if off-screen
            if sx + sw < 0 or sy + sh < 0 or sx > screen_width or sy > screen_height:
                continue
            
            # Choose color based on tile properties
            if tile.is_loaded:
                # Color by LOD level
                if self.lod_visualization:
                    lod_colors = [(0, 255, 0), (255, 255, 0), (255, 128, 0), (255, 0, 0)]
                    color = lod_colors[min(tile.lod_level, len(lod_colors) - 1)]
                else:
                    # Color by elevation
                    elevation_norm = (tile.bounds.min_elevation - bounds.min_elevation) / (bounds.max_elevation - bounds.min_elevation)
                    elevation_norm = max(0, min(1, elevation_norm))
                    
                    # Blue (water) to green (low) to brown (high) to white (peaks)
                    if elevation_norm < 0.3:
                        color = (int(100 * elevation_norm / 0.3), int(150 + 105 * elevation_norm / 0.3), 255)
                    elif elevation_norm < 0.7:
                        t = (elevation_norm - 0.3) / 0.4
                        color = (int(100 + 55 * t), 255, int(255 * (1 - t)))
                    else:
                        t = (elevation_norm - 0.7) / 0.3
                        color = (int(155 + 100 * t), int(255 * (1 - t) + 200 * t), int(100 * t))
                
                alpha = 200
            else:
                color = (64, 64, 64)  # Gray for unloaded
                alpha = 100
            
            # Create surface for alpha blending
            tile_surface = pygame.Surface((sw, sh))
            tile_surface.set_alpha(alpha)
            tile_surface.fill(color)
            self.screen.blit(tile_surface, (sx, sy))
            
            # Draw tile bounds
            if self.show_tile_bounds:
                border_color = (255, 255, 255) if tile.is_loaded else (128, 128, 128)
                pygame.draw.rect(self.screen, border_color, (sx, sy, sw, sh), 1)
        
        # Draw camera position
        cam_screen_x = int(0.5 * screen_width)
        cam_screen_y = int(0.5 * screen_height)
        pygame.draw.circle(self.screen, (255, 0, 0), (cam_screen_x, cam_screen_y), 5)
        
        # Draw camera direction
        forward_x = cam_screen_x + int(self.camera.forward[0] * 30)
        forward_y = cam_screen_y + int(self.camera.forward[1] * 30)
        pygame.draw.line(self.screen, (255, 0, 0), (cam_screen_x, cam_screen_y), (forward_x, forward_y), 3)
    
    def run(self):
        """Main application loop"""
        print("Starting Vulkan-Forge Terrain Viewer")
        print("Use WASD to move, mouse to look around")
        print("Press P to toggle performance overlay, F for fullscreen, ESC to exit")
        
        last_time = time.perf_counter()
        
        while self.running:
            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            self.handle_events()
            
            # Update
            self.update(dt)
            
            # Render
            self.render()
            
            # Cap frame rate
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive Terrain Viewer')
    parser.add_argument('geotiff_path', nargs='?', help='Path to GeoTIFF heightmap file')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic terrain data')
    parser.add_argument('--size', type=int, default=1024, help='Synthetic terrain size')
    parser.add_argument('--width', type=int, default=1920, help='Window width')
    parser.add_argument('--height', type=int, default=1080, help='Window height')
    parser.add_argument('--fullscreen', action='store_true', help='Start in fullscreen')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Create viewer
        viewer = TerrainViewer(args.width, args.height)
        
        if args.fullscreen:
            viewer._toggle_fullscreen()
        
        # Load terrain
        if args.synthetic or not args.geotiff_path:
            viewer.load_terrain(synthetic_size=args.size)
        else:
            terrain_path = Path(args.geotiff_path)
            if not terrain_path.exists():
                print(f"Error: GeoTIFF file not found: {terrain_path}")
                sys.exit(1)
            
            viewer.load_terrain(str(terrain_path))
        
        # Run viewer
        viewer.run()
        
    except KeyboardInterrupt:
        print("\nViewer interrupted by user")
    except Exception as e:
        logging.error(f"Viewer failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()