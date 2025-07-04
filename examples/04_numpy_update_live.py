"""Example: Real-time buffer updates with live data visualization."""

import numpy as np
import vulkan_forge as vf
from vulkan_forge.numpy_buffer import NumpyBuffer, MultiBuffer
import time


class WaveSimulation:
    """Simple 2D wave simulation using NumPy."""
    
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        
        # Wave state
        self.u = np.zeros((height, width), dtype=np.float32)
        self.u_prev = np.zeros((height, width), dtype=np.float32)
        self.u_next = np.zeros((height, width), dtype=np.float32)
        
        # Wave parameters
        self.c = 0.5  # Wave speed
        self.damping = 0.999
        
        # Initialize with some disturbance
        cx, cy = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        self.u[dist < 20] = 1.0
    
    def step(self, dt=0.1):
        """Advance wave simulation one time step."""
        # Wave equation with finite differences
        # u_next = 2*u - u_prev + c^2 * dt^2 * laplacian(u)
        
        # Compute Laplacian
        laplacian = (
            np.roll(self.u, 1, axis=0) +
            np.roll(self.u, -1, axis=0) +
            np.roll(self.u, 1, axis=1) +
            np.roll(self.u, -1, axis=1) -
            4 * self.u
        )
        
        # Update wave
        self.u_next = (
            2 * self.u - self.u_prev + 
            self.c**2 * dt**2 * laplacian
        ) * self.damping
        
        # Boundary conditions (fixed edges)
        self.u_next[0, :] = 0
        self.u_next[-1, :] = 0
        self.u_next[:, 0] = 0
        self.u_next[:, -1] = 0
        
        # Cycle buffers
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev
    
    def add_disturbance(self, x, y, amplitude=1.0):
        """Add a disturbance at given position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Add Gaussian disturbance
            cy, cx = np.ogrid[:self.height, :self.width]
            dist_sq = (cx - x)**2 + (cy - y)**2
            gaussian = amplitude * np.exp(-dist_sq / 50)
            self.u += gaussian
    
    def get_vertex_data(self):
        """Convert wave data to 3D vertices."""
        # Create grid of vertices
        y, x = np.mgrid[0:self.height, 0:self.width]
        
        # Normalize coordinates to [-1, 1]
        x = 2.0 * x / (self.width - 1) - 1.0
        y = 2.0 * y / (self.height - 1) - 1.0
        
        # Stack into vertex positions
        vertices = np.stack([x, self.u * 0.5, y], axis=-1)
        vertices = vertices.reshape(-1, 3).astype(np.float32)
        
        # Create colors based on height
        heights = vertices[:, 1]
        colors = np.zeros((len(vertices), 4), dtype=np.float32)
        colors[:, 0] = np.clip(heights + 0.5, 0, 1)  # Red for peaks
        colors[:, 2] = np.clip(-heights + 0.5, 0, 1)  # Blue for troughs
        colors[:, 1] = 0.2  # Slight green
        colors[:, 3] = 1.0  # Alpha
        
        return vertices, colors
    
    def get_indices(self):
        """Get triangle indices for the grid."""
        indices = []
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                # Two triangles per quad
                i = y * self.width + x
                indices.extend([
                    i, i + 1, i + self.width,
                    i + 1, i + self.width + 1, i + self.width
                ])
        
        return np.array(indices, dtype=np.uint32)


def main():
    """Main example function."""
    print("Live NumPy Buffer Update Example - Wave Simulation")
    print("=" * 50)
    
    # Initialize
    renderer = vf.Renderer(1920, 1080)
    allocator = vf.create_allocator()
    
    # Create wave simulation
    print("Initializing wave simulation...")
    wave = WaveSimulation(width=128, height=128)
    
    # Get initial data
    vertices, colors = wave.get_vertex_data()
    indices = wave.get_indices()
    
    print(f"Mesh size: {len(vertices):,} vertices, {len(indices)//3:,} triangles")
    
    # Create buffers
    print("Creating GPU buffers...")
    buffers = MultiBuffer(allocator)
    
    # Use zero-copy buffers for live updates
    vertex_buffer = buffers.add_vertex_buffer('vertices', vertices)
    color_buffer = buffers.add_vertex_buffer('colors', colors)
    index_buffer = buffers.add_index_buffer(indices)
    
    # Keep references to the arrays
    vertex_array = vertices
    color_array = colors
    
    print("\nStarting simulation...")
    print("Click to add disturbances (not implemented in this example)")
    
    frame_times = []
    update_times = []
    sync_times = []
    
    for frame in range(120):  # 20 seconds at 30 FPS
        frame_start = time.perf_counter()
        
        # Update simulation
        update_start = time.perf_counter()
        wave.step(dt=0.2)
        
        # Add periodic disturbances
        if frame % 60 == 0:
            x = np.random.randint(20, wave.width - 20)
            y = np.random.randint(20, wave.height - 20)
            wave.add_disturbance(x, y, amplitude=0.5)
        
        # Get new vertex data
        new_vertices, new_colors = wave.get_vertex_data()
        update_time = (time.perf_counter() - update_start) * 1000
        
        # Update buffers in-place
        sync_start = time.perf_counter()
        vertex_array[:] = new_vertices
        color_array[:] = new_colors
        
        # Sync to GPU
        vertex_buffer.sync_to_gpu()
        color_buffer.sync_to_gpu()
        sync_time = (time.perf_counter() - sync_start) * 1000
        
        # Render
        image = renderer.render_indexed(
            buffers,
            len(indices)
        )
        
        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)
        update_times.append(update_time)
        sync_times.append(sync_time)
        
        if frame % 30 == 0:
            avg_frame = np.mean(frame_times[-30:])
            avg_update = np.mean(update_times[-30:])
            avg_sync = np.mean(sync_times[-30:])
            
            print(f"Frame {frame}: Total: {avg_frame:.2f}ms "
                  f"(Update: {avg_update:.2f}ms, Sync: {avg_sync:.2f}ms) "
                  f"- {1000/avg_frame:.0f} FPS")
    
    # Save final frame
    vf.save_image(image, "wave_simulation.png")
    
    print("\nPerformance Summary:")
    print(f"Average frame time: {np.mean(frame_times):.2f}ms")
    print(f"Average update time: {np.mean(update_times):.2f}ms") 
    print(f"Average sync time: {np.mean(sync_times):.2f}ms")
    print(f"Sync bandwidth: {(vertices.nbytes + colors.nbytes) / 1024 / 1024 / (np.mean(sync_times) / 1000):.0f} MB/s")


if __name__ == "__main__":
    main()