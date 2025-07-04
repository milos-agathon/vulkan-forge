"""Example: Render 1 million points from NumPy array with zero-copy upload."""

import numpy as np
import vulkan_forge as vf
from vulkan_forge.numpy_buffer import numpy_buffer
import time


def generate_point_cloud(n_points=10_000):
    """Generate a random 3D point cloud."""
    # Generate random points in a sphere
    # Using spherical coordinates for better distribution
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0, 1, n_points) ** (1/3)  # Uniform volume distribution
    
    # Convert to Cartesian coordinates
    points = np.zeros((n_points, 3), dtype=np.float32)
    points[:, 0] = r * np.sin(phi) * np.cos(theta)  # x
    points[:, 1] = r * np.sin(phi) * np.sin(theta)  # y
    points[:, 2] = r * np.cos(phi)                  # z
    
    # Generate colors based on position
    colors = np.zeros((n_points, 4), dtype=np.float32)
    colors[:, 0] = (points[:, 0] + 1) * 0.5  # R from X
    colors[:, 1] = (points[:, 1] + 1) * 0.5  # G from Y
    colors[:, 2] = (points[:, 2] + 1) * 0.5  # B from Z
    colors[:, 3] = 1.0                        # Alpha
    
    return points, colors


def main():
    """Main example function."""
    print("NumPy Point Cloud Example")
    print("=" * 50)
    
    # Initialize Vulkan Forge
    renderer = vf.Renderer(1920, 1080)
    allocator = vf.create_allocator()
    
    # Generate point cloud
    print(f"Generating 1 million points...")
    start_time = time.perf_counter()
    points, colors = generate_point_cloud(10_000)
    generation_time = (time.perf_counter() - start_time) * 1000
    print(f"Generation time: {generation_time:.2f}ms")
    
    # Upload to GPU using zero-copy
    print(f"Uploading to GPU...")
    start_time = time.perf_counter()
    
    with numpy_buffer(allocator, points, vf.BUFFER_USAGE_VERTEX) as point_buffer:
        with numpy_buffer(allocator, colors, vf.BUFFER_USAGE_VERTEX) as color_buffer:
            upload_time = (time.perf_counter() - start_time) * 1000
            print(f"Upload time: {upload_time:.2f}ms")
            print(f"Bandwidth: {(points.nbytes + colors.nbytes) / 1024 / 1024 / (upload_time / 1000):.0f} MB/s")
            
            # Set up rendering
            renderer.set_vertex_buffer(point_buffer, binding=0)
            renderer.set_vertex_buffer(color_buffer, binding=1)
            
            # Animation loop
            print("\nRendering point cloud...")
            frame_times = []
            
            for frame in range(300):  # 10 seconds at 30 FPS
                frame_start = time.perf_counter()
                
                # Animate points (rotation)
                angle = frame * 0.01
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                # Rotate around Y axis (in-place modification)
                x_rot = points[:, 0] * cos_a - points[:, 2] * sin_a
                z_rot = points[:, 0] * sin_a + points[:, 2] * cos_a
                points[:, 0] = x_rot
                points[:, 2] = z_rot
                
                # Sync changes to GPU
                point_buffer.sync_to_gpu()
                
                # Render
                image = renderer.render_points()
                
                frame_time = (time.perf_counter() - frame_start) * 1000
                frame_times.append(frame_time)
                
                if frame % 30 == 0:
                    avg_time = np.mean(frame_times[-30:])
                    print(f"Frame {frame}: {avg_time:.2f}ms ({1000/avg_time:.0f} FPS)")
            
            # Save final frame
            vf.save_image(image, "pointcloud_final.png")
    
    print("\nPerformance Summary:")
    print(f"Average frame time: {np.mean(frame_times):.2f}ms")
    print(f"Min frame time: {np.min(frame_times):.2f}ms")
    print(f"Max frame time: {np.max(frame_times):.2f}ms")
    print(f"99th percentile: {np.percentile(frame_times, 99):.2f}ms")


if __name__ == "__main__":
    main()