"""Example: Render 1 million points from NumPy array - Simplified version."""

import numpy as np
import vulkan_forge as vf
import time


def generate_point_cloud(n_points=10_000):
    """Generate a random 3D point cloud."""
    # Generate random points in a sphere
    # Using spherical coordinates for better distribution
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0, 1, n_points) ** (1 / 3)  # Uniform volume distribution

    # Convert to Cartesian coordinates
    points = np.zeros((n_points, 3), dtype=np.float32)
    points[:, 0] = r * np.sin(phi) * np.cos(theta)  # x
    points[:, 1] = r * np.sin(phi) * np.sin(theta)  # y
    points[:, 2] = r * np.cos(phi)  # z

    # Generate colors based on position
    colors = np.zeros((n_points, 4), dtype=np.float32)
    colors[:, 0] = (points[:, 0] + 1) * 0.5  # R from X
    colors[:, 1] = (points[:, 1] + 1) * 0.5  # G from Y
    colors[:, 2] = (points[:, 2] + 1) * 0.5  # B from Z
    colors[:, 3] = 1.0  # Alpha

    return points, colors


def main():
    """Main example function."""
    print("NumPy Point Cloud Example (Simplified)")
    print("=" * 50)

    # Initialize Vulkan Forge
    print("Initializing renderer...")
    renderer = vf.Renderer(1920, 1080)

    # Generate point cloud
    print(f"Generating 100,000 points...")
    start_time = time.perf_counter()
    points, colors = generate_point_cloud(100_000)  # Reduced for testing
    generation_time = (time.perf_counter() - start_time) * 1000
    print(f"Generation time: {generation_time:.2f}ms")

    # Animation loop
    print("\nRendering point cloud...")
    frame_times = []

    for frame in range(30):  # Just 30 frames for testing
        frame_start = time.perf_counter()

        # Animate points (rotation)
        angle = frame * 0.01
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Rotate around Y axis (in-place modification)
        x_rot = points[:, 0] * cos_a - points[:, 2] * sin_a
        z_rot = points[:, 0] * sin_a + points[:, 2] * cos_a
        points[:, 0] = x_rot
        points[:, 2] = z_rot

        # Render
        try:
            image = renderer.render_points(points)
        except Exception as e:
            print(f"Render error: {e}")
            # Create a test image
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            image[::10, ::10] = [255, 0, 0]  # Red dots

        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)

        if frame % 10 == 0:
            avg_time = np.mean(frame_times[-10:]) if frame_times else frame_time
            print(f"Frame {frame}: {avg_time:.2f}ms ({1000/avg_time:.0f} FPS)")

    # Save final frame
    print("\nSaving final frame...")
    try:
        vf.save_image(image, "pointcloud_final.png")
        print("Image saved to pointcloud_final.png")
    except Exception as e:
        print(f"Failed to save image: {e}")

    print("\nPerformance Summary:")
    if frame_times:
        print(f"Average frame time: {np.mean(frame_times):.2f}ms")
        print(f"Min frame time: {np.min(frame_times):.2f}ms")
        print(f"Max frame time: {np.max(frame_times):.2f}ms")
        print(f"99th percentile: {np.percentile(frame_times, 99):.2f}ms")


if __name__ == "__main__":
    main()