"""Example: Compute shader writing to NumPy arrays."""

import numpy as np
import vulkan_forge as vf
from vulkan_forge.numpy_buffer import create_storage_buffer
import time


def main():
    """Main example function."""
    print("NumPy Compute Shader Example - Particle Simulation")
    print("=" * 50)
    
    # Initialize
    renderer = vf.Renderer(1920, 1080)
    allocator = vf.create_allocator()
    
    # Create particle data
    n_particles = 1_000_000
    print(f"Creating {n_particles:,} particles...")
    
    # Particle structure: position (vec3) + velocity (vec3) + life (float) + pad
    particle_dtype = np.dtype([
        ('position', np.float32, 3),
        ('velocity', np.float32, 3),
        ('life', np.float32),
        ('_pad', np.float32)  # Padding for alignment
    ])
    
    particles = np.zeros(n_particles, dtype=particle_dtype)
    
    # Initialize particles
    # Random positions in cube
    particles['position'] = np.random.uniform(-1, 1, (n_particles, 3)).astype(np.float32)
    
    # Random velocities
    particles['velocity'] = np.random.randn(n_particles, 3).astype(np.float32) * 0.1
    
    # Random life times
    particles['life'] = np.random.uniform(0, 1, n_particles).astype(np.float32)
    
    # Create storage buffer (read/write for compute)
    print("Creating GPU storage buffer...")
    particle_buffer = create_storage_buffer(allocator, particles, read_only=False)
    
    # Create compute pipeline for particle update
    compute_shader = """
    #version 450
    
    layout(local_size_x = 64) in;
    
    struct Particle {
        vec3 position;
        vec3 velocity;
        float life;
        float _pad;
    };
    
    layout(set = 0, binding = 0) buffer ParticleBuffer {
        Particle particles[];
    } particleBuffer;
    
    layout(push_constant) uniform PushConstants {
        float deltaTime;
        float time;
    } pc;
    
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        if (idx >= particleBuffer.particles.length()) return;
        
        Particle p = particleBuffer.particles[idx];
        
        // Update position
        p.position += p.velocity * pc.deltaTime;
        
        // Apply gravity
        p.velocity.y -= 9.81 * pc.deltaTime;
        
        // Update life
        p.life -= pc.deltaTime * 0.5;
        
        // Respawn dead particles
        if (p.life <= 0.0) {
            // Reset at emitter
            p.position = vec3(0.0);
            p.velocity = vec3(
                sin(pc.time * float(idx)) * 2.0,
                5.0 + cos(pc.time * float(idx)) * 2.0,
                sin(pc.time * float(idx) * 0.7) * 2.0
            );
            p.life = 1.0;
        }
        
        // Bounce off ground
        if (p.position.y < -1.0) {
            p.position.y = -1.0;
            p.velocity.y = abs(p.velocity.y) * 0.8;
        }
        
        particleBuffer.particles[idx] = p;
    }
    """
    
    # In real implementation, compile shader and create pipeline
    # compute_pipeline = vf.create_compute_pipeline(compute_shader)
    
    print("\nRunning particle simulation...")
    
    frame_times = []
    compute_times = []
    total_time = 0.0
    
    for frame in range(300):  # 10 seconds at 30 FPS
        frame_start = time.perf_counter()
        delta_time = 1.0 / 30.0
        total_time += delta_time
        
        # Run compute shader
        compute_start = time.perf_counter()
        
        # In real implementation:
        # renderer.dispatch_compute(
        #     compute_pipeline,
        #     particle_buffer,
        #     workgroups=(n_particles + 63) // 64,
        #     push_constants={'deltaTime': delta_time, 'time': total_time}
        # )
        
        # For demo, simulate compute shader in NumPy
        particles['position'] += particles['velocity'] * delta_time
        particles['velocity'][:, 1] -= 9.81 * delta_time
        particles['life'] -= delta_time * 0.5
        
        # Respawn dead particles
        dead = particles['life'] <= 0
        n_dead = np.sum(dead)
        if n_dead > 0:
            particles['position'][dead] = 0
            # Random directions
            angles = np.random.uniform(0, 2 * np.pi, n_dead)
            speeds = np.random.uniform(3, 7, n_dead)
            particles['velocity'][dead, 0] = np.cos(angles) * speeds
            particles['velocity'][dead, 1] = np.random.uniform(5, 10, n_dead)
            particles['velocity'][dead, 2] = np.sin(angles) * speeds
            particles['life'][dead] = 1.0
        
        # Bounce off ground
        below_ground = particles['position'][:, 1] < -1.0
        particles['position'][below_ground, 1] = -1.0
        particles['velocity'][below_ground, 1] = np.abs(particles['velocity'][below_ground, 1]) * 0.8
        
        # Sync to GPU
        particle_buffer.sync_to_gpu()
        
        compute_time = (time.perf_counter() - compute_start) * 1000
        
        # Render particles
        # Create vertex buffer view from particle positions
        positions = particles['position'].copy()
        
        # Color based on velocity
        speeds = np.linalg.norm(particles['velocity'], axis=1)
        colors = np.zeros((n_particles, 4), dtype=np.float32)
        colors[:, 0] = speeds / 10  # Red for fast
        colors[:, 2] = 1 - speeds / 10  # Blue for slow
        colors[:, 1] = particles['life']  # Green for life
        colors[:, 3] = particles['life']  # Alpha fade out
        
        # In real implementation, render point sprites
        # image = renderer.render_particles(positions, colors)
        
        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)
        compute_times.append(compute_time)
        
        if frame % 30 == 0:
            avg_frame = np.mean(frame_times[-30:])
            avg_compute = np.mean(compute_times[-30:])
            print(f"Frame {frame}: {avg_frame:.2f}ms "
                  f"(Compute: {avg_compute:.2f}ms) "
                  f"- {1000/avg_frame:.0f} FPS")
    
    print("\nPerformance Summary:")
    print(f"Average frame time: {np.mean(frame_times):.2f}ms")
    print(f"Average compute time: {np.mean(compute_times):.2f}ms")
    print(f"Particles processed per second: {n_particles * 30 / 1_000_000:.1f}M")
    
    # Read back final particle state
    print("\nReading back particle data...")
    particle_buffer.sync_from_gpu()
    
    # Analyze particle distribution
    positions = particles['position']
    print(f"Particle bounds:")
    print(f"  X: [{np.min(positions[:, 0]):.2f}, {np.max(positions[:, 0]):.2f}]")
    print(f"  Y: [{np.min(positions[:, 1]):.2f}, {np.max(positions[:, 1]):.2f}]")
    print(f"  Z: [{np.min(positions[:, 2]):.2f}, {np.max(positions[:, 2]):.2f}]")
    
    alive_particles = np.sum(particles['life'] > 0)
    print(f"Alive particles: {alive_particles:,} ({100 * alive_particles / n_particles:.1f}%)")


if __name__ == "__main__":
    main()