// Memory Management and GPU-Driven Rendering Tests
// Tests specialized terrain memory pools, GPU quadtree culling, and VMA integration

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <random>

#include "vf/terrain_memory_pool.hpp"
#include "vf/gpu_quadtree.hpp"
#include "vf/vulkan_context.hpp"
#include "vf/terrain_tile.hpp"

using namespace vf;
using ::testing::_;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::NiceMock;

// Mock VMA Allocator for testing without real GPU
class MockVmaAllocator {
public:
    MOCK_METHOD(VkResult, createBuffer, 
                (const VkBufferCreateInfo*, const VmaAllocationCreateInfo*, 
                 VkBuffer*, VmaAllocation*, VmaAllocationInfo*));
    MOCK_METHOD(void, destroyBuffer, (VkBuffer, VmaAllocation));
    MOCK_METHOD(VkResult, mapMemory, (VmaAllocation, void**));
    MOCK_METHOD(void, unmapMemory, (VmaAllocation));
    MOCK_METHOD(VkResult, createImage,
                (const VkImageCreateInfo*, const VmaAllocationCreateInfo*,
                 VkImage*, VmaAllocation*, VmaAllocationInfo*));
    MOCK_METHOD(void, destroyImage, (VkImage, VmaAllocation));
    MOCK_METHOD(void, getMemoryProperties, (VmaMemoryProperties*));
    MOCK_METHOD(void, getBudget, (VmaBudget*));
};

// Mock Vulkan Context for memory testing
class MockVulkanContext : public VulkanContext {
public:
    MOCK_METHOD(VkDevice, get_device, (), (const, override));
    MOCK_METHOD(VkPhysicalDevice, get_physical_device, (), (const, override));
    MOCK_METHOD(VmaAllocator, get_allocator, (), (const, override));
    MOCK_METHOD(VkCommandPool, get_command_pool, (), (const, override));
    MOCK_METHOD(VkQueue, get_graphics_queue, (), (const, override));
    MOCK_METHOD(VkQueue, get_compute_queue, (), (const, override));
    
    MockVulkanContext() {
        ON_CALL(*this, get_device())
            .WillByDefault(Return(reinterpret_cast<VkDevice>(0x12345678)));
        ON_CALL(*this, get_allocator())
            .WillByDefault(Return(reinterpret_cast<VmaAllocator>(0x87654321)));
    }
};

// Test fixture for memory management tests
class TerrainMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_context = std::make_unique<NiceMock<MockVulkanContext>>();
        
        // Configure pool settings
        pool_config.max_buffer_size_mb = 256;
        pool_config.max_texture_size_mb = 512;
        pool_config.alignment = 256;
        pool_config.enable_dedicated_allocations = true;
        pool_config.enable_memory_mapping = true;
    }
    
    void TearDown() override {
        mock_context.reset();
    }
    
    std::unique_ptr<MockVulkanContext> mock_context;
    TerrainMemoryPoolConfig pool_config;
};

// Test TerrainMemoryPool initialization and configuration
TEST_F(TerrainMemoryTest, MemoryPoolInitialization) {
    EXPECT_NO_THROW({
        TerrainMemoryPool pool(*mock_context, pool_config);
        
        EXPECT_EQ(pool.get_max_buffer_size(), pool_config.max_buffer_size_mb * 1024 * 1024);
        EXPECT_EQ(pool.get_max_texture_size(), pool_config.max_texture_size_mb * 1024 * 1024);
        EXPECT_EQ(pool.get_alignment(), pool_config.alignment);
    });
}

// Test vertex buffer allocation
TEST_F(TerrainMemoryTest, VertexBufferAllocation) {
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    constexpr size_t vertex_count = 1000;
    constexpr size_t vertex_size = sizeof(TerrainVertex);
    constexpr size_t buffer_size = vertex_count * vertex_size;
    
    // Mock successful allocation
    EXPECT_CALL(*mock_context, get_device())
        .WillRepeatedly(Return(reinterpret_cast<VkDevice>(0x12345678)));
    
    auto buffer_allocation = pool.allocate_vertex_buffer(buffer_size);
    EXPECT_NE(buffer_allocation, nullptr);
    
    if (buffer_allocation) {
        EXPECT_NE(buffer_allocation->buffer, VK_NULL_HANDLE);
        EXPECT_NE(buffer_allocation->allocation, VK_NULL_HANDLE);
        EXPECT_GE(buffer_allocation->size, buffer_size);
        
        // Test deallocation
        EXPECT_NO_THROW({
            pool.deallocate_vertex_buffer(std::move(buffer_allocation));
        });
    }
}

// Test index buffer allocation
TEST_F(TerrainMemoryTest, IndexBufferAllocation) {
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    constexpr size_t index_count = 6000; // 2000 triangles
    constexpr size_t index_size = sizeof(uint32_t);
    constexpr size_t buffer_size = index_count * index_size;
    
    auto buffer_allocation = pool.allocate_index_buffer(buffer_size);
    EXPECT_NE(buffer_allocation, nullptr);
    
    if (buffer_allocation) {
        EXPECT_NE(buffer_allocation->buffer, VK_NULL_HANDLE);
        EXPECT_GE(buffer_allocation->size, buffer_size);
        
        pool.deallocate_index_buffer(std::move(buffer_allocation));
    }
}

// Test texture allocation
TEST_F(TerrainMemoryTest, TextureAllocation) {
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R32_SFLOAT; // Height texture
    image_info.extent = {1024, 1024, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    
    auto texture_allocation = pool.allocate_texture(image_info);
    EXPECT_NE(texture_allocation, nullptr);
    
    if (texture_allocation) {
        EXPECT_NE(texture_allocation->image, VK_NULL_HANDLE);
        EXPECT_NE(texture_allocation->allocation, VK_NULL_HANDLE);
        
        pool.deallocate_texture(std::move(texture_allocation));
    }
}

// Test memory pool statistics
TEST_F(TerrainMemoryTest, MemoryStatistics) {
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    // Get initial statistics
    auto initial_stats = pool.get_statistics();
    EXPECT_EQ(initial_stats.total_allocations, 0);
    EXPECT_EQ(initial_stats.bytes_allocated, 0);
    EXPECT_EQ(initial_stats.active_allocations, 0);
    
    // Allocate some buffers
    constexpr size_t buffer_size = 1024 * 1024; // 1MB
    auto buffer1 = pool.allocate_vertex_buffer(buffer_size);
    auto buffer2 = pool.allocate_index_buffer(buffer_size);
    
    // Check updated statistics
    auto updated_stats = pool.get_statistics();
    EXPECT_GT(updated_stats.total_allocations, 0);
    EXPECT_GT(updated_stats.bytes_allocated, 0);
    EXPECT_GT(updated_stats.active_allocations, 0);
    
    // Deallocate and check again
    pool.deallocate_vertex_buffer(std::move(buffer1));
    pool.deallocate_index_buffer(std::move(buffer2));
    
    auto final_stats = pool.get_statistics();
    EXPECT_EQ(final_stats.active_allocations, 0);
}

// Test memory alignment requirements
TEST_F(TerrainMemoryTest, MemoryAlignment) {
    pool_config.alignment = 1024; // Large alignment for testing
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    constexpr size_t buffer_size = 777; // Non-aligned size
    auto buffer_allocation = pool.allocate_vertex_buffer(buffer_size);
    
    if (buffer_alignment) {
        // Check that allocated size respects alignment
        EXPECT_EQ(buffer_allocation->size % pool_config.alignment, 0);
        EXPECT_GE(buffer_allocation->size, buffer_size);
        
        pool.deallocate_vertex_buffer(std::move(buffer_allocation));
    }
}

// Test memory pool limits
TEST_F(TerrainMemoryTest, MemoryLimits) {
    pool_config.max_buffer_size_mb = 1; // Very small limit
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    constexpr size_t large_buffer_size = 2 * 1024 * 1024; // 2MB > 1MB limit
    
    // Should fail to allocate buffer larger than limit
    auto buffer_allocation = pool.allocate_vertex_buffer(large_buffer_size);
    EXPECT_EQ(buffer_allocation, nullptr);
}

// Test memory fragmentation handling
TEST_F(TerrainMemoryTest, FragmentationHandling) {
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    std::vector<std::unique_ptr<BufferAllocation>> allocations;
    
    // Allocate many small buffers
    constexpr size_t small_buffer_size = 1024;
    constexpr size_t num_allocations = 100;
    
    for (size_t i = 0; i < num_allocations; ++i) {
        auto allocation = pool.allocate_vertex_buffer(small_buffer_size);
        if (allocation) {
            allocations.push_back(std::move(allocation));
        }
    }
    
    // Deallocate every other buffer to create fragmentation
    for (size_t i = 0; i < allocations.size(); i += 2) {
        pool.deallocate_vertex_buffer(std::move(allocations[i]));
        allocations[i] = nullptr;
    }
    
    // Try to allocate a larger buffer
    constexpr size_t large_buffer_size = 10 * small_buffer_size;
    auto large_allocation = pool.allocate_vertex_buffer(large_buffer_size);
    
    // Should still succeed despite fragmentation
    EXPECT_NE(large_allocation, nullptr);
    
    if (large_allocation) {
        pool.deallocate_vertex_buffer(std::move(large_allocation));
    }
    
    // Cleanup remaining allocations
    for (auto& allocation : allocations) {
        if (allocation) {
            pool.deallocate_vertex_buffer(std::move(allocation));
        }
    }
}

// Test fixture for GPU quadtree tests
class GPUQuadtreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_context = std::make_unique<NiceMock<MockVulkanContext>>();
        
        // Configure quadtree
        quadtree_config.max_depth = 8;
        quadtree_config.max_objects_per_node = 16;
        quadtree_config.bounds = AxisAlignedBoundingBox(
            Eigen::Vector3f(-1000, -1000, -100),
            Eigen::Vector3f(1000, 1000, 1000)
        );
        quadtree_config.enable_gpu_culling = true;
        quadtree_config.cull_threads = 64;
    }
    
    std::unique_ptr<MockVulkanContext> mock_context;
    GPUQuadtreeConfig quadtree_config;
};

// Test GPU quadtree initialization
TEST_F(GPUQuadtreeTest, QuadtreeInitialization) {
    EXPECT_NO_THROW({
        GPUQuadtree quadtree(*mock_context, quadtree_config);
        
        EXPECT_EQ(quadtree.get_max_depth(), quadtree_config.max_depth);
        EXPECT_EQ(quadtree.get_max_objects_per_node(), quadtree_config.max_objects_per_node);
        EXPECT_TRUE(quadtree.is_gpu_culling_enabled());
    });
}

// Test object insertion into quadtree
TEST_F(GPUQuadtreeTest, ObjectInsertion) {
    GPUQuadtree quadtree(*mock_context, quadtree_config);
    
    // Create test objects (terrain tiles)
    std::vector<CullableObject> objects;
    for (int i = 0; i < 100; ++i) {
        CullableObject obj;
        obj.id = i;
        obj.bounds = AxisAlignedBoundingBox(
            Eigen::Vector3f(i * 10, i * 10, 0),
            Eigen::Vector3f(i * 10 + 50, i * 10 + 50, 100)
        );
        obj.lod_level = 0;
        obj.visible = true;
        objects.push_back(obj);
    }
    
    // Insert objects
    for (const auto& obj : objects) {
        bool inserted = quadtree.insert(obj);
        EXPECT_TRUE(inserted);
    }
    
    EXPECT_EQ(quadtree.get_object_count(), objects.size());
}

// Test frustum culling
TEST_F(GPUQuadtreeTest, FrustumCulling) {
    GPUQuadtree quadtree(*mock_context, quadtree_config);
    
    // Insert test objects
    std::vector<CullableObject> objects;
    for (int i = 0; i < 50; ++i) {
        CullableObject obj;
        obj.id = i;
        obj.bounds = AxisAlignedBoundingBox(
            Eigen::Vector3f(i * 100, 0, 0),
            Eigen::Vector3f(i * 100 + 50, 50, 50)
        );
        objects.push_back(obj);
        quadtree.insert(obj);
    }
    
    // Create camera frustum
    Camera camera;
    camera.set_position(Eigen::Vector3f(500, 0, 200));
    camera.look_at(Eigen::Vector3f(500, 0, 0));
    camera.set_fov(60.0f);
    camera.set_aspect_ratio(16.0f / 9.0f);
    camera.set_near_far(1.0f, 2000.0f);
    
    // Perform culling
    auto cull_results = quadtree.cull_frustum(camera.get_frustum_planes());
    
    EXPECT_GT(cull_results.visible_objects.size(), 0);
    EXPECT_LT(cull_results.visible_objects.size(), objects.size()); // Some should be culled
    EXPECT_GT(cull_results.culled_objects.size(), 0);
    
    // Verify culling statistics
    EXPECT_EQ(cull_results.visible_objects.size() + cull_results.culled_objects.size(), 
              objects.size());
}

// Test LOD selection
TEST_F(GPUQuadtreeTest, LODSelection) {
    GPUQuadtree quadtree(*mock_context, quadtree_config);
    
    // Insert objects at different distances with different LOD levels
    std::vector<CullableObject> objects;
    for (int lod = 0; lod < 4; ++lod) {
        for (int i = 0; i < 10; ++i) {
            CullableObject obj;
            obj.id = lod * 10 + i;
            obj.lod_level = lod;
            
            // Place objects at increasing distances
            float distance = (lod + 1) * 500.0f + i * 50.0f;
            obj.bounds = AxisAlignedBoundingBox(
                Eigen::Vector3f(distance, 0, 0),
                Eigen::Vector3f(distance + 25, 25, 25)
            );
            
            objects.push_back(obj);
            quadtree.insert(obj);
        }
    }
    
    // Create camera looking along X axis
    Camera camera;
    camera.set_position(Eigen::Vector3f(0, 0, 100));
    camera.look_at(Eigen::Vector3f(1000, 0, 0));
    
    // Perform LOD culling
    auto cull_results = quadtree.cull_lod(camera.get_position(), {100, 500, 1000, 2000});
    
    // Verify that appropriate LOD levels are selected
    EXPECT_GT(cull_results.visible_objects.size(), 0);
    
    // Check that closer objects have lower LOD levels (higher detail)
    for (const auto& obj : cull_results.visible_objects) {
        float distance = (obj.bounds.center() - camera.get_position()).norm();
        
        if (distance < 200) {
            EXPECT_LE(obj.lod_level, 1); // Close objects should be high detail
        } else if (distance > 1500) {
            EXPECT_GE(obj.lod_level, 2); // Far objects should be low detail
        }
    }
}

// Test GPU compute shader culling
TEST_F(GPUQuadtreeTest, GPUComputeCulling) {
    if (!quadtree_config.enable_gpu_culling) {
        GTEST_SKIP() << "GPU culling disabled";
    }
    
    GPUQuadtree quadtree(*mock_context, quadtree_config);
    
    // Insert many objects to justify GPU culling
    constexpr size_t num_objects = 10000;
    std::vector<CullableObject> objects;
    objects.reserve(num_objects);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-800, 800);
    
    for (size_t i = 0; i < num_objects; ++i) {
        CullableObject obj;
        obj.id = static_cast<uint32_t>(i);
        
        Eigen::Vector3f center(pos_dist(gen), pos_dist(gen), pos_dist(gen) * 0.1f);
        Eigen::Vector3f size(25, 25, 25);
        
        obj.bounds = AxisAlignedBoundingBox(center - size, center + size);
        obj.lod_level = i % 4;
        
        objects.push_back(obj);
        quadtree.insert(obj);
    }
    
    // Setup camera
    Camera camera;
    camera.set_position(Eigen::Vector3f(0, 0, 500));
    camera.look_at(Eigen::Vector3f(0, 0, 0));
    
    // Perform GPU culling
    auto start_time = std::chrono::high_resolution_clock::now();
    auto cull_results = quadtree.cull_gpu_compute(camera.get_frustum_planes(), camera.get_position());
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Verify results
    EXPECT_GT(cull_results.visible_objects.size(), 0);
    EXPECT_LT(cull_results.visible_objects.size(), num_objects);
    
    // GPU culling should be fast
    EXPECT_LT(duration.count(), 10000); // Less than 10ms
    
    // Verify culling accuracy (basic sanity check)
    size_t objects_in_frustum = 0;
    for (const auto& obj : objects) {
        if (camera.is_sphere_in_frustum(obj.bounds.center(), obj.bounds.radius())) {
            objects_in_frustum++;
        }
    }
    
    // GPU culling should find similar number of objects (within some tolerance)
    float accuracy = static_cast<float>(cull_results.visible_objects.size()) / objects_in_frustum;
    EXPECT_GT(accuracy, 0.8f); // At least 80% accuracy
    EXPECT_LT(accuracy, 1.2f); // At most 20% over-estimation
}

// Test quadtree performance with dynamic objects
TEST_F(GPUQuadtreeTest, DynamicObjectPerformance) {
    GPUQuadtree quadtree(*mock_context, quadtree_config);
    
    constexpr size_t num_objects = 1000;
    std::vector<CullableObject> objects;
    
    // Insert initial objects
    for (size_t i = 0; i < num_objects; ++i) {
        CullableObject obj;
        obj.id = static_cast<uint32_t>(i);
        obj.bounds = AxisAlignedBoundingBox(
            Eigen::Vector3f(i % 100 * 10, (i / 100) * 10, 0),
            Eigen::Vector3f((i % 100) * 10 + 5, (i / 100) * 10 + 5, 5)
        );
        objects.push_back(obj);
        quadtree.insert(obj);
    }
    
    // Measure update performance
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulate object movement
    for (size_t frame = 0; frame < 100; ++frame) {
        // Move some objects
        for (size_t i = 0; i < num_objects / 10; ++i) {
            size_t obj_index = (frame * 7 + i) % objects.size();
            CullableObject& obj = objects[obj_index];
            
            // Remove from old position
            quadtree.remove(obj.id);
            
            // Update position
            Eigen::Vector3f movement(std::sin(frame * 0.1f + i), std::cos(frame * 0.1f + i), 0);
            obj.bounds.translate(movement);
            
            // Reinsert at new position
            quadtree.insert(obj);
        }
        
        // Perform culling
        Camera camera;
        camera.set_position(Eigen::Vector3f(frame * 5, frame * 5, 100));
        auto cull_results = quadtree.cull_frustum(camera.get_frustum_planes());
        
        // Verify some objects are still visible
        EXPECT_GT(cull_results.visible_objects.size(), 0);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Performance should be reasonable (< 100ms for 100 frames)
    EXPECT_LT(duration.count(), 100);
}

// Test memory pool thread safety
TEST_F(TerrainMemoryTest, ThreadSafety) {
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    constexpr int num_threads = 4;
    constexpr int allocations_per_thread = 100;
    constexpr size_t buffer_size = 1024;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<std::unique_ptr<BufferAllocation>>> thread_allocations(num_threads);
    std::atomic<int> total_allocations(0);
    std::atomic<int> failed_allocations(0);
    
    // Launch worker threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            thread_allocations[t].reserve(allocations_per_thread);
            
            for (int i = 0; i < allocations_per_thread; ++i) {
                auto allocation = pool.allocate_vertex_buffer(buffer_size);
                if (allocation) {
                    thread_allocations[t].push_back(std::move(allocation));
                    total_allocations++;
                } else {
                    failed_allocations++;
                }
                
                // Small delay to encourage race conditions
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify results
    EXPECT_GT(total_allocations.load(), 0);
    EXPECT_LT(failed_allocations.load(), total_allocations.load());
    
    // Cleanup allocations from all threads
    for (int t = 0; t < num_threads; ++t) {
        for (auto& allocation : thread_allocations[t]) {
            pool.deallocate_vertex_buffer(std::move(allocation));
        }
    }
    
    // Verify final state
    auto final_stats = pool.get_statistics();
    EXPECT_EQ(final_stats.active_allocations, 0);
}

// Test memory budget management
TEST_F(TerrainMemoryTest, MemoryBudgetManagement) {
    // Configure pool with tight memory budget
    pool_config.enable_budget_management = true;
    pool_config.memory_budget_mb = 64; // Small budget
    pool_config.budget_warning_threshold = 0.8f; // 80%
    
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    constexpr size_t large_buffer_size = 10 * 1024 * 1024; // 10MB
    std::vector<std::unique_ptr<BufferAllocation>> allocations;
    
    // Allocate until budget is exceeded
    for (int i = 0; i < 10; ++i) {
        auto allocation = pool.allocate_vertex_buffer(large_buffer_size);
        if (allocation) {
            allocations.push_back(std::move(allocation));
        } else {
            break; // Budget exceeded
        }
        
        auto budget_info = pool.get_budget_info();
        if (budget_info.usage_ratio > pool_config.budget_warning_threshold) {
            // Should trigger warning
            EXPECT_TRUE(budget_info.warning_triggered);
        }
    }
    
    EXPECT_GT(allocations.size(), 0);
    EXPECT_LT(allocations.size(), 10); // Should not allocate all due to budget
    
    // Check final budget state
    auto final_budget = pool.get_budget_info();
    EXPECT_GT(final_budget.usage_ratio, 0.0f);
    EXPECT_LE(final_budget.usage_ratio, 1.0f);
    
    // Cleanup
    for (auto& allocation : allocations) {
        pool.deallocate_vertex_buffer(std::move(allocation));
    }
}

// Performance benchmarks
class MemoryPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_context = std::make_unique<NiceMock<MockVulkanContext>>();
        pool_config.max_buffer_size_mb = 1024; // Large pool for performance testing
    }
    
    std::unique_ptr<MockVulkanContext> mock_context;
    TerrainMemoryPoolConfig pool_config;
};

// Benchmark allocation performance
TEST_F(MemoryPerformanceTest, AllocationBenchmark) {
    TerrainMemoryPool pool(*mock_context, pool_config);
    
    constexpr int num_allocations = 10000;
    constexpr size_t buffer_size = 4096; // 4KB buffers
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::unique_ptr<BufferAllocation>> allocations;
    allocations.reserve(num_allocations);
    
    // Allocation benchmark
    for (int i = 0; i < num_allocations; ++i) {
        auto allocation = pool.allocate_vertex_buffer(buffer_size);
        if (allocation) {
            allocations.push_back(std::move(allocation));
        }
    }
    
    auto alloc_end_time = std::chrono::high_resolution_clock::now();
    
    // Deallocation benchmark
    for (auto& allocation : allocations) {
        pool.deallocate_vertex_buffer(std::move(allocation));
    }
    
    auto dealloc_end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate performance metrics
    auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        alloc_end_time - start_time);
    auto dealloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        dealloc_end_time - alloc_end_time);
    
    double alloc_rate = static_cast<double>(allocations.size()) / alloc_duration.count() * 1e6; // allocs/sec
    double dealloc_rate = static_cast<double>(allocations.size()) / dealloc_duration.count() * 1e6; // deallocs/sec
    
    std::cout << "Allocation performance:" << std::endl;
    std::cout << "  Allocations: " << allocations.size() << std::endl;
    std::cout << "  Allocation rate: " << alloc_rate << " allocs/sec" << std::endl;
    std::cout << "  Deallocation rate: " << dealloc_rate << " deallocs/sec" << std::endl;
    
    // Performance assertions
    EXPECT_GT(alloc_rate, 100000); // At least 100K allocs/sec
    EXPECT_GT(dealloc_rate, 100000); // At least 100K deallocs/sec
}

// Benchmark culling performance
TEST_F(MemoryPerformanceTest, CullingBenchmark) {
    GPUQuadtreeConfig quadtree_config;
    quadtree_config.max_depth = 10;
    quadtree_config.max_objects_per_node = 32;
    quadtree_config.bounds = AxisAlignedBoundingBox(
        Eigen::Vector3f(-5000, -5000, -500),
        Eigen::Vector3f(5000, 5000, 500)
    );
    
    GPUQuadtree quadtree(*mock_context, quadtree_config);
    
    // Insert many objects
    constexpr size_t num_objects = 50000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-4000, 4000);
    
    for (size_t i = 0; i < num_objects; ++i) {
        CullableObject obj;
        obj.id = static_cast<uint32_t>(i);
        
        Eigen::Vector3f center(pos_dist(gen), pos_dist(gen), pos_dist(gen) * 0.1f);
        obj.bounds = AxisAlignedBoundingBox(center - Eigen::Vector3f(10, 10, 10),
                                          center + Eigen::Vector3f(10, 10, 10));
        quadtree.insert(obj);
    }
    
    // Benchmark culling
    Camera camera;
    camera.set_position(Eigen::Vector3f(0, 0, 200));
    camera.look_at(Eigen::Vector3f(0, 0, 0));
    
    constexpr int num_cull_tests = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_cull_tests; ++i) {
        // Vary camera position slightly
        Eigen::Vector3f pos = camera.get_position() + Eigen::Vector3f(i * 0.1f, i * 0.1f, 0);
        camera.set_position(pos);
        
        auto cull_results = quadtree.cull_frustum(camera.get_frustum_planes());
        
        // Verify some objects are culled
        EXPECT_GT(cull_results.visible_objects.size(), 0);
        EXPECT_LT(cull_results.visible_objects.size(), num_objects);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    double cull_rate = static_cast<double>(num_cull_tests) / duration.count() * 1e6; // culls/sec
    
    std::cout << "Culling performance:" << std::endl;
    std::cout << "  Objects: " << num_objects << std::endl;
    std::cout << "  Cull tests: " << num_cull_tests << std::endl;
    std::cout << "  Cull rate: " << cull_rate << " culls/sec" << std::endl;
    
    // Performance assertion
    EXPECT_GT(cull_rate, 1000); // At least 1K culls/sec for 50K objects
}

// Main test runner
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    
    std::cout << "Running Vulkan-Forge Memory Management Tests" << std::endl;
    
    return RUN_ALL_TESTS();
}