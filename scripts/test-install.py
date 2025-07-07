#!/usr/bin/env python3
"""
============================================================================
Post-Install Validation Script for vulkan-forge
============================================================================
Comprehensive testing script to validate vulkan-forge installation
Tests functionality, performance, memory usage, and Vulkan integration
Suitable for CI/CD pipelines and developer verification
============================================================================
"""

import sys
import os
import time
import platform
import subprocess
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

# ============================================================================
# Configuration and Constants
# ============================================================================

# Performance targets from roadmap
PERFORMANCE_TARGETS = {
    "triangle_fps": 1000,           # M-0: Triangle at 1000 FPS  
    "scene_builds_per_sec": 100,   # Scene building performance
    "large_scene_indices": 10000,  # Large scene handling
    "memory_growth_mb": 100,       # Maximum memory growth
    "import_time_ms": 1000,        # Module import time
}

# Test configuration
DEFAULT_TEST_ITERATIONS = {
    "performance": 100,
    "memory": 50,
    "stress": 1000,
}

SUPPORTED_NUMPY_DTYPES = [
    "float32", "float64", "int32", "int64"
]

# Color codes for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def disable(cls):
        """Disable colors for CI environments"""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.CYAN = cls.PURPLE = cls.NC = ''

# ============================================================================
# Test Result Data Structures
# ============================================================================

@dataclass
class TestResult:
    """Individual test result"""
    name: str
    passed: bool
    duration: float
    message: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class TestSuite:
    """Collection of test results"""
    name: str
    results: List[TestResult]
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return (self.passed_count / len(self.results)) * 100

@dataclass
class SystemInfo:
    """System information for context"""
    platform: str
    python_version: str
    architecture: str
    processor: str
    memory_gb: float
    vulkan_sdk: Optional[str] = None
    gpu_info: Optional[str] = None

# ============================================================================
# Utility Functions
# ============================================================================

def log(message: str, color: str = Colors.NC):
    """Log message with timestamp and color"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Colors.CYAN}[{timestamp}]{Colors.NC} {color}{message}{Colors.NC}")

def log_success(message: str):
    """Log success message"""
    log(f"✅ {message}", Colors.GREEN)

def log_warning(message: str):
    """Log warning message"""
    log(f"⚠️  {message}", Colors.YELLOW)

def log_error(message: str):
    """Log error message"""
    log(f"❌ {message}", Colors.RED)

def log_info(message: str):
    """Log info message"""
    log(f"ℹ️  {message}", Colors.BLUE)

@contextmanager
def timer():
    """Context manager for timing operations"""
    start = time.time()
    yield lambda: time.time() - start
    
def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.1f}m"

def get_system_info() -> SystemInfo:
    """Collect system information"""
    import psutil
    
    # Get memory in GB
    memory_bytes = psutil.virtual_memory().total
    memory_gb = memory_bytes / (1024**3)
    
    # Get processor info
    try:
        if platform.system() == "Windows":
            processor = platform.processor()
        else:
            processor = subprocess.check_output(
                ["cat", "/proc/cpuinfo"], 
                universal_newlines=True
            ).split('\n')[4].split(':')[1].strip()
    except:
        processor = "Unknown"
    
    # Check for Vulkan SDK
    vulkan_sdk = os.environ.get('VULKAN_SDK', 'Not set')
    
    return SystemInfo(
        platform=f"{platform.system()} {platform.release()}",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        architecture=platform.machine(),
        processor=processor,
        memory_gb=memory_gb,
        vulkan_sdk=vulkan_sdk
    )

def check_dependencies():
    """Check if required dependencies are available"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    
    if missing:
        log_error(f"Missing required dependencies: {', '.join(missing)}")
        log_info("Install with: pip install numpy psutil")
        return False
    
    return True

# ============================================================================
# Test Classes
# ============================================================================

class VulkanForgeValidator:
    """Main validation class for vulkan-forge"""
    
    def __init__(self, args):
        self.args = args
        self.system_info = get_system_info()
        self.test_suites: List[TestSuite] = []
        self.vulkan_forge = None
        self.numpy = None
        
    def run_all_tests(self) -> bool:
        """Run all validation tests"""
        log("🚀 Starting vulkan-forge post-install validation")
        log(f"Platform: {self.system_info.platform}")
        log(f"Python: {self.system_info.python_version}")
        log(f"Architecture: {self.system_info.architecture}")
        print()
        
        # Check dependencies first
        if not check_dependencies():
            return False
        
        # Import required modules
        try:
            import numpy as np
            import psutil
            self.numpy = np
        except ImportError as e:
            log_error(f"Failed to import dependencies: {e}")
            return False
        
        # Run test suites
        test_suites = [
            ("Import Tests", self.test_imports),
            ("Basic Functionality", self.test_basic_functionality), 
            ("Performance Tests", self.test_performance),
            ("Memory Tests", self.test_memory_handling),
            ("Vulkan Integration", self.test_vulkan_integration),
            ("Stress Tests", self.test_stress) if self.args.stress else None,
        ]
        
        # Filter out None entries
        test_suites = [suite for suite in test_suites if suite is not None]
        
        all_passed = True
        for suite_name, test_func in test_suites:
            success = self.run_test_suite(suite_name, test_func)
            if not success:
                all_passed = False
                if self.args.fail_fast:
                    break
        
        # Generate report
        self.generate_report()
        
        return all_passed
    
    def run_test_suite(self, name: str, test_func) -> bool:
        """Run a test suite and record results"""
        log(f"🧪 Running {name}...")
        
        start_time = time.time()
        results = []
        
        try:
            test_results = test_func()
            if isinstance(test_results, list):
                results.extend(test_results)
            else:
                results.append(test_results)
        except Exception as e:
            log_error(f"Test suite {name} crashed: {e}")
            if self.args.verbose:
                traceback.print_exc()
            results.append(TestResult(
                name=f"{name} (crashed)",
                passed=False,
                duration=0.0,
                message=str(e)
            ))
        
        end_time = time.time()
        
        # Create test suite
        suite = TestSuite(
            name=name,
            results=results,
            start_time=start_time,
            end_time=end_time
        )
        self.test_suites.append(suite)
        
        # Log results
        passed = suite.passed_count
        total = len(suite.results)
        duration = format_duration(suite.duration)
        
        if suite.failed_count == 0:
            log_success(f"{name}: {passed}/{total} tests passed ({duration})")
        else:
            log_warning(f"{name}: {passed}/{total} tests passed ({duration})")
            
            # Show failed tests
            for result in suite.results:
                if not result.passed:
                    log_error(f"  - {result.name}: {result.message}")
        
        print()
        return suite.failed_count == 0
    
    def test_imports(self) -> List[TestResult]:
        """Test module imports"""
        results = []
        
        # Test vulkan-forge import
        with timer() as get_time:
            try:
                import vulkan_forge as vf
                self.vulkan_forge = vf
                import_time = get_time() * 1000  # Convert to ms
                
                if import_time > PERFORMANCE_TARGETS["import_time_ms"]:
                    results.append(TestResult(
                        name="vulkan_forge import speed",
                        passed=False,
                        duration=import_time/1000,
                        message=f"Import took {import_time:.1f}ms (target: {PERFORMANCE_TARGETS['import_time_ms']}ms)",
                        details={"import_time_ms": import_time}
                    ))
                else:
                    results.append(TestResult(
                        name="vulkan_forge import",
                        passed=True,
                        duration=import_time/1000,
                        message=f"Imported in {import_time:.1f}ms",
                        details={"import_time_ms": import_time}
                    ))
                    
            except ImportError as e:
                results.append(TestResult(
                    name="vulkan_forge import",
                    passed=False,
                    duration=get_time(),
                    message=f"Import failed: {e}"
                ))
                return results
        
        # Test version access
        try:
            version = self.vulkan_forge.__version__
            results.append(TestResult(
                name="version attribute",
                passed=True,
                duration=0.0,
                message=f"Version: {version}",
                details={"version": version}
            ))
        except AttributeError:
            results.append(TestResult(
                name="version attribute",
                passed=False,
                duration=0.0,
                message="__version__ attribute not found"
            ))
        
        # Test required classes
        required_classes = ["HeightFieldScene", "Renderer"]
        for class_name in required_classes:
            if hasattr(self.vulkan_forge, class_name):
                results.append(TestResult(
                    name=f"{class_name} class",
                    passed=True,
                    duration=0.0,
                    message=f"{class_name} available"
                ))
            else:
                results.append(TestResult(
                    name=f"{class_name} class",
                    passed=False,
                    duration=0.0,
                    message=f"{class_name} not found"
                ))
        
        return results
    
    def test_basic_functionality(self) -> List[TestResult]:
        """Test basic vulkan-forge functionality"""
        if not self.vulkan_forge:
            return [TestResult("basic_functionality", False, 0.0, "vulkan_forge not imported")]
        
        results = []
        
        # Check if HeightFieldScene class exists and is callable
        if not hasattr(self.vulkan_forge, 'HeightFieldScene'):
            results.append(TestResult(
                name="HeightFieldScene class availability",
                passed=False,
                duration=0.0,
                message="HeightFieldScene class not found in module"
            ))
            return results
        
        scene_class = getattr(self.vulkan_forge, 'HeightFieldScene')
        if not callable(scene_class):
            results.append(TestResult(
                name="HeightFieldScene class callable",
                passed=False,
                duration=0.0,
                message=f"HeightFieldScene is not callable: {type(scene_class)}"
            ))
            return results
        
        # Test scene creation
        with timer() as get_time:
            try:
                scene = scene_class()
                if scene is None:
                    results.append(TestResult(
                        name="HeightFieldScene creation",
                        passed=False,
                        duration=get_time(),
                        message="HeightFieldScene constructor returned None"
                    ))
                    return results
                
                results.append(TestResult(
                    name="HeightFieldScene creation",
                    passed=True,
                    duration=get_time(),
                    message=f"Scene created successfully: {type(scene)}"
                ))
            except Exception as e:
                results.append(TestResult(
                    name="HeightFieldScene creation",
                    passed=False,
                    duration=get_time(),
                    message=f"Scene creation failed: {e}"
                ))
                return results
        
        # Check if scene has build method
        if not hasattr(scene, 'build'):
            results.append(TestResult(
                name="Scene build method availability",
                passed=False,
                duration=0.0,
                message="Scene object has no 'build' method"
            ))
            return results
        
        build_method = getattr(scene, 'build')
        if not callable(build_method):
            results.append(TestResult(
                name="Scene build method callable",
                passed=False,
                duration=0.0,
                message=f"Scene.build is not callable: {type(build_method)}"
            ))
            return results
        
        # Test scene building with different data types
        for dtype_name in SUPPORTED_NUMPY_DTYPES:
            try:
                dtype = getattr(self.numpy, dtype_name)
                heights = self.numpy.ones((8, 8), dtype=dtype)
                
                with timer() as get_time:
                    scene = scene_class()
                    if scene is None:
                        results.append(TestResult(
                            name=f"Scene building ({dtype_name})",
                            passed=False,
                            duration=get_time(),
                            message="Scene constructor returned None"
                        ))
                        continue
                    
                    scene.build(heights, zscale=1.0)
                    
                    # Check if scene has n_indices attribute
                    if hasattr(scene, 'n_indices'):
                        indices = scene.n_indices
                    else:
                        indices = "unknown"
                    
                results.append(TestResult(
                    name=f"Scene building ({dtype_name})",
                    passed=True,
                    duration=get_time(),
                    message=f"Built scene with {indices} indices",
                    details={
                        "dtype": dtype_name,
                        "indices": indices if isinstance(indices, int) else -1,
                        "shape": heights.shape
                    }
                ))
                
            except Exception as e:
                results.append(TestResult(
                    name=f"Scene building ({dtype_name})",
                    passed=False,
                    duration=0.0,
                    message=f"Failed with {dtype_name}: {e}"
                ))
        
        # Check if Renderer class exists and is callable
        if not hasattr(self.vulkan_forge, 'Renderer'):
            results.append(TestResult(
                name="Renderer class availability",
                passed=False,
                duration=0.0,
                message="Renderer class not found in module"
            ))
            return results
        
        renderer_class = getattr(self.vulkan_forge, 'Renderer')
        if not callable(renderer_class):
            results.append(TestResult(
                name="Renderer class callable",
                passed=False,
                duration=0.0,
                message=f"Renderer is not callable: {type(renderer_class)}"
            ))
            return results
        
        # Test renderer creation
        with timer() as get_time:
            try:
                renderer = renderer_class(64, 64)
                if renderer is None:
                    results.append(TestResult(
                        name="Renderer creation",
                        passed=False,
                        duration=get_time(),
                        message="Renderer constructor returned None"
                    ))
                else:
                    results.append(TestResult(
                        name="Renderer creation",
                        passed=True,
                        duration=get_time(),
                        message=f"Renderer created successfully: {type(renderer)}"
                    ))
            except Exception as e:
                results.append(TestResult(
                    name="Renderer creation", 
                    passed=False,
                    duration=get_time(),
                    message=f"Renderer creation failed: {e}"
                ))
        
        # Test different scene sizes
        scene_sizes = [(4, 4), (16, 16), (32, 32), (64, 64)]
        for width, height in scene_sizes:
            try:
                heights = self.numpy.random.rand(height, width).astype(self.numpy.float32)
                scene = scene_class()
                
                if scene is None:
                    results.append(TestResult(
                        name=f"Scene size {width}x{height}",
                        passed=False,
                        duration=0.0,
                        message="Scene constructor returned None"
                    ))
                    continue
                
                with timer() as get_time:
                    scene.build(heights, zscale=2.0)
                    
                    # Check if scene has n_indices attribute
                    if hasattr(scene, 'n_indices'):
                        indices = scene.n_indices
                    else:
                        indices = "unknown"
                    
                results.append(TestResult(
                    name=f"Scene size {width}x{height}",
                    passed=True,
                    duration=get_time(),
                    message=f"{indices} indices",
                    details={
                        "scene_size": [width, height],
                        "indices": indices if isinstance(indices, int) else -1
                    }
                ))
                
            except Exception as e:
                results.append(TestResult(
                    name=f"Scene size {width}x{height}",
                    passed=False,
                    duration=0.0,
                    message=f"Failed: {e}"
                ))
        
        return results
    
    def test_performance(self) -> List[TestResult]:
        """Test performance against roadmap targets"""
        if not self.vulkan_forge:
            return [TestResult("performance", False, 0.0, "vulkan_forge not imported")]
        
        results = []
        iterations = self.args.iterations or DEFAULT_TEST_ITERATIONS["performance"]
        
        # Check if required classes exist
        if not hasattr(self.vulkan_forge, 'HeightFieldScene'):
            return [TestResult("performance", False, 0.0, "HeightFieldScene class not found")]
        
        if not hasattr(self.vulkan_forge, 'Renderer'):
            return [TestResult("performance", False, 0.0, "Renderer class not found")]
        
        scene_class = getattr(self.vulkan_forge, 'HeightFieldScene')
        renderer_class = getattr(self.vulkan_forge, 'Renderer')
        
        if not callable(scene_class) or not callable(renderer_class):
            return [TestResult("performance", False, 0.0, "Required classes are not callable")]
        
        # Test scene building performance
        log_info(f"Testing scene building performance ({iterations} iterations)...")
        
        heights = self.numpy.ones((32, 32), dtype=self.numpy.float32)
        
        with timer() as get_time:
            scene_build_times = []
            failed_builds = 0
            
            for i in range(iterations):
                try:
                    scene = scene_class()
                    if scene is None:
                        failed_builds += 1
                        continue
                    
                    build_start = time.time()
                    scene.build(heights, zscale=1.0)
                    build_time = time.time() - build_start
                    
                    scene_build_times.append(build_time)
                except Exception as e:
                    failed_builds += 1
                    if i == 0:  # Log first failure
                        log_warning(f"Scene building failed: {e}")
        
        if not scene_build_times:
            results.append(TestResult(
                name="Scene building performance",
                passed=False,
                duration=get_time(),
                message="All scene building attempts failed",
                details={"failed_builds": failed_builds, "iterations": iterations}
            ))
        else:
            total_time = get_time()
            avg_build_time = sum(scene_build_times) / len(scene_build_times)
            builds_per_sec = 1.0 / avg_build_time
            
            target_fps = PERFORMANCE_TARGETS["scene_builds_per_sec"]
            passed = builds_per_sec >= target_fps
            
            results.append(TestResult(
                name="Scene building performance",
                passed=passed,
                duration=total_time,
                message=f"{builds_per_sec:.1f} builds/sec (target: {target_fps}), {failed_builds} failures",
                details={
                    "builds_per_sec": builds_per_sec,
                    "avg_build_time_ms": avg_build_time * 1000,
                    "successful_builds": len(scene_build_times),
                    "failed_builds": failed_builds,
                    "total_iterations": iterations,
                    "target_fps": target_fps
                }
            ))
        
        # Test rendering performance (if GPU available)
        try:
            log_info("Testing rendering performance...")
            
            heights = self.numpy.ones((16, 16), dtype=self.numpy.float32)
            scene = self.vulkan_forge.HeightFieldScene()
            scene.build(heights)
            
            renderer = self.vulkan_forge.Renderer(64, 64)
            
            # Warmup renders
            for _ in range(5):
                try:
                    renderer.render(scene)
                except:
                    break
            
            # Timed renders
            render_times = []
            successful_renders = 0
            
            with timer() as get_time:
                for i in range(min(50, iterations)):
                    try:
                        render_start = time.time()
                        img = renderer.render(scene)
                        render_time = time.time() - render_start
                        
                        render_times.append(render_time)
                        successful_renders += 1
                        
                        # Validate image
                        if img.shape != (64, 64, 4) or img.dtype != self.numpy.uint8:
                            raise ValueError(f"Invalid image format: {img.shape}, {img.dtype}")
                            
                    except Exception as e:
                        if i == 0:  # First render failed
                            results.append(TestResult(
                                name="Rendering capability",
                                passed=False,
                                duration=0.0,
                                message=f"Rendering failed (expected in headless environments): {e}"
                            ))
                            break
            
            if successful_renders > 0:
                total_render_time = get_time()
                avg_render_time = sum(render_times) / len(render_times)
                fps = 1.0 / avg_render_time
                
                target_fps = PERFORMANCE_TARGETS["triangle_fps"]
                # Lower expectation for actual rendering vs simple triangle
                adjusted_target = target_fps / 10  # 100 FPS for complex scenes
                passed = fps >= adjusted_target
                
                results.append(TestResult(
                    name="Rendering performance", 
                    passed=passed,
                    duration=total_render_time,
                    message=f"{fps:.1f} FPS (target: {adjusted_target})",
                    details={
                        "fps": fps,
                        "avg_render_time_ms": avg_render_time * 1000,
                        "successful_renders": successful_renders,
                        "target_fps": adjusted_target
                    }
                ))
            
        except Exception as e:
            results.append(TestResult(
                name="Rendering performance",
                passed=False,
                duration=0.0,
                message=f"Rendering test failed: {e}"
            ))
        
        return results
    
    def test_memory_handling(self) -> List[TestResult]:
        """Test memory usage and leak detection"""
        if not self.vulkan_forge:
            return [TestResult("memory", False, 0.0, "vulkan_forge not imported")]
        
        import psutil
        import gc
        
        results = []
        process = psutil.Process()
        iterations = self.args.iterations or DEFAULT_TEST_ITERATIONS["memory"]
        
        # Check if required classes exist and are callable
        if not hasattr(self.vulkan_forge, 'HeightFieldScene') or not hasattr(self.vulkan_forge, 'Renderer'):
            return [TestResult("memory", False, 0.0, "Required classes not found")]
        
        scene_class = getattr(self.vulkan_forge, 'HeightFieldScene')
        renderer_class = getattr(self.vulkan_forge, 'Renderer')
        
        if not callable(scene_class) or not callable(renderer_class):
            return [TestResult("memory", False, 0.0, "Required classes are not callable")]
        
        # Record initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        log_info(f"Testing memory handling ({iterations} iterations)...")
        log_info(f"Initial memory: {initial_memory:.1f} MB")
        
        # Memory leak test
        failed_iterations = 0
        with timer() as get_time:
            for i in range(iterations):
                try:
                    # Create large scene
                    heights = self.numpy.random.rand(128, 128).astype(self.numpy.float32)
                    scene = scene_class()
                    
                    if scene is None:
                        failed_iterations += 1
                        continue
                    
                    scene.build(heights, zscale=2.0)
                    
                    # Create renderer
                    renderer = renderer_class(64, 64)
                    
                    if renderer is None:
                        failed_iterations += 1
                        continue
                    
                    # Force garbage collection every 10 iterations
                    if i % 10 == 0:
                        gc.collect()
                        current_memory = process.memory_info().rss / 1024 / 1024
                        if self.args.verbose:
                            log_info(f"  Iteration {i}: {current_memory:.1f} MB")
                            
                except Exception as e:
                    failed_iterations += 1
                    if i == 0:  # Log first failure
                        log_warning(f"Memory test iteration failed: {e}")
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        total_time = get_time()
        
        target_growth = PERFORMANCE_TARGETS["memory_growth_mb"]
        passed = memory_growth <= target_growth and failed_iterations < iterations * 0.1  # Allow 10% failures
        
        results.append(TestResult(
            name="Memory leak test",
            passed=passed,
            duration=total_time,
            message=f"Memory growth: {memory_growth:.1f} MB (limit: {target_growth} MB), {failed_iterations} failures",
            details={
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_growth_mb": memory_growth,
                "target_growth_mb": target_growth,
                "iterations": iterations,
                "failed_iterations": failed_iterations
            }
        ))
        
        # Large scene test
        try:
            log_info("Testing large scene handling...")
            
            large_heights = self.numpy.random.rand(512, 512).astype(self.numpy.float32)
            large_scene = scene_class()
            
            if large_scene is None:
                results.append(TestResult(
                    name="Large scene handling",
                    passed=False,
                    duration=0.0,
                    message="Cannot create large scene - constructor returned None"
                ))
            else:
                with timer() as get_time:
                    large_scene.build(large_heights)
                
                # Check if scene has n_indices attribute
                if hasattr(large_scene, 'n_indices'):
                    indices = large_scene.n_indices
                    target_indices = PERFORMANCE_TARGETS["large_scene_indices"]
                    passed = indices >= target_indices
                else:
                    indices = "unknown"
                    passed = True  # If we can't check indices, at least building succeeded
                
                results.append(TestResult(
                    name="Large scene handling",
                    passed=passed,
                    duration=get_time(),
                    message=f"{indices} indices (target: {PERFORMANCE_TARGETS['large_scene_indices']}+)",
                    details={
                        "scene_size": [512, 512],
                        "indices": indices if isinstance(indices, int) else -1,
                        "target_indices": PERFORMANCE_TARGETS["large_scene_indices"]
                    }
                ))
            
        except Exception as e:
            results.append(TestResult(
                name="Large scene handling",
                passed=False,
                duration=0.0,
                message=f"Large scene test failed: {e}"
            ))
        
        return results
    
    def test_vulkan_integration(self) -> List[TestResult]:
        """Test Vulkan SDK integration"""
        results = []
        
        # Check Vulkan SDK environment
        vulkan_sdk = os.environ.get('VULKAN_SDK')
        if vulkan_sdk:
            results.append(TestResult(
                name="VULKAN_SDK environment variable",
                passed=True,
                duration=0.0,
                message=f"Set to: {vulkan_sdk}",
                details={"vulkan_sdk_path": vulkan_sdk}
            ))
        else:
            results.append(TestResult(
                name="VULKAN_SDK environment variable",
                passed=False,
                duration=0.0,
                message="VULKAN_SDK not set"
            ))
        
        # Check for vulkaninfo
        vulkaninfo_found = False
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                vulkaninfo_found = True
                results.append(TestResult(
                    name="vulkaninfo execution",
                    passed=True,
                    duration=0.0,
                    message="vulkaninfo executed successfully"
                ))
                
                # Parse GPU information
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'deviceName' in line or 'GPU' in line:
                        self.system_info.gpu_info = line.strip()
                        break
                        
            else:
                results.append(TestResult(
                    name="vulkaninfo execution",
                    passed=False,
                    duration=0.0,
                    message=f"vulkaninfo failed: {result.stderr}"
                ))
                
        except subprocess.TimeoutExpired:
            results.append(TestResult(
                name="vulkaninfo execution",
                passed=False,
                duration=0.0,
                message="vulkaninfo timed out"
            ))
        except FileNotFoundError:
            results.append(TestResult(
                name="vulkaninfo execution",
                passed=False,
                duration=0.0,
                message="vulkaninfo not found in PATH"
            ))
        except Exception as e:
            results.append(TestResult(
                name="vulkaninfo execution",
                passed=False,
                duration=0.0,
                message=f"vulkaninfo error: {e}"
            ))
        
        # Check for glslc (shader compiler)
        try:
            result = subprocess.run(
                ["glslc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                results.append(TestResult(
                    name="glslc availability",
                    passed=True,
                    duration=0.0,
                    message="Shader compiler available"
                ))
            else:
                results.append(TestResult(
                    name="glslc availability",
                    passed=False,
                    duration=0.0,
                    message="glslc execution failed"
                ))
                
        except FileNotFoundError:
            results.append(TestResult(
                name="glslc availability",
                passed=False,
                duration=0.0,
                message="glslc not found (shader compilation may fail)"
            ))
        except Exception as e:
            results.append(TestResult(
                name="glslc availability",
                passed=False,
                duration=0.0,
                message=f"glslc check failed: {e}"
            ))
        
        return results
    
    def test_stress(self) -> List[TestResult]:
        """Stress test the installation"""
        if not self.vulkan_forge:
            return [TestResult("stress", False, 0.0, "vulkan_forge not imported")]
        
        results = []
        iterations = self.args.iterations or DEFAULT_TEST_ITERATIONS["stress"]
        
        # Check if required classes exist and are callable
        if not hasattr(self.vulkan_forge, 'HeightFieldScene') or not hasattr(self.vulkan_forge, 'Renderer'):
            return [TestResult("stress", False, 0.0, "Required classes not found")]
        
        scene_class = getattr(self.vulkan_forge, 'HeightFieldScene')
        renderer_class = getattr(self.vulkan_forge, 'Renderer')
        
        if not callable(scene_class) or not callable(renderer_class):
            return [TestResult("stress", False, 0.0, "Required classes are not callable")]
        
        log_info(f"Running stress test ({iterations} iterations)...")
        
        with timer() as get_time:
            failed_operations = 0
            null_objects = 0
            
            for i in range(iterations):
                try:
                    # Random scene size
                    size = self.numpy.random.randint(8, 128)
                    heights = self.numpy.random.rand(size, size).astype(self.numpy.float32)
                    
                    # Create and build scene
                    scene = scene_class()
                    if scene is None:
                        null_objects += 1
                        failed_operations += 1
                        continue
                    
                    scene.build(heights, zscale=self.numpy.random.uniform(0.5, 5.0))
                    
                    # Create renderer with random size
                    render_size = self.numpy.random.randint(32, 256)
                    renderer = renderer_class(render_size, render_size)
                    
                    if renderer is None:
                        null_objects += 1
                        failed_operations += 1
                        continue
                    
                    if i % 100 == 0 and self.args.verbose:
                        log_info(f"  Stress test iteration {i}/{iterations}")
                        
                except Exception as e:
                    failed_operations += 1
                    if self.args.verbose and i < 10:  # Only log first 10 failures
                        log_warning(f"  Stress test iteration {i} failed: {e}")
        
        total_time = get_time()
        success_rate = ((iterations - failed_operations) / iterations) * 100
        passed = success_rate >= 95.0  # 95% success rate required
        
        message_parts = [f"{success_rate:.1f}% success rate ({failed_operations}/{iterations} failures)"]
        if null_objects > 0:
            message_parts.append(f"{null_objects} null objects")
        
        results.append(TestResult(
            name="Stress test",
            passed=passed,
            duration=total_time,
            message=", ".join(message_parts),
            details={
                "iterations": iterations,
                "failed_operations": failed_operations,
                "null_objects": null_objects,
                "success_rate": success_rate,
                "operations_per_sec": iterations / total_time
            }
        ))
        
        return results
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("=" * 80)
        print(f"{Colors.PURPLE}VULKAN-FORGE INSTALLATION VALIDATION REPORT{Colors.NC}")
        print("=" * 80)
        
        # System information
        print(f"\n{Colors.CYAN}System Information:{Colors.NC}")
        print(f"  Platform: {self.system_info.platform}")
        print(f"  Python: {self.system_info.python_version}")
        print(f"  Architecture: {self.system_info.architecture}")
        print(f"  Processor: {self.system_info.processor}")
        print(f"  Memory: {self.system_info.memory_gb:.1f} GB")
        print(f"  Vulkan SDK: {self.system_info.vulkan_sdk}")
        if self.system_info.gpu_info:
            print(f"  GPU: {self.system_info.gpu_info}")
        
        # Test results summary
        total_tests = sum(len(suite.results) for suite in self.test_suites)
        total_passed = sum(suite.passed_count for suite in self.test_suites)
        total_duration = sum(suite.duration for suite in self.test_suites)
        
        print(f"\n{Colors.CYAN}Test Summary:{Colors.NC}")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_tests - total_passed}")
        print(f"  Success Rate: {(total_passed/total_tests)*100:.1f}%")
        print(f"  Total Duration: {format_duration(total_duration)}")
        
        # Suite breakdown
        print(f"\n{Colors.CYAN}Test Suite Breakdown:{Colors.NC}")
        for suite in self.test_suites:
            status = "✅ PASS" if suite.failed_count == 0 else "❌ FAIL"
            print(f"  {suite.name}: {status} ({suite.passed_count}/{len(suite.results)}) - {format_duration(suite.duration)}")
        
        # Performance metrics
        print(f"\n{Colors.CYAN}Performance Metrics:{Colors.NC}")
        for suite in self.test_suites:
            for result in suite.results:
                if result.details and any(key.endswith('_fps') or key.endswith('_per_sec') for key in result.details.keys()):
                    for key, value in result.details.items():
                        if key.endswith('_fps') or key.endswith('_per_sec'):
                            print(f"  {result.name}: {value:.1f} {key.replace('_', ' ')}")
        
        # Save detailed report if requested
        if self.args.output:
            self.save_json_report()
    
    def save_json_report(self):
        """Save detailed JSON report"""
        report_data = {
            "system_info": asdict(self.system_info),
            "test_suites": [
                {
                    "name": suite.name,
                    "duration": suite.duration,
                    "passed_count": suite.passed_count,
                    "failed_count": suite.failed_count,
                    "success_rate": suite.success_rate,
                    "results": [asdict(result) for result in suite.results]
                }
                for suite in self.test_suites
            ],
            "summary": {
                "total_tests": sum(len(suite.results) for suite in self.test_suites),
                "total_passed": sum(suite.passed_count for suite in self.test_suites),
                "total_duration": sum(suite.duration for suite in self.test_suites),
                "overall_success": all(suite.failed_count == 0 for suite in self.test_suites)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(self.args)
        }
        
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        log_info(f"Detailed report saved to: {output_path}")

# ============================================================================
# Command Line Interface
# ============================================================================

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Post-install validation script for vulkan-forge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test-install.py                    # Basic validation
  python test-install.py --verbose         # Verbose output
  python test-install.py --stress          # Include stress tests
  python test-install.py --iterations 200  # Custom iteration count
  python test-install.py --output report.json --no-color  # CI-friendly
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--stress',
        action='store_true',
        help='Include stress testing (takes longer)'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        help='Number of iterations for performance tests'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Save detailed JSON report to file'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output (useful for CI)'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first test suite failure'
    )
    
    parser.add_argument(
        '--performance-only',
        action='store_true',
        help='Run only performance tests'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick validation (reduced iterations)'
    )
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Disable colors for CI or if requested
    if args.no_color or os.environ.get('CI') or not sys.stdout.isatty():
        Colors.disable()
    
    # Adjust iterations for quick mode
    if args.quick and not args.iterations:
        args.iterations = 10
    
    # Run validation
    validator = VulkanForgeValidator(args)
    
    try:
        success = validator.run_all_tests()
        
        if success:
            log_success("🎉 All tests passed! vulkan-forge installation is working correctly.")
            sys.exit(0)
        else:
            log_error("💥 Some tests failed. Please check the output above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        log_warning("⚠️  Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_error(f"💥 Testing crashed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()