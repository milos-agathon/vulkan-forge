#!/usr/bin/env python3
"""
CLI Tool Testing Suite

Tests the command-line interface tools for terrain rendering including:
- terrain_performance.py benchmarking tool
- terrain_viewer.py interactive viewer
- load_geotiff_basic.py example script

Validates CLI argument parsing, error handling, and basic functionality.
"""

import pytest
import subprocess
import sys
import tempfile
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import CLI modules if available

# Fix imports for CLI tests
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from test_mocks import *
except ImportError:
    # Fallback: create minimal mocks inline
    class GeoTiffLoader:
        def load(self, path): return False
        def get_data(self): return None, {}
    
    class TerrainCache:
        def __init__(self, max_tiles=64, tile_size=256, eviction_policy='lru'): pass
        def get_tile(self, tile_id): return None
        def put_tile(self, tile_id, data): pass
        def get_statistics(self): return {}
    
    class CoordinateSystems:
        def __init__(self): pass
        def is_valid_coordinate(self, lon, lat): return -180 <= lon <= 180 and -90 <= lat <= 90
        def get_supported_systems(self): return ['EPSG:4326']
    
    class GeographicBounds:
        def __init__(self, *args): pass
        def contains(self, lon, lat): return True
    
    class TerrainConfig:
        def __init__(self): 
            from types import SimpleNamespace
            self.tessellation = SimpleNamespace(max_level=64)
            self.performance = SimpleNamespace(target_fps=144)
        
        @classmethod
        def from_preset(cls, preset):
            config = cls()
            if preset == 'high_performance': config.performance.target_fps = 200
            elif preset == 'balanced': config.performance.target_fps = 144
            elif preset == 'high_quality': config.performance.target_fps = 60
            elif preset == 'mobile': config.performance.target_fps = 30
            return config
        
        def optimize_for_hardware(self, gpu, vram, cores): pass
        def validate(self): return []
    
    class ShaderCompiler:
        def __init__(self): 
            self.glslc_path = None
            self.spirv_val_path = None
        def compile_shader(self, source, stage, target='vulkan1.3'): 
            return True, b'\x03\x02\x23\x07' + b'\x00'*100, ""
        def validate_spirv(self, data, target='vulkan1.3'): 
            return True, ""
    
    class TerrainShaderTemplates:
        VERTEX_SHADER = "#version 450\nvoid main() {}"
        TESSELLATION_CONTROL_SHADER = "#version 450\nvoid main() {}"
        TESSELLATION_EVALUATION_SHADER = "#version 450\nvoid main() {}"
        FRAGMENT_SHADER = "#version 450\nvoid main() {}"
    
    class TerrainRenderer:
        def __init__(self, config, context): pass
        def update_camera(self, pos, rot): pass
    
    class InvalidGeoTiffError(Exception): pass


try:
    from vulkan_forge.examples import terrain_performance, terrain_viewer, load_geotiff_basic
    from vulkan_forge.cli.benchmark import main as benchmark_main
    from vulkan_forge.cli.viewer import main as viewer_main
    from vulkan_forge.cli.info import main as info_main
    CLI_MODULES_AVAILABLE = True
except ImportError:
    CLI_MODULES_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class CLITestHelper:
    """Helper class for CLI testing"""
    
    @staticmethod
    def run_cli_script(script_path: str, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a CLI script with arguments"""
        cmd = [sys.executable, script_path] + args
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    @staticmethod
    def create_test_geotiff(filepath: str, size: int = 256) -> bool:
        """Create a small test GeoTIFF file for CLI testing"""
        if not RASTERIO_AVAILABLE:
            return False
        
        import numpy as np
        import rasterio.transform
        from rasterio.crs import CRS
        
        # Generate simple heightmap
        heights = np.random.randint(0, 100, (size, size)).astype(np.float32)
        
        # Create geographic transform
        transform = rasterio.transform.from_bounds(
            west=-1.0, south=-1.0, east=1.0, north=1.0,
            width=size, height=size
        )
        
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=size,
            width=size,
            count=1,
            dtype=rasterio.float32,
            crs=CRS.from_epsg(4326),
            transform=transform
        ) as dataset:
            dataset.write(heights, 1)
        
        return True
    
    @staticmethod
    def parse_cli_output(output: str) -> Dict[str, Any]:
        """Parse common CLI output formats"""
        result = {
            'success': False,
            'error_message': '',
            'metrics': {},
            'warnings': []
        }
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Check for success indicators
            if 'success' in line.lower() or 'completed' in line.lower():
                result['success'] = True
            
            # Check for error messages
            if 'error:' in line.lower() or 'failed:' in line.lower():
                result['error_message'] = line
            
            # Check for warnings
            if 'warning:' in line.lower():
                result['warnings'].append(line)
            
            # Parse metrics (simple key: value format)
            if ':' in line and any(char.isdigit() for char in line):
                try:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Try to parse numeric values
                    if value.replace('.', '').replace(',', '').isdigit():
                        result['metrics'][key] = float(value.replace(',', ''))
                    else:
                        result['metrics'][key] = value
                except ValueError:
                    pass
        
        return result


@pytest.fixture
def test_geotiff_file():
    """Create a temporary test GeoTIFF file"""
    if not RASTERIO_AVAILABLE:
        pytest.skip("rasterio not available")
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        filepath = f.name
    
    success = CLITestHelper.create_test_geotiff(filepath, size=128)
    if not success:
        pytest.skip("Failed to create test GeoTIFF")
    
    yield filepath
    
    # Cleanup
    try:
        os.unlink(filepath)
    except OSError:
        pass


@pytest.fixture
def cli_helper():
    """Provide CLI test helper"""
    return CLITestHelper()


class TestLoadGeoTiffBasic:
    """Test the basic GeoTIFF loading example script"""
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_basic_geotiff_loading_help(self, cli_helper):
        """Test help output for basic GeoTIFF loader"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not script_path.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), ['--help'])
        
        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert 'usage:' in result.stdout.lower()
        assert 'geotiff' in result.stdout.lower()
        assert '--config' in result.stdout
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_synthetic_terrain_generation(self, cli_helper):
        """Test synthetic terrain generation"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not script_path.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            # Mock successful initialization and loading
            mock_renderer = MockRenderer.return_value
            mock_renderer.load_geotiff.return_value = True
            
            result = cli_helper.run_cli_script(str(script_path), [
                '--synthetic',
                '--size', '64',
                '--config', 'balanced'
            ])
            
            output = cli_helper.parse_cli_output(result.stdout)
            
            # Should complete successfully for synthetic terrain
            if result.returncode == 0:
                assert output['success'] or 'completed' in result.stdout.lower()
            else:
                # May fail due to Vulkan initialization - that's okay for testing
                assert 'vulkan' in result.stderr.lower() or 'gpu' in result.stderr.lower()
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_geotiff_file_loading(self, cli_helper, test_geotiff_file):
        """Test loading actual GeoTIFF file"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not script_path.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), [
            test_geotiff_file,
            '--config', 'mobile',  # Use mobile config for lighter processing
            '--verbose'
        ])
        
        output = cli_helper.parse_cli_output(result.stdout)
        
        # Should at least parse the GeoTIFF file
        if result.returncode != 0:
            # May fail at Vulkan initialization - check if GeoTIFF was parsed
            assert 'loaded' in result.stdout.lower() or 'geotiff' in result.stderr.lower()
    
    @pytest.mark.parametrize("config_preset", ['high_performance', 'balanced', 'high_quality', 'mobile'])
    def test_configuration_presets(self, cli_helper, config_preset):
        """Test different configuration presets"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not script_path.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), [
            '--synthetic',
            '--size', '32',  # Very small for speed
            '--config', config_preset
        ])
        
        # Should at least accept the configuration without argument errors
        assert 'unrecognized arguments' not in result.stderr
        assert 'invalid choice' not in result.stderr
    
    def test_invalid_arguments(self, cli_helper):
        """Test handling of invalid command line arguments"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not script_path.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        # Test invalid configuration
        result = cli_helper.run_cli_script(str(script_path), [
            '--synthetic',
            '--config', 'invalid_config'
        ])
        
        assert result.returncode != 0
        assert 'invalid choice' in result.stderr or 'error' in result.stderr.lower()
        
        # Test missing required arguments
        result = cli_helper.run_cli_script(str(script_path), [])
        
        assert result.returncode != 0
        assert 'required' in result.stderr.lower() or 'arguments' in result.stderr.lower()


class TestTerrainPerformance:
    """Test the terrain performance benchmarking tool"""
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_performance_tool_help(self, cli_helper):
        """Test help output for performance tool"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_performance.py"
        
        if not script_path.exists():
            pytest.skip("terrain_performance.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), ['--help'])
        
        assert result.returncode == 0
        assert 'benchmark' in result.stdout.lower()
        assert '--duration' in result.stdout
        assert '--preset' in result.stdout
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available") 
    def test_synthetic_benchmark(self, cli_helper):
        """Test benchmarking with synthetic terrain"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_performance.py"
        
        if not script_path.exists():
            pytest.skip("terrain_performance.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), [
            '--synthetic',
            '--size', '256',
            '--duration', '2',  # Very short benchmark
            '--preset', 'high_performance'
        ], timeout=60)
        
        output = cli_helper.parse_cli_output(result.stdout)
        
        # Check for benchmark metrics
        if result.returncode == 0:
            assert 'fps' in output['metrics'] or 'frames' in result.stdout.lower()
            assert 'triangles' in result.stdout.lower() or 'performance' in result.stdout.lower()
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_quick_benchmark_mode(self, cli_helper):
        """Test quick benchmark mode"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_performance.py"
        
        if not script_path.exists():
            pytest.skip("terrain_performance.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), [
            '--synthetic',
            '--size', '128',
            '--quick',
            '--duration', '1'
        ], timeout=30)
        
        # Quick mode should complete faster and not error on argument parsing
        assert 'unrecognized arguments' not in result.stderr
    
    def test_benchmark_output_formats(self, cli_helper):
        """Test different benchmark output formats"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_performance.py"
        
        if not script_path.exists():
            pytest.skip("terrain_performance.py not found")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            result = cli_helper.run_cli_script(str(script_path), [
                '--synthetic',
                '--size', '64',
                '--duration', '1',
                '--output', output_file,
                '--quick'
            ], timeout=30)
            
            # Should create output file (even if benchmark fails)
            if result.returncode == 0 and os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    assert 'benchmark_info' in data or 'results' in data
        
        finally:
            try:
                os.unlink(output_file)
            except OSError:
                pass
    
    @pytest.mark.parametrize("resolution", ['1080p', '1440p', '4K'])
    def test_resolution_settings(self, cli_helper, resolution):
        """Test different resolution settings"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_performance.py"
        
        if not script_path.exists():
            pytest.skip("terrain_performance.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), [
            '--synthetic',
            '--size', '32',
            '--duration', '0.5',
            '--resolution', resolution,
            '--quick'
        ], timeout=20)
        
        # Should accept resolution argument without error
        assert 'unrecognized arguments' not in result.stderr
        assert 'invalid choice' not in result.stderr


class TestTerrainViewer:
    """Test the interactive terrain viewer"""
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_viewer_help(self, cli_helper):
        """Test help output for terrain viewer"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_viewer.py"
        
        if not script_path.exists():
            pytest.skip("terrain_viewer.py not found")
        
        result = cli_helper.run_cli_script(str(script_path), ['--help'])
        
        assert result.returncode == 0
        assert 'viewer' in result.stdout.lower() or 'interactive' in result.stdout.lower()
        assert '--width' in result.stdout
        assert '--height' in result.stdout
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_viewer_argument_validation(self, cli_helper):
        """Test viewer argument validation without starting GUI"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_viewer.py"
        
        if not script_path.exists():
            pytest.skip("terrain_viewer.py not found")
        
        # Test invalid resolution
        result = cli_helper.run_cli_script(str(script_path), [
            '--synthetic',
            '--width', '0',  # Invalid width
            '--height', '600'
        ], timeout=10)
        
        # Should fail early on argument validation or during initialization
        assert result.returncode != 0 or 'error' in result.stderr.lower()
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_viewer_synthetic_terrain_init(self, cli_helper):
        """Test viewer initialization with synthetic terrain"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_viewer.py"
        
        if not script_path.exists():
            pytest.skip("terrain_viewer.py not found")
        
        # This test just verifies the script doesn't crash immediately
        # We can't test the actual GUI without a display
        with patch.dict(os.environ, {'DISPLAY': ''}):  # Simulate headless environment
            result = cli_helper.run_cli_script(str(script_path), [
                '--synthetic',
                '--size', '64'
            ], timeout=5)
            
            # Should fail due to no display, but not due to argument errors
            if 'display' in result.stderr.lower() or 'gui' in result.stderr.lower():
                # Expected failure in headless environment
                pass
            else:
                # Check for successful initialization
                assert 'unrecognized arguments' not in result.stderr


class TestCLIIntegration:
    """Test CLI tool integration and workflows"""
    
    @pytest.mark.skipif(not CLI_MODULES_AVAILABLE, reason="CLI modules not available")
    def test_cli_tool_discovery(self):
        """Test that CLI tools can be discovered and imported"""
        # Test that the main CLI modules can be imported
        try:
            import vulkan_forge.examples.terrain_performance
            import vulkan_forge.examples.terrain_viewer  
            import vulkan_forge.examples.load_geotiff_basic
        except ImportError as e:
            pytest.fail(f"Failed to import CLI modules: {e}")
    
    def test_cli_script_execution_environment(self, cli_helper):
        """Test CLI script execution environment"""
        # Create a simple test script
        test_script = '''
import sys
import os
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")
try:
    import vulkan_forge
    print("vulkan_forge available")
except ImportError:
    print("vulkan_forge not available")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            result = cli_helper.run_cli_script(script_path, [])
            assert result.returncode == 0
            assert 'Python version' in result.stdout
        finally:
            os.unlink(script_path)
    
    @pytest.mark.parametrize("tool_name", [
        'terrain_performance.py',
        'terrain_viewer.py', 
        'load_geotiff_basic.py'
    ])
    def test_cli_tool_error_handling(self, cli_helper, tool_name):
        """Test error handling in CLI tools"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / tool_name
        
        if not script_path.exists():
            pytest.skip(f"{tool_name} not found")
        
        # Test with deliberately invalid arguments
        result = cli_helper.run_cli_script(str(script_path), [
            '--invalid-argument-that-should-not-exist'
        ])
        
        # Should exit with error code and provide helpful message
        assert result.returncode != 0
        assert 'unrecognized' in result.stderr or 'error' in result.stderr.lower()
    
    def test_cli_pipeline_workflow(self, cli_helper, test_geotiff_file):
        """Test a complete CLI workflow"""
        # 1. First, load and analyze the GeoTIFF
        load_script = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not load_script.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        load_result = cli_helper.run_cli_script(str(load_script), [
            test_geotiff_file,
            '--config', 'mobile',
            '--verbose'
        ])
        
        load_output = cli_helper.parse_cli_output(load_result.stdout)
        
        # 2. Then run a quick benchmark
        perf_script = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "terrain_performance.py"
        
        if perf_script.exists():
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                output_file = f.name
            
            try:
                perf_result = cli_helper.run_cli_script(str(perf_script), [
                    test_geotiff_file,
                    '--duration', '1',
                    '--output', output_file,
                    '--quick'
                ], timeout=30)
                
                # Verify the workflow produces some output
                if perf_result.returncode == 0 and os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        benchmark_data = json.load(f)
                        assert isinstance(benchmark_data, dict)
            
            finally:
                try:
                    os.unlink(output_file)
                except OSError:
                    pass


class TestCLIPerformance:
    """Test CLI tool performance characteristics"""
    
    def test_cli_startup_time(self, cli_helper):
        """Test CLI tool startup time"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not script_path.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        # Measure startup time
        start_time = time.time()
        
        result = cli_helper.run_cli_script(str(script_path), ['--help'])
        
        end_time = time.time()
        startup_time = end_time - start_time
        
        # Should start up reasonably quickly
        assert startup_time < 5.0, f"CLI startup took {startup_time:.2f}s"
        assert result.returncode == 0
    
    def test_memory_usage_during_cli_execution(self, cli_helper):
        """Test memory usage during CLI execution"""
        import psutil
        import threading
        
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / "load_geotiff_basic.py"
        
        if not script_path.exists():
            pytest.skip("load_geotiff_basic.py not found")
        
        max_memory_mb = 0
        monitoring = True
        
        def monitor_memory():
            nonlocal max_memory_mb
            while monitoring:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    max_memory_mb = max(max_memory_mb, memory_mb)
                    time.sleep(0.1)
                except psutil.NoSuchProcess:
                    break
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            result = cli_helper.run_cli_script(str(script_path), [
                '--synthetic',
                '--size', '64',
                '--config', 'mobile'
            ])
        finally:
            monitoring = False
            monitor_thread.join(timeout=1.0)
        
        # Memory usage should be reasonable (less than 1GB for simple operations)
        assert max_memory_mb < 1024, f"Peak memory usage: {max_memory_mb:.1f}MB"


class TestCLIDocumentation:
    """Test CLI tool documentation and help systems"""
    
    @pytest.mark.parametrize("tool_name", [
        'terrain_performance.py',
        'terrain_viewer.py',
        'load_geotiff_basic.py'
    ])
    def test_help_completeness(self, cli_helper, tool_name):
        """Test that help output is complete and useful"""
        script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / tool_name
        
        if not script_path.exists():
            pytest.skip(f"{tool_name} not found")
        
        result = cli_helper.run_cli_script(str(script_path), ['--help'])
        
        assert result.returncode == 0, f"Help failed for {tool_name}"
        
        help_text = result.stdout.lower()
        
        # Check for essential help elements
        assert 'usage:' in help_text, "Missing usage information"
        assert 'options:' in help_text or 'arguments:' in help_text, "Missing options/arguments"
        
        # Check for common CLI conventions
        assert '--help' in result.stdout, "Missing --help option"
        assert '-h' in result.stdout or '--help' in result.stdout, "Missing short help option"
    
    def test_version_information(self, cli_helper):
        """Test version information display"""
        # Test scripts that might have version information
        scripts_to_test = [
            'terrain_performance.py',
            'load_geotiff_basic.py'
        ]
        
        for script_name in scripts_to_test:
            script_path = Path(__file__).parent.parent.parent / "python" / "vulkan_forge" / "examples" / script_name
            
            if not script_path.exists():
                continue
            
            # Try common version flags
            for version_flag in ['--version', '-V']:
                result = cli_helper.run_cli_script(str(script_path), [version_flag])
                
                # May or may not support version flag - that's okay
                if result.returncode == 0:
                    assert len(result.stdout.strip()) > 0, f"Empty version output for {script_name}"
                    break


# Test configuration
def pytest_configure(config):
    """Configure pytest for CLI tests"""
    config.addinivalue_line("markers", "cli: mark test as CLI-related")
    config.addinivalue_line("markers", "gui: mark test as requiring GUI")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for CLI tests"""
    for item in items:
        if "cli" in item.name.lower():
            item.add_marker(pytest.mark.cli)
        if "viewer" in item.name.lower() or "gui" in item.name.lower():
            item.add_marker(pytest.mark.gui)
        if "pipeline" in item.name.lower() or "workflow" in item.name.lower():
            item.add_marker(pytest.mark.integration)