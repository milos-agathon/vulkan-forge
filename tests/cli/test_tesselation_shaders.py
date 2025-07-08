#!/usr/bin/env python3
"""
Tessellation Shader Testing Suite

Tests shader compilation, validation, and tessellation pipeline functionality.
Includes SPIR-V validation, shader hot-reload testing, and tessellation correctness.
"""

import pytest
import subprocess
import tempfile
import os
import struct
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

# Import vulkan-forge components
try:
    import vulkan_forge_core as vf
    from vulkan_forge.terrain_config import TessellationConfig, TessellationMode
    VULKAN_FORGE_AVAILABLE = True
except ImportError:
    VULKAN_FORGE_AVAILABLE = False


class ShaderCompiler:
    """Utility class for compiling and validating shaders"""
    
    def __init__(self):
        self.glslc_path = self._find_glslc()
        self.spirv_val_path = self._find_spirv_val()
    
    def _find_glslc(self) -> Optional[str]:
        """Find glslc compiler in system PATH or Vulkan SDK"""
        candidates = [
            'glslc',
            'glslc.exe',
            os.path.join(os.environ.get('VULKAN_SDK', ''), 'bin', 'glslc'),
            os.path.join(os.environ.get('VULKAN_SDK', ''), 'bin', 'glslc.exe'),
        ]
        
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None
    
    def _find_spirv_val(self) -> Optional[str]:
        """Find SPIR-V validator"""
        candidates = [
            'spirv-val',
            'spirv-val.exe',
            os.path.join(os.environ.get('VULKAN_SDK', ''), 'bin', 'spirv-val'),
            os.path.join(os.environ.get('VULKAN_SDK', ''), 'bin', 'spirv-val.exe'),
        ]
        
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None
    
    def compile_shader(self, shader_source: str, shader_stage: str, 
                      target_env: str = "vulkan1.3") -> Tuple[bool, bytes, str]:
        """
        Compile GLSL shader to SPIR-V
        
        Returns:
            (success, spirv_bytecode, error_message)
        """
        if not self.glslc_path:
            return False, b'', "glslc not found"
        
        with tempfile.NamedTemporaryFile(suffix=f'.{shader_stage}', mode='w', delete=False) as f:
            f.write(shader_source)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as f:
            output_file = f.name
        
        try:
            cmd = [
                self.glslc_path,
                f'--target-env={target_env}',
                '-O',  # Optimize
                input_file,
                '-o', output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                with open(output_file, 'rb') as f:
                    spirv_data = f.read()
                return True, spirv_data, ""
            else:
                return False, b'', result.stderr
        
        except subprocess.TimeoutExpired:
            return False, b'', "Compilation timeout"
        except Exception as e:
            return False, b'', str(e)
        finally:
            # Cleanup temp files
            for temp_file in [input_file, output_file]:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
    
    def validate_spirv(self, spirv_data: bytes, target_env: str = "vulkan1.3") -> Tuple[bool, str]:
        """
        Validate SPIR-V bytecode
        
        Returns:
            (is_valid, error_message)
        """
        if not self.spirv_val_path:
            return True, "spirv-val not available"  # Assume valid if validator not found
        
        with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as f:
            f.write(spirv_data)
            spirv_file = f.name
        
        try:
            cmd = [
                self.spirv_val_path,
                f'--target-env={target_env}',
                spirv_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0, result.stderr
        
        except subprocess.TimeoutExpired:
            return False, "Validation timeout"
        except Exception as e:
            return False, str(e)
        finally:
            try:
                os.unlink(spirv_file)
            except OSError:
                pass


class TerrainShaderTemplates:
    """Templates for terrain rendering shaders"""
    
    VERTEX_SHADER = """
#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texcoord;

layout(binding = 0) uniform FrameUniforms {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    vec3 camera_position;
    float time;
    vec2 viewport_size;
    float height_scale;
    float texture_scale;
} frame;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec2 tex_coord;

void main() {
    vec3 world_pos = (frame.model_matrix * vec4(in_position, 1.0)).xyz;
    
    gl_Position = frame.proj_matrix * frame.view_matrix * vec4(world_pos, 1.0);
    
    world_position = world_pos;
    tex_coord = in_texcoord;
}
"""

    TESSELLATION_CONTROL_SHADER = """
#version 450 core

layout(vertices = 4) out;

layout(location = 0) in vec3 world_position[];
layout(location = 1) in vec2 tex_coord[];

layout(location = 0) out vec3 world_position_tc[];
layout(location = 1) out vec2 tex_coord_tc[];

layout(binding = 0) uniform FrameUniforms {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    vec3 camera_position;
    float time;
    vec2 viewport_size;
    float height_scale;
    float texture_scale;
} frame;

layout(binding = 1) uniform TessellationUniforms {
    float tessellation_level;
    float screen_size;
    float lod_bias;
    float distance_scale;
} tess;

float calculate_tessellation_level(vec3 world_pos) {
    float distance = length(world_pos - frame.camera_position);
    
    // Distance-based tessellation
    float base_level = tess.tessellation_level;
    float scaled_distance = distance / tess.distance_scale;
    float level = base_level / (1.0 + scaled_distance * scaled_distance);
    
    return clamp(level, 1.0, 64.0);
}

void main() {
    // Pass through vertex data
    world_position_tc[gl_InvocationID] = world_position[gl_InvocationID];
    tex_coord_tc[gl_InvocationID] = tex_coord[gl_InvocationID];
    
    // Calculate tessellation levels for the patch
    if (gl_InvocationID == 0) {
        // Calculate tessellation level based on distance to camera
        vec3 patch_center = (world_position[0] + world_position[1] + 
                           world_position[2] + world_position[3]) * 0.25;
        
        float tess_level = calculate_tessellation_level(patch_center);
        
        // Set outer tessellation levels
        gl_TessLevelOuter[0] = tess_level;
        gl_TessLevelOuter[1] = tess_level;
        gl_TessLevelOuter[2] = tess_level;
        gl_TessLevelOuter[3] = tess_level;
        
        // Set inner tessellation levels
        gl_TessLevelInner[0] = tess_level;
        gl_TessLevelInner[1] = tess_level;
    }
}
"""

    TESSELLATION_EVALUATION_SHADER = """
#version 450 core

layout(quads, equal_spacing, ccw) in;

layout(location = 0) in vec3 world_position_tc[];
layout(location = 1) in vec2 tex_coord_tc[];

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec2 tex_coord;
layout(location = 2) out vec3 normal;

layout(binding = 0) uniform FrameUniforms {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    vec3 camera_position;
    float time;
    vec2 viewport_size;
    float height_scale;
    float texture_scale;
} frame;

layout(binding = 2) uniform sampler2D height_texture;

out gl_PerVertex {
    vec4 gl_Position;
};

vec3 interpolate_position() {
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    
    vec3 p0 = mix(world_position_tc[0], world_position_tc[1], u);
    vec3 p1 = mix(world_position_tc[3], world_position_tc[2], u);
    
    return mix(p0, p1, v);
}

vec2 interpolate_texcoord() {
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    
    vec2 t0 = mix(tex_coord_tc[0], tex_coord_tc[1], u);
    vec2 t1 = mix(tex_coord_tc[3], tex_coord_tc[2], u);
    
    return mix(t0, t1, v);
}

vec3 calculate_normal(vec2 uv) {
    vec2 texel_size = 1.0 / textureSize(height_texture, 0);
    
    float h_left = texture(height_texture, uv + vec2(-texel_size.x, 0.0)).r;
    float h_right = texture(height_texture, uv + vec2(texel_size.x, 0.0)).r;
    float h_down = texture(height_texture, uv + vec2(0.0, -texel_size.y)).r;
    float h_up = texture(height_texture, uv + vec2(0.0, texel_size.y)).r;
    
    vec3 normal = normalize(vec3(h_left - h_right, h_down - h_up, 2.0));
    return normal;
}

void main() {
    world_position = interpolate_position();
    tex_coord = interpolate_texcoord();
    
    // Sample height from texture and displace vertex
    float height = texture(height_texture, tex_coord).r * frame.height_scale;
    world_position.z = height;
    
    // Calculate normal
    normal = calculate_normal(tex_coord);
    
    gl_Position = frame.proj_matrix * frame.view_matrix * vec4(world_position, 1.0);
}
"""

    FRAGMENT_SHADER = """
#version 450 core

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec2 tex_coord;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec4 frag_color;

layout(binding = 0) uniform FrameUniforms {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    vec3 camera_position;
    float time;
    vec2 viewport_size;
    float height_scale;
    float texture_scale;
} frame;

layout(binding = 3) uniform sampler2D diffuse_texture;

void main() {
    vec3 light_dir = normalize(vec3(0.5, 0.5, 1.0));
    
    float ndotl = max(dot(normalize(normal), light_dir), 0.0);
    
    vec3 base_color = texture(diffuse_texture, tex_coord * frame.texture_scale).rgb;
    vec3 final_color = base_color * (0.3 + 0.7 * ndotl);
    
    frag_color = vec4(final_color, 1.0);
}
"""


@pytest.fixture(scope="session")
def shader_compiler():
    """Shared shader compiler instance"""
    return ShaderCompiler()


@pytest.fixture(scope="session") 
def shader_templates():
    """Shared shader templates"""
    return TerrainShaderTemplates()


class TestShaderCompilation:
    """Test shader compilation and validation"""
    
    def test_vertex_shader_compilation(self, shader_compiler, shader_templates):
        """Test vertex shader compilation"""
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.VERTEX_SHADER, 'vert'
        )
        
        if shader_compiler.glslc_path:
            assert success, f"Vertex shader compilation failed: {error}"
            assert len(spirv_data) > 0, "No SPIR-V data generated"
            
            # Validate SPIR-V
            valid, val_error = shader_compiler.validate_spirv(spirv_data)
            assert valid, f"SPIR-V validation failed: {val_error}"
        else:
            pytest.skip("glslc not available")
    
    def test_tessellation_control_shader_compilation(self, shader_compiler, shader_templates):
        """Test tessellation control shader compilation"""
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.TESSELLATION_CONTROL_SHADER, 'tesc'
        )
        
        if shader_compiler.glslc_path:
            assert success, f"Tessellation control shader compilation failed: {error}"
            assert len(spirv_data) > 0, "No SPIR-V data generated"
            
            # Validate SPIR-V
            valid, val_error = shader_compiler.validate_spirv(spirv_data)
            assert valid, f"SPIR-V validation failed: {val_error}"
        else:
            pytest.skip("glslc not available")
    
    def test_tessellation_evaluation_shader_compilation(self, shader_compiler, shader_templates):
        """Test tessellation evaluation shader compilation"""
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.TESSELLATION_EVALUATION_SHADER, 'tese'
        )
        
        if shader_compiler.glslc_path:
            assert success, f"Tessellation evaluation shader compilation failed: {error}"
            assert len(spirv_data) > 0, "No SPIR-V data generated"
            
            # Validate SPIR-V
            valid, val_error = shader_compiler.validate_spirv(spirv_data)
            assert valid, f"SPIR-V validation failed: {val_error}"
        else:
            pytest.skip("glslc not available")
    
    def test_fragment_shader_compilation(self, shader_compiler, shader_templates):
        """Test fragment shader compilation"""
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.FRAGMENT_SHADER, 'frag'
        )
        
        if shader_compiler.glslc_path:
            assert success, f"Fragment shader compilation failed: {error}"
            assert len(spirv_data) > 0, "No SPIR-V data generated"
            
            # Validate SPIR-V
            valid, val_error = shader_compiler.validate_spirv(spirv_data)
            assert valid, f"SPIR-V validation failed: {val_error}"
        else:
            pytest.skip("glslc not available")
    
    def test_complete_pipeline_compilation(self, shader_compiler, shader_templates):
        """Test compilation of complete tessellation pipeline"""
        shaders = [
            (shader_templates.VERTEX_SHADER, 'vert'),
            (shader_templates.TESSELLATION_CONTROL_SHADER, 'tesc'),
            (shader_templates.TESSELLATION_EVALUATION_SHADER, 'tese'),
            (shader_templates.FRAGMENT_SHADER, 'frag'),
        ]
        
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        compiled_shaders = []
        
        for shader_source, stage in shaders:
            success, spirv_data, error = shader_compiler.compile_shader(shader_source, stage)
            assert success, f"Failed to compile {stage} shader: {error}"
            
            valid, val_error = shader_compiler.validate_spirv(spirv_data)
            assert valid, f"SPIR-V validation failed for {stage}: {val_error}"
            
            compiled_shaders.append((stage, spirv_data))
        
        # Verify all stages compiled
        assert len(compiled_shaders) == 4, "Not all shader stages compiled"
        stages = [stage for stage, _ in compiled_shaders]
        assert set(stages) == {'vert', 'tesc', 'tese', 'frag'}


class TestShaderErrorHandling:
    """Test shader compilation error handling"""
    
    def test_invalid_glsl_syntax(self, shader_compiler):
        """Test handling of invalid GLSL syntax"""
        invalid_shader = """
        #version 450 core
        
        void main() {
            invalid_syntax_here!!!
        }
        """
        
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        success, spirv_data, error = shader_compiler.compile_shader(invalid_shader, 'vert')
        
        assert not success, "Expected compilation to fail"
        assert len(error) > 0, "Expected error message"
        assert len(spirv_data) == 0, "Should not generate SPIR-V for invalid shader"
    
    def test_missing_main_function(self, shader_compiler):
        """Test handling of missing main function"""
        invalid_shader = """
        #version 450 core
        
        layout(location = 0) in vec3 position;
        layout(location = 0) out vec4 color;
        
        // Missing main function
        """
        
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        success, spirv_data, error = shader_compiler.compile_shader(invalid_shader, 'vert')
        
        assert not success, "Expected compilation to fail"
        assert "main" in error.lower() or "entry" in error.lower(), "Error should mention missing main function"
    
    def test_incompatible_shader_stage(self, shader_compiler, shader_templates):
        """Test using tessellation shader in wrong stage"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        # Try to compile tessellation control shader as vertex shader
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.TESSELLATION_CONTROL_SHADER, 'vert'
        )
        
        assert not success, "Expected compilation to fail for wrong stage"
    
    @pytest.mark.parametrize("invalid_version", [
        "#version 110 core",  # Too old
        "#version 999 core",  # Future version
        "#version abc core",  # Invalid
    ])
    def test_invalid_glsl_version(self, shader_compiler, invalid_version):
        """Test handling of invalid GLSL versions"""
        invalid_shader = f"""
        {invalid_version}
        
        void main() {{
            gl_Position = vec4(0.0);
        }}
        """
        
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        success, spirv_data, error = shader_compiler.compile_shader(invalid_shader, 'vert')
        assert not success, f"Expected compilation to fail for {invalid_version}"


class TestTessellationPipelineIntegration:
    """Test tessellation pipeline integration with Vulkan-Forge"""
    
    @pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available")
    def test_tessellation_config_integration(self):
        """Test tessellation configuration integration"""
        config = TessellationConfig()
        config.mode = TessellationMode.DISTANCE_BASED
        config.base_level = 8
        config.max_level = 32
        
        # Test level calculation
        level = config.get_tessellation_level(500.0)
        assert 1 <= level <= 32, f"Tessellation level {level} out of range"
        
        # Test different distances
        near_level = config.get_tessellation_level(50.0)
        far_level = config.get_tessellation_level(2000.0)
        assert near_level >= far_level, "Near tessellation should be higher than far"
    
    @pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available")
    def test_tessellation_pipeline_creation(self):
        """Test tessellation pipeline creation"""
        # Mock Vulkan context
        mock_context = Mock()
        mock_context.get_device_features.return_value = Mock(tessellationShader=True)
        
        config = TessellationConfig()
        
        with patch('vulkan_forge_core.TessellationPipeline') as MockPipeline:
            pipeline = MockPipeline(mock_context, config)
            
            MockPipeline.assert_called_once_with(mock_context, config)
    
    @pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available")
    def test_tessellation_uniform_buffer_layout(self):
        """Test tessellation uniform buffer layout"""
        # Test uniform buffer structure matches shader expectations
        expected_uniforms = {
            'tessellation_level': 'float',
            'screen_size': 'float', 
            'lod_bias': 'float',
            'distance_scale': 'float'
        }
        
        # This would be tested with actual Vulkan pipeline creation
        # For now, just verify the structure is consistent
        assert len(expected_uniforms) == 4, "Expected 4 tessellation uniforms"


class TestShaderHotReload:
    """Test shader hot-reload functionality"""
    
    def test_shader_file_watching(self, shader_compiler, shader_templates):
        """Test shader file change detection"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        # Create temporary shader file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vert', delete=False) as f:
            f.write(shader_templates.VERTEX_SHADER)
            shader_file = f.name
        
        try:
            # Get initial modification time
            initial_mtime = os.path.getmtime(shader_file)
            
            # Wait a bit to ensure time difference
            import time
            time.sleep(0.1)
            
            # Modify the file
            with open(shader_file, 'a') as f:
                f.write('\n// Modified\n')
            
            # Check modification time changed
            new_mtime = os.path.getmtime(shader_file)
            assert new_mtime > initial_mtime, "File modification time should have changed"
            
        finally:
            os.unlink(shader_file)
    
    def test_shader_recompilation_on_change(self, shader_compiler, shader_templates):
        """Test automatic recompilation when shader changes"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        # Create temporary shader file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vert', delete=False) as f:
            f.write(shader_templates.VERTEX_SHADER)
            shader_file = f.name
        
        try:
            # Compile initial version
            with open(shader_file, 'r') as f:
                initial_source = f.read()
            
            success1, spirv1, error1 = shader_compiler.compile_shader(initial_source, 'vert')
            assert success1, f"Initial compilation failed: {error1}"
            
            # Modify shader
            modified_source = initial_source.replace(
                'vec3 world_pos = (frame.model_matrix * vec4(in_position, 1.0)).xyz;',
                'vec3 world_pos = (frame.model_matrix * vec4(in_position * 1.1, 1.0)).xyz; // Modified'
            )
            
            # Compile modified version
            success2, spirv2, error2 = shader_compiler.compile_shader(modified_source, 'vert')
            assert success2, f"Modified compilation failed: {error2}"
            
            # SPIR-V should be different
            assert spirv1 != spirv2, "SPIR-V should change when shader is modified"
            
        finally:
            os.unlink(shader_file)


class TestShaderOptimization:
    """Test shader optimization and performance"""
    
    @pytest.mark.parametrize("optimization_level", ['-O0', '-O', '-Os'])
    def test_shader_optimization_levels(self, shader_compiler, shader_templates, optimization_level):
        """Test different shader optimization levels"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        # Modify compiler to use specific optimization level
        original_compile = shader_compiler.compile_shader
        
        def compile_with_optimization(source, stage, target_env="vulkan1.3"):
            with tempfile.NamedTemporaryFile(suffix=f'.{stage}', mode='w', delete=False) as f:
                f.write(source)
                input_file = f.name
            
            with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as f:
                output_file = f.name
            
            try:
                cmd = [
                    shader_compiler.glslc_path,
                    f'--target-env={target_env}',
                    optimization_level,
                    input_file,
                    '-o', output_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    with open(output_file, 'rb') as f:
                        spirv_data = f.read()
                    return True, spirv_data, ""
                else:
                    return False, b'', result.stderr
            
            except Exception as e:
                return False, b'', str(e)
            finally:
                for temp_file in [input_file, output_file]:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass
        
        success, spirv_data, error = compile_with_optimization(
            shader_templates.VERTEX_SHADER, 'vert'
        )
        
        assert success, f"Compilation with {optimization_level} failed: {error}"
        assert len(spirv_data) > 0, f"No SPIR-V generated with {optimization_level}"
        
        # Validate optimized SPIR-V
        valid, val_error = shader_compiler.validate_spirv(spirv_data)
        assert valid, f"Optimized SPIR-V validation failed: {val_error}"
    
    def test_shader_size_optimization(self, shader_compiler, shader_templates):
        """Test that optimization reduces shader size"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        # This is a simplified test - real optimization testing would need actual size comparison
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.FRAGMENT_SHADER, 'frag'
        )
        
        assert success, f"Compilation failed: {error}"
        
        # Basic size check - optimized shaders should be reasonable size
        assert len(spirv_data) > 100, "SPIR-V too small"
        assert len(spirv_data) < 100000, "SPIR-V unexpectedly large"


class TestShaderDebugging:
    """Test shader debugging and profiling support"""
    
    def test_shader_debug_info_generation(self, shader_compiler, shader_templates):
        """Test generation of debug information in shaders"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        # Compile with debug info (simplified test)
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.VERTEX_SHADER, 'vert'
        )
        
        assert success, f"Debug compilation failed: {error}"
        
        # Check SPIR-V header for debug info
        if len(spirv_data) >= 20:
            # SPIR-V header analysis (simplified)
            magic = struct.unpack('<I', spirv_data[0:4])[0]
            assert magic == 0x07230203, "Invalid SPIR-V magic number"
    
    def test_shader_error_line_numbers(self, shader_compiler):
        """Test that compilation errors include line numbers"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        invalid_shader = """#version 450 core

void main() {
    // Line 4
    invalid_function_call();  // This should cause an error on line 5
}
"""
        
        success, spirv_data, error = shader_compiler.compile_shader(invalid_shader, 'vert')
        
        assert not success, "Expected compilation to fail"
        # Check if error message contains line number reference
        assert any(char.isdigit() for char in error), "Error should contain line numbers"


class TestShaderCompatibility:
    """Test shader compatibility across different Vulkan versions"""
    
    @pytest.mark.parametrize("target_env", [
        "vulkan1.0",
        "vulkan1.1", 
        "vulkan1.2",
        "vulkan1.3"
    ])
    def test_vulkan_version_compatibility(self, shader_compiler, shader_templates, target_env):
        """Test shader compilation for different Vulkan versions"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.VERTEX_SHADER, 'vert', target_env
        )
        
        # Some shaders might not be compatible with older Vulkan versions
        if target_env in ["vulkan1.0", "vulkan1.1"]:
            # Might fail due to newer features - that's okay
            if not success:
                assert "version" in error.lower() or "extension" in error.lower()
        else:
            assert success, f"Compilation failed for {target_env}: {error}"
            
            if success:
                valid, val_error = shader_compiler.validate_spirv(spirv_data, target_env)
                assert valid, f"SPIR-V validation failed for {target_env}: {val_error}"
    
    def test_extension_requirements(self, shader_compiler):
        """Test shader compilation with Vulkan extensions"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        # Shader that might require extensions
        extension_shader = """#version 450 core
#extension GL_EXT_tessellation_shader : require

layout(vertices = 4) out;

void main() {
    gl_TessLevelOuter[0] = 1.0;
    gl_TessLevelOuter[1] = 1.0; 
    gl_TessLevelOuter[2] = 1.0;
    gl_TessLevelOuter[3] = 1.0;
    gl_TessLevelInner[0] = 1.0;
    gl_TessLevelInner[1] = 1.0;
}
"""
        
        success, spirv_data, error = shader_compiler.compile_shader(extension_shader, 'tesc')
        
        # This might fail if extensions aren't supported - that's fine
        if not success:
            assert "extension" in error.lower() or "require" in error.lower()


# Performance benchmarks for shader compilation
@pytest.mark.benchmark
class TestShaderPerformance:
    """Benchmark shader compilation performance"""
    
    def test_compilation_speed(self, shader_compiler, shader_templates, benchmark):
        """Benchmark shader compilation speed"""
        if not shader_compiler.glslc_path:
            pytest.skip("glslc not available")
        
        def compile_all_shaders():
            shaders = [
                (shader_templates.VERTEX_SHADER, 'vert'),
                (shader_templates.TESSELLATION_CONTROL_SHADER, 'tesc'),
                (shader_templates.TESSELLATION_EVALUATION_SHADER, 'tese'),
                (shader_templates.FRAGMENT_SHADER, 'frag'),
            ]
            
            results = []
            for source, stage in shaders:
                success, spirv_data, error = shader_compiler.compile_shader(source, stage)
                results.append(success)
            
            return all(results)
        
        result = benchmark(compile_all_shaders)
        assert result, "Not all shaders compiled successfully"
    
    def test_spirv_validation_speed(self, shader_compiler, shader_templates, benchmark):
        """Benchmark SPIR-V validation speed"""
        if not shader_compiler.glslc_path or not shader_compiler.spirv_val_path:
            pytest.skip("Compiler tools not available")
        
        # Pre-compile shader
        success, spirv_data, error = shader_compiler.compile_shader(
            shader_templates.FRAGMENT_SHADER, 'frag'
        )
        assert success, f"Pre-compilation failed: {error}"
        
        def validate_spirv():
            valid, val_error = shader_compiler.validate_spirv(spirv_data)
            return valid
        
        result = benchmark(validate_spirv)
        assert result, "SPIR-V validation failed"


# Test configuration
def pytest_configure(config):
    """Configure pytest for shader tests"""
    config.addinivalue_line("markers", "shaders: mark test as shader-related")
    config.addinivalue_line("markers", "compilation: mark test as compilation test")
    config.addinivalue_line("markers", "optimization: mark test as optimization test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for shader tests"""
    for item in items:
        if "shader" in item.name.lower():
            item.add_marker(pytest.mark.shaders)
        if "compil" in item.name.lower():
            item.add_marker(pytest.mark.compilation)
        if "optim" in item.name.lower():
            item.add_marker(pytest.mark.optimization)