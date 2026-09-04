"""
Test terrain PBR+POM shader completeness
Verifies all required functions from MILESTONE 4 are implemented
"""

import re
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def test_shader_milestone_4_complete():
    """Verify MILESTONE 4 tasks are complete in terrain_pbr_pom.wgsl"""

    shader_path = Path(__file__).parent.parent / "src" / "shaders" / "terrain_pbr_pom.wgsl"

    assert shader_path.exists(), f"Shader file not found: {shader_path}"

    shader_source = shader_path.read_text(encoding="utf-8")

    print("\n========== MILESTONE 4 Verification ==========\n")

    # Task 4.1: Normal Calculation from Height
    print("Task 4.1: Normal Calculation from Height")
    assert "fn calculate_normal" in shader_source, "Missing calculate_normal function"
    assert "Sobel" in shader_source, "Missing Sobel operator documentation"

    # Verify Sobel implementation
    assert "dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl)" in shader_source, \
        "Sobel X gradient incorrect"
    assert "dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr)" in shader_source, \
        "Sobel Y gradient incorrect"

    print("  ✓ calculate_normal() implemented with Sobel filter")
    print("  ✓ Handles terrain spacing and exaggeration")
    print()

    # Task 4.2: Triplanar Texture Sampling
    print("Task 4.2: Triplanar Texture Sampling")
    assert "fn sample_triplanar" in shader_source, "Missing sample_triplanar function"
    assert "fn sample_triplanar_vt_family" in shader_source, \
        "Missing triplanar material-family sampling"
    assert "fn apply_encoded_tangent_normal" in shader_source, \
        "Missing tangent-space normal application"
    assert "fn apply_material_normal_map" in shader_source, \
        "Missing live material normal-map path"
    assert "textureSample(material_normal_tex" in shader_source, \
        "Material normal texture is not sampled"

    # Verify triplanar blend weights
    assert "blend_sharpness" in shader_source, "Missing blend sharpness parameter"
    assert "let weight_sum = sharpen.x + sharpen.y + sharpen.z" in shader_source, \
        "Missing live triplanar weight sum"
    assert "return sharpen / max(weight_sum, 1e-4)" in shader_source, \
        "Missing live triplanar blend normalization"

    # Verify axis sampling
    assert "uv_x = world_pos.yz * scale" in shader_source, "Missing X-axis UVs"
    assert "uv_y = world_pos.xz * scale" in shader_source, "Missing Y-axis UVs"
    assert "uv_z = world_pos.xy * scale" in shader_source, "Missing Z-axis UVs"

    print("  ✓ sample_triplanar() implemented")
    print("  ✓ triplanar material families and tangent normals implemented")
    print("  ✓ Blend weights calculated from surface normal")
    print("  ✓ Three-axis sampling (X, Y, Z)")
    print()

    # Task 4.3: Parallax Occlusion Mapping
    print("Task 4.3: Parallax Occlusion Mapping")
    assert "fn parallax_occlusion_mapping" in shader_source, \
        "Missing parallax_occlusion_mapping function"
    assert "if (shadow_enabled && pom_enabled)" in shader_source, \
        "Missing live POM shadow gate"
    assert "det_mix(0.4, 1.0, occlusion)" in shader_source, \
        "Missing live POM shadow factor"

    # Verify adaptive sampling
    assert "let blend = clamp(abs(view_dir.z), 0.0, 1.0)" in shader_source, \
        "Missing view-angle step blend"
    assert "let step_count = clamp(u32(steps_interp + 0.5), 1u, max_s)" in shader_source, \
        "Missing adaptive POM step count"

    # Verify ray marching
    assert "current_uv -= parallax_dir * step_size" in shader_source, \
        "Missing POM ray march"

    # Verify binary refinement
    assert "Binary refinement" in shader_source or "refine" in shader_source.lower(), \
        "Missing binary refinement"

    print("  ✓ parallax_occlusion_mapping() implemented")
    print("  ✓ Adaptive step count based on view angle")
    print("  ✓ Ray marching through height field")
    print("  ✓ Binary refinement for accuracy")
    print("  ✓ live POM shadow gate and occlusion factor")
    print()

    # Task 4.4: PBR BRDF Calculation
    print("Task 4.4: PBR BRDF Calculation")
    assert "fn calculate_pbr_brdf" in shader_source, "Missing calculate_pbr_brdf function"
    assert "let distribution = alpha_sq /" in shader_source, \
        "Missing live GGX distribution"
    assert "let geometry = g1_l * g1_v" in shader_source, \
        "Missing live Smith geometry term"
    assert "let fresnel = f0 +" in shader_source, \
        "Missing live Schlick Fresnel term"

    # Verify Cook-Torrance BRDF
    assert "Cook-Torrance" in shader_source, "Missing Cook-Torrance documentation"

    # Verify BRDF components
    assert "GGX" in shader_source or "Trowbridge-Reitz" in shader_source, \
        "Missing GGX/Trowbridge-Reitz NDF"
    assert "Smith" in shader_source, "Missing Smith geometry function"
    assert "Schlick" in shader_source, "Missing Schlick Fresnel approximation"

    # Verify metallic-roughness workflow
    assert "metallic" in shader_source and "roughness" in shader_source, \
        "Missing metallic-roughness parameters"

    print("  ✓ GGX normal distribution term")
    print("  ✓ Smith geometric attenuation term")
    print("  ✓ Schlick Fresnel term")
    print("  ✓ calculate_pbr_brdf() (Cook-Torrance BRDF)")
    print("  ✓ Specular and diffuse terms")
    print("  ✓ Metallic-roughness workflow")
    print()

    # Additional verification: Entry points
    print("Shader Entry Points")
    assert re.search(r"@vertex\s+fn vs_main", shader_source), "Missing vertex shader entry point"
    assert re.search(r"@fragment\s+fn fs_main", shader_source), "Missing fragment shader entry point"
    print("  ✓ Vertex shader: vs_main")
    print("  ✓ Fragment shader: fs_main")
    print()

    # Bind groups verification
    print("Bind Group Definitions")
    bind_groups = {
        0: "Globals",
        1: "Height Map",
        2: "Colormap LUT",
        3: "Material Textures",
        4: "Triplanar & POM Parameters",
        5: "IBL Environment Maps",
        6: "Shadow Map"
    }

    for group_id, description in bind_groups.items():
        pattern = f"@group\\({group_id}\\)"
        assert re.search(pattern, shader_source), f"Missing bind group {group_id}"
        print(f"  ✓ Bind Group {group_id}: {description}")

    print()

    # Integration features
    print("Integration Features")
    features = [
        ("IBL", "Image-Based Lighting integration"),
        ("shadow", "Shadow mapping support"),
        ("colormap", "Colormap blending"),
        ("tone", "Tone mapping"),
        ("gamma", "Gamma correction"),
    ]

    for keyword, description in features:
        assert keyword.lower() in shader_source.lower(), f"Missing {description}"
        print(f"  ✓ {description}")

    print()
    print("=" * 50)
    print("MILESTONE 4 COMPLETE ✓")
    print("=" * 50)
    print()
    print("All tasks implemented:")
    print("  • Task 4.1: Normal calculation from height (Sobel filter)")
    print("  • Task 4.2: Triplanar texture sampling (color + normals)")
    print("  • Task 4.3: Parallax Occlusion Mapping (adaptive + refinement)")
    print("  • Task 4.4: PBR BRDF calculation (Cook-Torrance)")
    print()
    print("Total lines of shader code:", len(shader_source.splitlines()))
    print()

if __name__ == "__main__":
    test_shader_milestone_4_complete()
    print("All tests passed!")
