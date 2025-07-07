from pathlib import Path

import vulkan_forge as vf


def test_obj_loader_basic(tmp_path: Path) -> None:
    obj = "\n".join(
        [
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 2 3",
        ]
    )
    file = tmp_path / "triangle.obj"
    file.write_text(obj)

    mesh = vf.load_obj(file)

    assert mesh.vertex_count == 3
    assert mesh.index_count == 3
