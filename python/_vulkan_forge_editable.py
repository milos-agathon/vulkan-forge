"""Site-editable hook for Vulkan-Forge.
Auto-adds the repo root to sys.path when this module is imported by the
*.pth* file generated via `pip install -e .`.
"""

import pathlib
import sys


def _activate() -> None:
    # <repo>/python/_vulkan_forge_editable.py → repo root = three up
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_activate()


# Optional: expose a noop attribute for *.pth* call style
def __call__() -> None:  # type: ignore
    _activate()
