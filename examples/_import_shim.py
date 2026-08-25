# Shared import shim: put "from _import_shim import ensure_repo_import" at top of examples
import os
import sys
from pathlib import Path

def ensure_repo_import():
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    # Installed-wheel acceptance must exercise the installed package even when
    # an example is launched as a subprocess from the repository.
    if (
        os.environ.get("FORGE3D_TEST_INSTALLED_WHEEL") != "1"
        and python_dir.exists()
        and str(python_dir) not in sys.path
    ):
        sys.path.insert(0, str(python_dir))
    # Also include repo root for tools/scripts that import by absolute path
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
