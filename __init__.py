# This file helps with development - it redirects imports to the python subdirectory
import sys
from pathlib import Path

# Add the python directory to the path
python_dir = Path(__file__).parent / "python"
if python_dir.exists() and str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

# Now import from the actual package
from vulkan_forge import *