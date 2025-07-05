#!/usr/bin/env python
"""Helper script to run examples with proper Python path."""

import sys
import os

# Add the python directory to Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(root_dir, 'python')
sys.path.insert(0, python_dir)

# Now import and run the example
if __name__ == "__main__":
    # Import the example module
    # Change to the script's directory so relative imports work
    example_path = os.path.join(root_dir, 'examples', '02_numpy_pointcloud.py')
    
    # Execute the example file
    with open(example_path, 'r') as f:
        code = compile(f.read(), example_path, 'exec')
        exec(code, {'__file__': example_path, '__name__': '__main__'})
    
    # Run the main function
    example.main()