#!/usr/bin/env python3
"""Run working tests and show summary"""

import subprocess
import sys

def run_tests():
    """Run tests and categorize results"""
    
    print("Running Vulkan-Forge Tests")
    print("=" * 60)
    
    # Run tests and capture output
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    output = result.stdout + result.stderr
    
    # Count results
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    errors = output.count(" ERROR")
    skipped = output.count(" SKIPPED")
    
    print(f"\nTest Summary:")
    print(f"  ✓ PASSED:  {passed}")
    print(f"  ✗ FAILED:  {failed}")
    print(f"  ! ERROR:   {errors}")
    print(f"  - SKIPPED: {skipped}")
    print(f"  TOTAL:     {passed + failed + errors + skipped}")
    
    print(f"\nSuccess Rate: {passed / (passed + failed + errors) * 100:.1f}%")
    
    if failed > 0 or errors > 0:
        print("\nKnown issues:")
        if "render_cpu" in output:
            print("  - CPU rendering tests (missing implementation)")
        if "render_indexed" in output:
            print("  - Indexed rendering tests (missing implementation)")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
