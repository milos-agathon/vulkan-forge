"""
Simple setup.py for vulkan-forge
"""

from setuptools import setup, find_packages
import os

# Read README if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="vulkan-forge",
    version="0.1.0",
    author="VulkanForge Team",
    author_email="team@vulkanforge.dev",
    description="High-performance GPU renderer for height fields using Vulkan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vulkanforge/vulkan-forge",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0,<3.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "psutil>=5.9.0",
        ],
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
)