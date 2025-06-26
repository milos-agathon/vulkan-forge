from setuptools import setup, find_packages
setup(
    name="vulkan-forge",
    version="0.1.0",
    description="Vulkan-powered 3-D height-map rendering for Python",
    author="Your Name",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
