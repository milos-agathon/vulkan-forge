#!/bin/bash

# Tracy Setup Script for Vulkan-Forge
# This script initializes the Tracy submodule and sets it to a stable version
#
# Usage: chmod +x setup_tracy.sh && ./setup_tracy.sh

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRACY_DIR="$SCRIPT_DIR/upstream"

echo "🔧 Setting up Tracy profiler for Vulkan-Forge..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Navigate to repo root
cd "$REPO_ROOT"

# Add Tracy submodule if it doesn't exist
if [ ! -d "$TRACY_DIR/.git" ]; then
    echo "📥 Adding Tracy submodule..."
    git submodule add https://github.com/wolfpld/tracy.git third_party/tracy/upstream
else
    echo "✅ Tracy submodule already exists"
fi

# Initialize and update submodules
echo "🔄 Updating submodules..."
git submodule update --init --recursive

# Switch to stable version
cd "$TRACY_DIR"

# Get latest stable version (v0.10.x series)
STABLE_VERSION=$(git tag -l "v0.10*" | sort -V | tail -1)

if [ -z "$STABLE_VERSION" ]; then
    echo "⚠️  Warning: No v0.10.x tags found, using v0.10"
    STABLE_VERSION="v0.10"
fi

echo "📌 Switching to stable version: $STABLE_VERSION"
git checkout "$STABLE_VERSION" 2>/dev/null || {
    echo "⚠️  Warning: Could not checkout $STABLE_VERSION, fetching latest..."
    git fetch origin
    git checkout "$STABLE_VERSION" || {
        echo "❌ Error: Could not checkout $STABLE_VERSION"
        echo "Available tags:"
        git tag -l "v0.*" | sort -V | tail -10
        exit 1
    }
}

# Return to repo root and commit the submodule update
cd "$REPO_ROOT"
git add third_party/tracy/upstream

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "✅ Tracy submodule already up to date"
else
    echo "💾 Committing Tracy submodule update..."
    git commit -m "Add/update Tracy profiler submodule to $STABLE_VERSION"
fi

echo ""
echo "🎉 Tracy setup complete!"
echo ""
echo "Next steps:"
echo "  1. Build with profiling: cmake -B build -DVULKAN_FORGE_ENABLE_PROFILING=ON"
echo "  2. Add profiling to your code: #include \"third_party/tracy/vulkan_forge_tracy.hpp\""
echo "  3. Download Tracy profiler GUI from: https://github.com/wolfpld/tracy/releases"
echo ""
echo "Tracy version: $STABLE_VERSION"
echo "Tracy directory: $TRACY_DIR"
echo ""

# Show configuration summary
echo "Configuration:"
if [ -f "$SCRIPT_DIR/tracy_config.hpp" ]; then
    echo "  ✅ tracy_config.hpp"
else
    echo "  ❌ tracy_config.hpp (missing)"
fi

if [ -f "$SCRIPT_DIR/vulkan_forge_tracy.hpp" ]; then
    echo "  ✅ vulkan_forge_tracy.hpp"
else
    echo "  ❌ vulkan_forge_tracy.hpp (missing)"
fi

if [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
    echo "  ✅ CMakeLists.txt"
else
    echo "  ❌ CMakeLists.txt (missing)"
fi

echo ""
echo "📖 See third_party/tracy/README.md for usage examples"