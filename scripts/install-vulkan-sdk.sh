#!/bin/bash
# ============================================================================
# Vulkan SDK Installer for Linux and macOS
# ============================================================================
# Comprehensive script to install Vulkan SDK on Linux and macOS systems
# Handles multiple distributions, package managers, and installation methods
# Supports both system-wide and user-local installations
# ============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# Configuration and Constants
# ============================================================================

# Default SDK versions
DEFAULT_VULKAN_VERSION="1.3.280"
DEFAULT_MACOS_VULKAN_VERSION="1.3.280.1"

# Installation directories
VULKAN_SDK_DIR="$HOME/VulkanSDK"
SYSTEM_VULKAN_PREFIX="/usr"

# Script configuration
SCRIPT_NAME="$(basename "$0")"
LOG_FILE="/tmp/vulkan-install-$(date +%Y%m%d-%H%M%S).log"
VERBOSE=false
FORCE_REINSTALL=false
USER_INSTALL=true
INSTALL_METHOD="auto"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo -e "${CYAN}[$(date +'%H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

verbose_log() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[VERBOSE]${NC} $*" | tee -a "$LOG_FILE"
    fi
}

show_help() {
    cat << EOF
${CYAN}Vulkan SDK Installer for Linux and macOS${NC}

${YELLOW}USAGE:${NC}
    $SCRIPT_NAME [OPTIONS]

${YELLOW}OPTIONS:${NC}
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -f, --force             Force reinstallation even if already installed
    -s, --system            Install system-wide (requires sudo)
    -u, --user              Install for current user only (default)
    -m, --method METHOD     Installation method: auto, package, manual, sdk
    --version VERSION       Vulkan SDK version (default: $DEFAULT_VULKAN_VERSION)
    --macos-version VERSION macOS Vulkan SDK version (default: $DEFAULT_MACOS_VULKAN_VERSION)
    --log-file FILE         Log file path (default: $LOG_FILE)

${YELLOW}INSTALLATION METHODS:${NC}
    auto        Automatically choose best method for platform
    package     Use system package manager (recommended for Linux)
    manual      Download and install SDK manually
    sdk         Install official Vulkan SDK (macOS default)

${YELLOW}EXAMPLES:${NC}
    $SCRIPT_NAME                                    # Auto-install for current user
    $SCRIPT_NAME --system --method package         # System-wide via package manager
    $SCRIPT_NAME --force --version 1.3.275         # Force install specific version
    $SCRIPT_NAME --verbose --method sdk            # Verbose SDK installation

${YELLOW}PLATFORMS SUPPORTED:${NC}
    - macOS (Intel and Apple Silicon)
    - Ubuntu/Debian (apt)
    - RHEL/CentOS/Rocky Linux (yum/dnf)
    - Fedora (dnf)
    - Arch Linux (pacman)
    - Alpine Linux (apk)
    - Generic Linux (manual installation)

EOF
}

detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

detect_linux_distro() {
    if [[ ! -f /etc/os-release ]]; then
        echo "unknown"
        return
    fi
    
    source /etc/os-release
    case "$ID" in
        ubuntu|debian)
            echo "debian"
            ;;
        rhel|centos|rocky|almalinux)
            echo "rhel"
            ;;
        fedora)
            echo "fedora"
            ;;
        arch|manjaro)
            echo "arch"
            ;;
        alpine)
            echo "alpine"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

check_vulkan_installation() {
    local sdk_path="$1"
    
    # Check for vulkaninfo
    local vulkaninfo=""
    if [[ -n "$sdk_path" && -f "$sdk_path/bin/vulkaninfo" ]]; then
        vulkaninfo="$sdk_path/bin/vulkaninfo"
    elif check_command vulkaninfo; then
        vulkaninfo="vulkaninfo"
    else
        return 1
    fi
    
    # Test vulkaninfo
    if "$vulkaninfo" --summary >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

get_macos_arch() {
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "arm64"
    else
        echo "x86_64"
    fi
}

# ============================================================================
# Installation Methods
# ============================================================================

install_vulkan_macos_sdk() {
    local version="$1"
    local install_dir="$2"
    
    log_info "Installing Vulkan SDK $version for macOS..."
    
    # Download URL
    local dmg_url="https://sdk.lunarg.com/sdk/download/${version}/mac/vulkansdk-macos-${version}.dmg"
    local dmg_file="/tmp/vulkansdk-macos-${version}.dmg"
    
    # Download DMG
    log_info "Downloading Vulkan SDK from: $dmg_url"
    if ! curl -L "$dmg_url" -o "$dmg_file"; then
        log_error "Failed to download Vulkan SDK"
        return 1
    fi
    
    # Mount DMG
    log_info "Mounting Vulkan SDK DMG..."
    local mount_point="/Volumes/vulkansdk-macos-${version}"
    if ! hdiutil attach "$dmg_file" -quiet; then
        log_error "Failed to mount DMG"
        rm -f "$dmg_file"
        return 1
    fi
    
    # Install SDK
    log_info "Installing Vulkan SDK to $install_dir..."
    local installer="$mount_point/InstallVulkan.app/Contents/MacOS/InstallVulkan"
    
    if [[ "$USER_INSTALL" == "true" ]]; then
        # User installation
        mkdir -p "$install_dir"
        if ! "$installer" --root "$install_dir" --accept-licenses --default-answer --confirm-command install; then
            log_error "Failed to install Vulkan SDK"
            hdiutil detach "$mount_point" -quiet || true
            rm -f "$dmg_file"
            return 1
        fi
    else
        # System installation
        if ! sudo "$installer" --accept-licenses --default-answer --confirm-command install; then
            log_error "Failed to install Vulkan SDK system-wide"
            hdiutil detach "$mount_point" -quiet || true
            rm -f "$dmg_file"
            return 1
        fi
    fi
    
    # Cleanup
    hdiutil detach "$mount_point" -quiet || true
    rm -f "$dmg_file"
    
    log_success "Vulkan SDK installed successfully"
    return 0
}

install_vulkan_linux_packages() {
    local distro="$1"
    
    log_info "Installing Vulkan packages for $distro..."
    
    case "$distro" in
        debian)
            log_info "Using apt package manager..."
            sudo apt-get update
            sudo apt-get install -y \
                libvulkan-dev \
                vulkan-tools \
                vulkan-validationlayers-dev \
                spirv-tools \
                glslang-tools \
                libglfw3-dev \
                libglm-dev \
                pkg-config
            ;;
        rhel)
            log_info "Using yum/dnf package manager..."
            if check_command dnf; then
                sudo dnf update -y
                sudo dnf install -y \
                    vulkan-devel \
                    vulkan-tools \
                    mesa-vulkan-devel \
                    spirv-tools \
                    glslang \
                    glfw-devel \
                    glm-devel \
                    pkgconfig
            else
                sudo yum update -y
                # Enable EPEL for additional packages
                sudo yum install -y epel-release
                sudo yum install -y \
                    vulkan-devel \
                    vulkan-tools \
                    mesa-vulkan-devel \
                    spirv-tools \
                    glslang \
                    glfw-devel \
                    pkgconfig
            fi
            ;;
        fedora)
            log_info "Using dnf package manager..."
            sudo dnf update -y
            sudo dnf install -y \
                vulkan-devel \
                vulkan-tools \
                mesa-vulkan-devel \
                spirv-tools \
                glslang \
                glfw-devel \
                glm-devel \
                pkgconfig
            ;;
        arch)
            log_info "Using pacman package manager..."
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                vulkan-devel \
                vulkan-tools \
                vulkan-validation-layers \
                spirv-tools \
                glslang \
                glfw-wayland \
                glm
            ;;
        alpine)
            log_info "Using apk package manager..."
            sudo apk update
            sudo apk add \
                vulkan-headers \
                vulkan-loader-dev \
                vulkan-tools \
                spirv-tools \
                glslang \
                glfw-dev \
                glm-dev
            ;;
        *)
            log_error "Unsupported Linux distribution: $distro"
            return 1
            ;;
    esac
    
    log_success "Vulkan packages installed successfully"
    return 0
}

install_vulkan_linux_sdk() {
    local version="$1"
    local install_dir="$2"
    
    log_info "Installing Vulkan SDK $version for Linux..."
    
    # Determine architecture
    local arch
    case "$(uname -m)" in
        x86_64)
            arch="x86_64"
            ;;
        aarch64|arm64)
            arch="aarch64"
            ;;
        *)
            log_error "Unsupported architecture: $(uname -m)"
            return 1
            ;;
    esac
    
    # Download URL
    local sdk_url="https://sdk.lunarg.com/sdk/download/${version}/linux/vulkansdk-linux-${arch}-${version}.tar.gz"
    local sdk_file="/tmp/vulkansdk-linux-${arch}-${version}.tar.gz"
    
    # Download SDK
    log_info "Downloading Vulkan SDK from: $sdk_url"
    if ! curl -L "$sdk_url" -o "$sdk_file"; then
        log_error "Failed to download Vulkan SDK"
        return 1
    fi
    
    # Extract SDK
    log_info "Extracting Vulkan SDK to $install_dir..."
    mkdir -p "$install_dir"
    if ! tar -xzf "$sdk_file" -C "$install_dir" --strip-components=1; then
        log_error "Failed to extract Vulkan SDK"
        rm -f "$sdk_file"
        return 1
    fi
    
    # Cleanup
    rm -f "$sdk_file"
    
    log_success "Vulkan SDK installed successfully"
    return 0
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    local vulkan_sdk_path="$1"
    local platform="$2"
    
    log_info "Setting up Vulkan environment..."
    
    # Determine shell configuration file
    local shell_rc=""
    if [[ -n "${BASH_VERSION:-}" ]]; then
        shell_rc="$HOME/.bashrc"
    elif [[ -n "${ZSH_VERSION:-}" ]]; then
        shell_rc="$HOME/.zshrc"
    elif [[ "$platform" == "macos" ]]; then
        # macOS default is zsh
        shell_rc="$HOME/.zshrc"
    else
        shell_rc="$HOME/.bashrc"
    fi
    
    # Create shell configuration if it doesn't exist
    touch "$shell_rc"
    
    # Remove existing Vulkan SDK configuration
    if grep -q "# Vulkan SDK Configuration" "$shell_rc" 2>/dev/null; then
        log_info "Removing existing Vulkan SDK configuration..."
        # Remove lines between markers
        sed -i.bak '/# Vulkan SDK Configuration - START/,/# Vulkan SDK Configuration - END/d' "$shell_rc"
    fi
    
    # Add new configuration
    log_info "Adding Vulkan SDK configuration to $shell_rc..."
    cat >> "$shell_rc" << EOF

# Vulkan SDK Configuration - START
# Added by vulkan-forge Vulkan SDK installer on $(date)
export VULKAN_SDK="$vulkan_sdk_path"
export PATH="\$VULKAN_SDK/bin:\$PATH"
export LD_LIBRARY_PATH="\$VULKAN_SDK/lib:\$LD_LIBRARY_PATH"
export VK_LAYER_PATH="\$VULKAN_SDK/etc/vulkan/explicit_layer.d"
export VK_ICD_FILENAMES="\$VULKAN_SDK/etc/vulkan/icd.d/lvp_icd.x86_64.json"
# Vulkan SDK Configuration - END
EOF
    
    # Set environment for current session
    export VULKAN_SDK="$vulkan_sdk_path"
    export PATH="$VULKAN_SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
    export VK_LAYER_PATH="$VULKAN_SDK/etc/vulkan/explicit_layer.d"
    
    log_success "Environment configured successfully"
    log_info "Restart your shell or run: source $shell_rc"
}

# ============================================================================
# Verification
# ============================================================================

verify_installation() {
    local vulkan_sdk_path="$1"
    
    log_info "Verifying Vulkan installation..."
    
    # Check SDK directory structure
    if [[ -n "$vulkan_sdk_path" && ! -d "$vulkan_sdk_path" ]]; then
        log_error "Vulkan SDK directory not found: $vulkan_sdk_path"
        return 1
    fi
    
    # Check for vulkaninfo
    local vulkaninfo=""
    if [[ -n "$vulkan_sdk_path" && -f "$vulkan_sdk_path/bin/vulkaninfo" ]]; then
        vulkaninfo="$vulkan_sdk_path/bin/vulkaninfo"
    elif check_command vulkaninfo; then
        vulkaninfo="vulkaninfo"
    else
        log_error "vulkaninfo not found"
        return 1
    fi
    
    # Test vulkaninfo
    log_info "Testing vulkaninfo..."
    if "$vulkaninfo" --summary > /tmp/vulkaninfo-test.log 2>&1; then
        log_success "vulkaninfo test passed"
        
        # Show brief summary
        if [[ "$VERBOSE" == "true" ]]; then
            echo "Vulkan Summary:"
            head -20 /tmp/vulkaninfo-test.log
        fi
    else
        log_warning "vulkaninfo test failed (this may be normal in headless environments)"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "vulkaninfo output:"
            cat /tmp/vulkaninfo-test.log
        fi
    fi
    
    # Check for glslc (shader compiler)
    local glslc=""
    if [[ -n "$vulkan_sdk_path" && -f "$vulkan_sdk_path/bin/glslc" ]]; then
        glslc="$vulkan_sdk_path/bin/glslc"
    elif check_command glslc; then
        glslc="glslc"
    fi
    
    if [[ -n "$glslc" ]]; then
        log_success "glslc (shader compiler) found"
        if [[ "$VERBOSE" == "true" ]]; then
            "$glslc" --version
        fi
    else
        log_warning "glslc not found (shader compilation may not work)"
    fi
    
    # Check for validation layers
    if "$vulkaninfo" | grep -q "VK_LAYER_KHRONOS_validation" 2>/dev/null; then
        log_success "Vulkan validation layers available"
    else
        log_warning "Vulkan validation layers not found"
    fi
    
    log_success "Vulkan installation verification completed"
    return 0
}

# ============================================================================
# Main Installation Logic
# ============================================================================

determine_install_method() {
    local platform="$1"
    local method="$2"
    
    if [[ "$method" != "auto" ]]; then
        echo "$method"
        return
    fi
    
    case "$platform" in
        macos)
            echo "sdk"
            ;;
        linux)
            # Prefer package manager for known distributions
            local distro
            distro="$(detect_linux_distro)"
            if [[ "$distro" != "unknown" ]]; then
                echo "package"
            else
                echo "manual"
            fi
            ;;
        *)
            echo "manual"
            ;;
    esac
}

install_vulkan() {
    local platform="$1"
    local method="$2"
    local version="$3"
    local macos_version="$4"
    
    log_info "Installing Vulkan SDK..."
    log_info "Platform: $platform"
    log_info "Method: $method"
    log_info "Version: $version"
    
    # Determine installation directory
    local install_dir
    if [[ "$USER_INSTALL" == "true" ]]; then
        case "$platform" in
            macos)
                install_dir="$VULKAN_SDK_DIR"
                ;;
            linux)
                install_dir="$VULKAN_SDK_DIR/$version"
                ;;
        esac
    else
        install_dir="$SYSTEM_VULKAN_PREFIX"
    fi
    
    # Check if already installed
    if [[ "$FORCE_REINSTALL" != "true" ]]; then
        case "$method" in
            package)
                if check_command vulkaninfo && vulkaninfo --summary >/dev/null 2>&1; then
                    log_success "Vulkan is already installed via system packages"
                    verify_installation ""
                    return 0
                fi
                ;;
            sdk|manual)
                if [[ -d "$install_dir" ]] && check_vulkan_installation "$install_dir"; then
                    log_success "Vulkan SDK is already installed at $install_dir"
                    setup_environment "$install_dir" "$platform"
                    verify_installation "$install_dir"
                    return 0
                fi
                ;;
        esac
    fi
    
    # Perform installation
    case "$method" in
        package)
            if [[ "$platform" != "linux" ]]; then
                log_error "Package installation only supported on Linux"
                return 1
            fi
            
            local distro
            distro="$(detect_linux_distro)"
            if ! install_vulkan_linux_packages "$distro"; then
                return 1
            fi
            
            # Verify package installation
            verify_installation ""
            ;;
            
        sdk)
            case "$platform" in
                macos)
                    if ! install_vulkan_macos_sdk "$macos_version" "$install_dir"; then
                        return 1
                    fi
                    setup_environment "$install_dir/macOS" "$platform"
                    verify_installation "$install_dir/macOS"
                    ;;
                linux)
                    if ! install_vulkan_linux_sdk "$version" "$install_dir"; then
                        return 1
                    fi
                    setup_environment "$install_dir" "$platform"
                    verify_installation "$install_dir"
                    ;;
                *)
                    log_error "SDK installation not supported on $platform"
                    return 1
                    ;;
            esac
            ;;
            
        manual)
            case "$platform" in
                linux)
                    if ! install_vulkan_linux_sdk "$version" "$install_dir"; then
                        return 1
                    fi
                    setup_environment "$install_dir" "$platform"
                    verify_installation "$install_dir"
                    ;;
                *)
                    log_error "Manual installation not implemented for $platform"
                    return 1
                    ;;
            esac
            ;;
            
        *)
            log_error "Unknown installation method: $method"
            return 1
            ;;
    esac
    
    log_success "Vulkan installation completed successfully!"
    return 0
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--force)
                FORCE_REINSTALL=true
                shift
                ;;
            -s|--system)
                USER_INSTALL=false
                shift
                ;;
            -u|--user)
                USER_INSTALL=true
                shift
                ;;
            -m|--method)
                INSTALL_METHOD="$2"
                shift 2
                ;;
            --version)
                DEFAULT_VULKAN_VERSION="$2"
                shift 2
                ;;
            --macos-version)
                DEFAULT_MACOS_VULKAN_VERSION="$2"
                shift 2
                ;;
            --log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# Main Function
# ============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Initialize logging
    log "Starting Vulkan SDK installation..."
    log "Log file: $LOG_FILE"
    
    # Detect platform
    local platform
    platform="$(detect_platform)"
    if [[ "$platform" == "unknown" ]]; then
        log_error "Unsupported platform: $OSTYPE"
        exit 1
    fi
    
    # Determine installation method
    local method
    method="$(determine_install_method "$platform" "$INSTALL_METHOD")"
    
    # Validate installation method
    case "$method" in
        package|sdk|manual)
            ;;
        *)
            log_error "Invalid installation method: $method"
            exit 1
            ;;
    esac
    
    # Check prerequisites
    if [[ "$method" == "sdk" || "$method" == "manual" ]]; then
        if ! check_command curl; then
            log_error "curl is required for SDK download"
            exit 1
        fi
    fi
    
    if [[ "$USER_INSTALL" == "false" && "$method" == "package" ]]; then
        if ! check_command sudo; then
            log_error "sudo is required for system-wide installation"
            exit 1
        fi
    fi
    
    # Perform installation
    if install_vulkan "$platform" "$method" "$DEFAULT_VULKAN_VERSION" "$DEFAULT_MACOS_VULKAN_VERSION"; then
        log_success "Vulkan SDK installation completed successfully!"
        echo ""
        echo -e "${GREEN}Next steps:${NC}"
        echo "1. Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
        echo "2. Test the installation: vulkaninfo --summary"
        echo "3. Install vulkan-forge: pip install vulkan-forge"
        echo ""
        echo -e "${BLUE}Installation log saved to: $LOG_FILE${NC}"
    else
        log_error "Vulkan SDK installation failed!"
        echo ""
        echo -e "${RED}Installation failed. Check the log for details: $LOG_FILE${NC}"
        exit 1
    fi
}

# ============================================================================
# Script Entry Point
# ============================================================================

# Ensure script is not being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi