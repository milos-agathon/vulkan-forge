# ./CMakeLists.txt
# Configures CMake-based build scaffolding for optional consumers
# Exists to mirror cargo/maturin targets for native integrators
# RELEVANT FILES:Cargo.toml,pyproject.toml,docs/conf.py,.github/workflows/build.yml
cmake_minimum_required(VERSION 3.20)
project(forge3d VERSION 1.35.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(FORGE3D_BUILD_PYTHON "Build Python bindings" ON)
option(FORGE3D_BUILD_EXAMPLES "Build example executables" OFF)
option(FORGE3D_BUILD_TESTS "Build tests" OFF)
option(FORGE3D_DEVELOPMENT_MODE "Enable development features" OFF)
option(FORGE3D_USE_SYSTEM_RUST "Use system Rust instead of bundled" ON)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Detect Rust toolchain
if(FORGE3D_USE_SYSTEM_RUST)
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    find_program(RUSTC_EXECUTABLE rustc REQUIRED)
else()
    # Use bundled Rust (not implemented - requires rustup download)
    message(FATAL_ERROR "Bundled Rust not implemented. Please install Rust and set FORGE3D_USE_SYSTEM_RUST=ON")
endif()

# Get Rust target triple for current platform
execute_process(
    COMMAND ${RUSTC_EXECUTABLE} --version --verbose
    OUTPUT_VARIABLE RUST_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCH "host: ([a-zA-Z0-9_-]+)" _ ${RUST_VERSION_OUTPUT})
set(RUST_TARGET_TRIPLE ${CMAKE_MATCH_1})

message(STATUS "Rust target triple: ${RUST_TARGET_TRIPLE}")

# Determine Rust library extension based on platform
if(WIN32)
    set(RUST_LIB_EXT "dll")
    set(RUST_LIB_PREFIX "")
    set(PYTHON_EXT_SUFFIX ".pyd")
elseif(APPLE)
    set(RUST_LIB_EXT "dylib") 
    set(RUST_LIB_PREFIX "lib")
    set(PYTHON_EXT_SUFFIX ".so")
else()
    set(RUST_LIB_EXT "so")
    set(RUST_LIB_PREFIX "lib")
    set(PYTHON_EXT_SUFFIX ".so")
endif()

# Set build type for Cargo
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_BUILD_TYPE "debug")
    set(CARGO_BUILD_FLAG "")
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target/debug")
else()
    set(CARGO_BUILD_TYPE "release")
    set(CARGO_BUILD_FLAG "--release")
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target/release")
endif()

# Rust library target
set(RUST_LIB_NAME "${RUST_LIB_PREFIX}forge3d${PYTHON_EXT_SUFFIX}")
set(RUST_LIB_PATH "${RUST_TARGET_DIR}/${RUST_LIB_NAME}")

# Custom command to build Rust library
add_custom_command(
    OUTPUT ${RUST_LIB_PATH}
    COMMAND ${CMAKE_COMMAND} -E env 
            "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/target"
            ${CARGO_EXECUTABLE} build ${CARGO_BUILD_FLAG}
            --target-dir ${CMAKE_CURRENT_BINARY_DIR}/target
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Rust library with Cargo"
    VERBATIM
)

# Custom target for Rust build
add_custom_target(forge3d_rust_build ALL
    DEPENDS ${RUST_LIB_PATH}
)

# Python bindings target (if enabled)
if(FORGE3D_BUILD_PYTHON)
    # Find Python site-packages directory
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Install Python package files
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/forge3d/
        DESTINATION ${PYTHON_SITE_PACKAGES}/forge3d/
        FILES_MATCHING 
        PATTERN "*.py"
        PATTERN "*.pyi"
    )

    # Install Rust extension module
    install(
        FILES ${RUST_LIB_PATH}
        DESTINATION ${PYTHON_SITE_PACKAGES}/forge3d/
        RENAME "_forge3d${PYTHON_EXT_SUFFIX}"
    )

    # Install data files
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
        install(
            DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
            DESTINATION ${PYTHON_SITE_PACKAGES}/forge3d/data/
            FILES_MATCHING
            PATTERN "*.rgba"
            PATTERN "*.txt"
        )
    endif()

    # Install assets
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/assets)
        install(
            DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets/
            DESTINATION ${PYTHON_SITE_PACKAGES}/forge3d/assets/
            FILES_MATCHING
            PATTERN "*.png"
            PATTERN "*.json"
        )
    endif()

    # Development install target
    add_custom_target(python_dev_install
        COMMAND ${Python3_EXECUTABLE} -m pip install -e .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Installing Python package in development mode"
        DEPENDS forge3d_rust_build
    )
endif()

# Examples (if enabled)
if(FORGE3D_BUILD_EXAMPLES)
    # Build Rust examples
    add_custom_target(examples
        COMMAND ${CMAKE_COMMAND} -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/target"
                ${CARGO_EXECUTABLE} build ${CARGO_BUILD_FLAG} --examples
                --target-dir ${CMAKE_CURRENT_BINARY_DIR}/target
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust examples"
        VERBATIM
    )
endif()

# Tests (if enabled)
if(FORGE3D_BUILD_TESTS)
    enable_testing()

    # Rust tests
    add_test(
        NAME rust_tests
        COMMAND ${CMAKE_COMMAND} -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/target"
                ${CARGO_EXECUTABLE} test ${CARGO_BUILD_FLAG}
                --target-dir ${CMAKE_CURRENT_BINARY_DIR}/target
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Python tests (requires Python bindings)
    if(FORGE3D_BUILD_PYTHON)
        add_test(
            NAME python_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest tests/ -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_tests_properties(python_tests PROPERTIES
            DEPENDS python_dev_install
        )
    endif()
endif()

# Development targets
if(FORGE3D_DEVELOPMENT_MODE)
    # Format Rust code
    add_custom_target(format
        COMMAND ${CARGO_EXECUTABLE} fmt
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting Rust code"
    )

    # Lint Rust code
    add_custom_target(clippy
        COMMAND ${CARGO_EXECUTABLE} clippy -- -D warnings
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running Clippy linter"
    )

    # Clean target
    add_custom_target(clean_rust
        COMMAND ${CARGO_EXECUTABLE} clean
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Cleaning Rust build artifacts"
    )
endif()

# Documentation
add_custom_target(docs
    COMMAND ${CARGO_EXECUTABLE} doc --no-deps
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating Rust documentation"
)

# Print configuration summary
message(STATUS "")
message(STATUS "forge3d CMake Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Rust target: ${RUST_TARGET_TRIPLE}")
message(STATUS "  Python bindings: ${FORGE3D_BUILD_PYTHON}")
message(STATUS "  Examples: ${FORGE3D_BUILD_EXAMPLES}")
message(STATUS "  Tests: ${FORGE3D_BUILD_TESTS}")
message(STATUS "  Development mode: ${FORGE3D_DEVELOPMENT_MODE}")
if(FORGE3D_BUILD_PYTHON)
    message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "  Python site-packages: ${PYTHON_SITE_PACKAGES}")
endif()
message(STATUS "")

# Install configuration files
install(
    FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE  
        ${CMAKE_CURRENT_SOURCE_DIR}/CHANGELOG.md
    DESTINATION share/doc/forge3d/
    OPTIONAL
)

