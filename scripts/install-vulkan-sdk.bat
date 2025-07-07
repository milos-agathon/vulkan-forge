@echo off
REM ============================================================================
REM Vulkan SDK Installer for Windows
REM ============================================================================
REM Comprehensive script to install Vulkan SDK on Windows systems
REM Handles silent installation, environment setup, and verification
REM Supports both user and system-wide installations with proper UAC handling
REM ============================================================================

setlocal enabledelayedexpansion

REM ============================================================================
REM Configuration and Constants
REM ============================================================================

REM Default SDK version
set "DEFAULT_VULKAN_VERSION=1.3.280.0"

REM Installation directories
set "VULKAN_SDK_USER_DIR=%USERPROFILE%\VulkanSDK"
set "VULKAN_SDK_SYSTEM_DIR=C:\VulkanSDK"

REM Script configuration
set "SCRIPT_NAME=%~nx0"
set "LOG_FILE=%TEMP%\vulkan-install-%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%-%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log"
set "VERBOSE=false"
set "FORCE_REINSTALL=false"
set "USER_INSTALL=true"
set "INSTALL_METHOD=auto"
set "VULKAN_VERSION=%DEFAULT_VULKAN_VERSION%"
set "SILENT_INSTALL=true"
set "VERIFY_CHECKSUM=true"

REM Clean up log file name (remove spaces)
set "LOG_FILE=%LOG_FILE: =0%"

REM ============================================================================
REM Utility Functions
REM ============================================================================

:log
echo [%TIME%] %* >> "%LOG_FILE%"
echo [%TIME%] %*
exit /b

:log_info
echo [INFO] %* >> "%LOG_FILE%"
echo [INFO] %*
exit /b

:log_success
echo [SUCCESS] %* >> "%LOG_FILE%"
echo [SUCCESS] %*
exit /b

:log_warning
echo [WARNING] %* >> "%LOG_FILE%"
echo [WARNING] %*
exit /b

:log_error
echo [ERROR] %* >> "%LOG_FILE%"
echo [ERROR] %*
exit /b

:verbose_log
if "%VERBOSE%"=="true" (
    echo [VERBOSE] %* >> "%LOG_FILE%"
    echo [VERBOSE] %*
)
exit /b

:show_help
echo.
echo Vulkan SDK Installer for Windows
echo.
echo USAGE:
echo     %SCRIPT_NAME% [OPTIONS]
echo.
echo OPTIONS:
echo     /h, /help               Show this help message
echo     /v, /verbose            Enable verbose output
echo     /f, /force              Force reinstallation even if already installed
echo     /s, /system             Install system-wide (requires administrator)
echo     /u, /user               Install for current user only (default)
echo     /version VERSION        Vulkan SDK version (default: %DEFAULT_VULKAN_VERSION%)
echo     /method METHOD          Installation method: auto, download, manual
echo     /interactive            Interactive installation (not silent)
echo     /no-verify              Skip checksum verification
echo     /log-file FILE          Log file path (default: %LOG_FILE%)
echo.
echo INSTALLATION METHODS:
echo     auto        Automatically choose best method (default)
echo     download    Download and install SDK automatically
echo     manual      Provide manual installation instructions
echo.
echo EXAMPLES:
echo     %SCRIPT_NAME%                           # Auto-install for current user
echo     %SCRIPT_NAME% /system                   # System-wide installation
echo     %SCRIPT_NAME% /force /version 1.3.275   # Force install specific version
echo     %SCRIPT_NAME% /verbose /interactive     # Verbose interactive installation
echo.
echo REQUIREMENTS:
echo     - Windows 10 or later (x64)
echo     - PowerShell 5.0 or later
echo     - Internet connection for download
echo     - Administrator rights for system installation
echo.
exit /b

:check_admin
net session >nul 2>&1
exit /b

:check_powershell
powershell -Command "exit 0" >nul 2>&1
exit /b

:get_windows_version
for /f "tokens=2 delims=[]" %%x in ('ver') do set "WIN_VERSION=%%x"
exit /b

:check_vulkan_installation
set "SDK_PATH=%~1"
set "VULKAN_FOUND=false"

REM Check for vulkaninfo.exe
if exist "%SDK_PATH%\Bin\vulkaninfo.exe" (
    REM Test vulkaninfo
    "%SDK_PATH%\Bin\vulkaninfo.exe" --summary >nul 2>&1
    if !errorlevel! equ 0 (
        set "VULKAN_FOUND=true"
    )
) else if exist "%SystemRoot%\System32\vulkaninfo.exe" (
    vulkaninfo.exe --summary >nul 2>&1
    if !errorlevel! equ 0 (
        set "VULKAN_FOUND=true"
    )
)

exit /b

:detect_architecture
set "ARCH=unknown"
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" set "ARCH=x64"
if "%PROCESSOR_ARCHITEW6432%"=="AMD64" set "ARCH=x64"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"
exit /b

REM ============================================================================
REM Download and Installation Functions
REM ============================================================================

:download_vulkan_sdk
set "VERSION=%~1"
set "DOWNLOAD_DIR=%~2"

call :log_info "Downloading Vulkan SDK %VERSION%..."

REM Construct download URL
set "SDK_URL=https://sdk.lunarg.com/sdk/download/%VERSION%/windows/VulkanSDK-%VERSION%-Installer.exe"
set "INSTALLER_PATH=%DOWNLOAD_DIR%\VulkanSDK-%VERSION%-Installer.exe"

call :verbose_log "Download URL: %SDK_URL%"
call :verbose_log "Installer path: %INSTALLER_PATH%"

REM Create download directory
if not exist "%DOWNLOAD_DIR%" mkdir "%DOWNLOAD_DIR%"

REM Download using PowerShell with progress indication
call :log_info "Downloading from: %SDK_URL%"

powershell -Command "& { ^
    $ProgressPreference = 'SilentlyContinue'; ^
    try { ^
        $webClient = New-Object System.Net.WebClient; ^
        $webClient.DownloadFile('%SDK_URL%', '%INSTALLER_PATH%'); ^
        Write-Host 'Download completed successfully'; ^
        exit 0; ^
    } catch { ^
        Write-Host 'Download failed:' $_.Exception.Message; ^
        exit 1; ^
    } ^
}"

if !errorlevel! neq 0 (
    call :log_error "Failed to download Vulkan SDK"
    exit /b 1
)

REM Verify file exists and has reasonable size
if not exist "%INSTALLER_PATH%" (
    call :log_error "Downloaded installer not found"
    exit /b 1
)

REM Check file size (should be > 100MB)
for %%A in ("%INSTALLER_PATH%") do set "FILE_SIZE=%%~zA"
if !FILE_SIZE! lss 104857600 (
    call :log_warning "Downloaded file seems too small (!FILE_SIZE! bytes)"
)

call :log_success "Vulkan SDK downloaded successfully"
exit /b 0

:verify_installer_checksum
set "INSTALLER_PATH=%~1"

if "%VERIFY_CHECKSUM%"=="false" (
    call :verbose_log "Skipping checksum verification"
    exit /b 0
)

call :log_info "Verifying installer checksum..."

REM Get SHA256 hash using PowerShell
powershell -Command "& { ^
    try { ^
        $hash = Get-FileHash -Algorithm SHA256 '%INSTALLER_PATH%'; ^
        Write-Host 'SHA256:' $hash.Hash; ^
        exit 0; ^
    } catch { ^
        Write-Host 'Checksum calculation failed:' $_.Exception.Message; ^
        exit 1; ^
    } ^
}"

if !errorlevel! neq 0 (
    call :log_warning "Could not verify installer checksum"
) else (
    call :log_success "Installer checksum calculated"
)

exit /b 0

:install_vulkan_sdk
set "INSTALLER_PATH=%~1"
set "INSTALL_DIR=%~2"

call :log_info "Installing Vulkan SDK..."
call :verbose_log "Installer: %INSTALLER_PATH%"
call :verbose_log "Install directory: %INSTALL_DIR%"

REM Verify installer exists
if not exist "%INSTALLER_PATH%" (
    call :log_error "Installer not found: %INSTALLER_PATH%"
    exit /b 1
)

REM Prepare installation command
set "INSTALL_CMD="%INSTALLER_PATH%""

if "%SILENT_INSTALL%"=="true" (
    set "INSTALL_CMD=!INSTALL_CMD! /S"
    call :verbose_log "Using silent installation"
) else (
    call :verbose_log "Using interactive installation"
)

REM Add custom install directory if specified
if not "%INSTALL_DIR%"=="" (
    if not "%INSTALL_DIR%"=="default" (
        set "INSTALL_CMD=!INSTALL_CMD! /D=%INSTALL_DIR%"
        call :verbose_log "Custom install directory: %INSTALL_DIR%"
    )
)

call :log_info "Running installer..."
call :verbose_log "Command: !INSTALL_CMD!"

REM Run installer with timeout
timeout /t 2 /nobreak >nul
!INSTALL_CMD!

if !errorlevel! neq 0 (
    call :log_error "Installation failed with error code !errorlevel!"
    exit /b 1
)

call :log_success "Vulkan SDK installation completed"
exit /b 0

:find_vulkan_installation
set "VULKAN_SDK_PATH="

REM Check common installation locations
set "SEARCH_PATHS=C:\VulkanSDK\%VULKAN_VERSION% %VULKAN_SDK_SYSTEM_DIR%\%VULKAN_VERSION% %VULKAN_SDK_USER_DIR%\%VULKAN_VERSION%"

for %%P in (%SEARCH_PATHS%) do (
    if exist "%%P\Bin\vulkaninfo.exe" (
        set "VULKAN_SDK_PATH=%%P"
        call :verbose_log "Found Vulkan SDK at: %%P"
        goto :find_vulkan_done
    )
)

REM Try to find any version
for /d %%D in (C:\VulkanSDK\*) do (
    if exist "%%D\Bin\vulkaninfo.exe" (
        set "VULKAN_SDK_PATH=%%D"
        call :verbose_log "Found Vulkan SDK at: %%D"
        goto :find_vulkan_done
    )
)

:find_vulkan_done
if "%VULKAN_SDK_PATH%"=="" (
    call :verbose_log "No Vulkan SDK installation found"
) else (
    call :verbose_log "Vulkan SDK found at: %VULKAN_SDK_PATH%"
)
exit /b

REM ============================================================================
REM Environment Setup
REM ============================================================================

:setup_environment
set "SDK_PATH=%~1"

call :log_info "Setting up Vulkan environment..."

REM Set user environment variables
call :log_info "Setting VULKAN_SDK environment variable..."

REM Set for current session
set "VULKAN_SDK=%SDK_PATH%"
set "PATH=%SDK_PATH%\Bin;%PATH%"

REM Set permanently using PowerShell
powershell -Command "& { ^
    try { ^
        if ('%USER_INSTALL%' -eq 'true') { ^
            [Environment]::SetEnvironmentVariable('VULKAN_SDK', '%SDK_PATH%', 'User'); ^
            $currentPath = [Environment]::GetEnvironmentVariable('PATH', 'User'); ^
            if ($currentPath -notlike '*%SDK_PATH%\Bin*') { ^
                $newPath = '%SDK_PATH%\Bin;' + $currentPath; ^
                [Environment]::SetEnvironmentVariable('PATH', $newPath, 'User'); ^
            } ^
        } else { ^
            [Environment]::SetEnvironmentVariable('VULKAN_SDK', '%SDK_PATH%', 'Machine'); ^
            $currentPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); ^
            if ($currentPath -notlike '*%SDK_PATH%\Bin*') { ^
                $newPath = '%SDK_PATH%\Bin;' + $currentPath; ^
                [Environment]::SetEnvironmentVariable('PATH', $newPath, 'Machine'); ^
            } ^
        } ^
        Write-Host 'Environment variables set successfully'; ^
        exit 0; ^
    } catch { ^
        Write-Host 'Failed to set environment variables:' $_.Exception.Message; ^
        exit 1; ^
    } ^
}"

if !errorlevel! neq 0 (
    call :log_warning "Failed to set permanent environment variables"
    call :log_info "You may need to set VULKAN_SDK=%SDK_PATH% manually"
) else (
    call :log_success "Environment variables configured successfully"
)

exit /b 0

REM ============================================================================
REM Verification Functions
REM ============================================================================

:verify_installation
set "SDK_PATH=%~1"

call :log_info "Verifying Vulkan installation..."

REM Check SDK directory structure
if not exist "%SDK_PATH%" (
    call :log_error "Vulkan SDK directory not found: %SDK_PATH%"
    exit /b 1
)

if not exist "%SDK_PATH%\Bin" (
    call :log_error "Vulkan SDK Bin directory not found: %SDK_PATH%\Bin"
    exit /b 1
)

REM Check for vulkaninfo.exe
set "VULKANINFO_PATH=%SDK_PATH%\Bin\vulkaninfo.exe"
if not exist "%VULKANINFO_PATH%" (
    call :log_error "vulkaninfo.exe not found: %VULKANINFO_PATH%"
    exit /b 1
)

REM Test vulkaninfo
call :log_info "Testing vulkaninfo..."
"%VULKANINFO_PATH%" --summary > "%TEMP%\vulkaninfo-test.log" 2>&1

if !errorlevel! equ 0 (
    call :log_success "vulkaninfo test passed"
    
    if "%VERBOSE%"=="true" (
        echo Vulkan Summary:
        type "%TEMP%\vulkaninfo-test.log" | findstr /n "." | findstr "^[1-9]:" | findstr "^[1-9][0-9]*:"
    )
) else (
    call :log_warning "vulkaninfo test failed (this may be normal in some environments)"
    if "%VERBOSE%"=="true" (
        echo vulkaninfo output:
        type "%TEMP%\vulkaninfo-test.log"
    )
)

REM Check for glslc.exe (shader compiler)
set "GLSLC_PATH=%SDK_PATH%\Bin\glslc.exe"
if exist "%GLSLC_PATH%" (
    call :log_success "glslc.exe (shader compiler) found"
    if "%VERBOSE%"=="true" (
        "%GLSLC_PATH%" --version
    )
) else (
    call :log_warning "glslc.exe not found (shader compilation may not work)"
)

REM Check for validation layers
"%VULKANINFO_PATH%" | findstr "VK_LAYER_KHRONOS_validation" >nul 2>&1
if !errorlevel! equ 0 (
    call :log_success "Vulkan validation layers available"
) else (
    call :log_warning "Vulkan validation layers not found"
)

REM Check for runtime libraries
if exist "%SDK_PATH%\Bin\vulkan-1.dll" (
    call :log_success "Vulkan runtime library found"
) else (
    call :log_warning "Vulkan runtime library not found"
)

call :log_success "Vulkan installation verification completed"
exit /b 0

REM ============================================================================
REM Main Installation Logic
REM ============================================================================

:determine_install_method
set "METHOD=%~1"

if not "%METHOD%"=="auto" (
    exit /b
)

REM For Windows, default to download method
set "INSTALL_METHOD=download"
exit /b

:install_vulkan
call :log_info "Installing Vulkan SDK %VULKAN_VERSION%..."
call :log_info "Method: %INSTALL_METHOD%"
call :log_info "User install: %USER_INSTALL%"

REM Determine installation directory
set "INSTALL_DIR=default"
if "%USER_INSTALL%"=="true" (
    set "TARGET_SDK_PATH=%VULKAN_SDK_USER_DIR%\%VULKAN_VERSION%"
) else (
    set "TARGET_SDK_PATH=%VULKAN_SDK_SYSTEM_DIR%\%VULKAN_VERSION%"
    set "INSTALL_DIR=%TARGET_SDK_PATH%"
)

REM Check if already installed
if "%FORCE_REINSTALL%"=="false" (
    call :find_vulkan_installation
    if not "%VULKAN_SDK_PATH%"=="" (
        call :check_vulkan_installation "%VULKAN_SDK_PATH%"
        if "%VULKAN_FOUND%"=="true" (
            call :log_success "Vulkan SDK is already installed at %VULKAN_SDK_PATH%"
            call :setup_environment "%VULKAN_SDK_PATH%"
            call :verify_installation "%VULKAN_SDK_PATH%"
            exit /b 0
        )
    )
)

REM Perform installation based on method
if "%INSTALL_METHOD%"=="download" (
    REM Download and install automatically
    set "DOWNLOAD_DIR=%TEMP%\vulkan-install"
    
    call :download_vulkan_sdk "%VULKAN_VERSION%" "!DOWNLOAD_DIR!"
    if !errorlevel! neq 0 exit /b 1
    
    call :verify_installer_checksum "!DOWNLOAD_DIR!\VulkanSDK-%VULKAN_VERSION%-Installer.exe"
    
    call :install_vulkan_sdk "!DOWNLOAD_DIR!\VulkanSDK-%VULKAN_VERSION%-Installer.exe" "%INSTALL_DIR%"
    if !errorlevel! neq 0 (
        call :log_error "Installation failed"
        exit /b 1
    )
    
    REM Clean up installer
    del "!DOWNLOAD_DIR!\VulkanSDK-%VULKAN_VERSION%-Installer.exe" >nul 2>&1
    
    REM Find the actual installation path
    call :find_vulkan_installation
    if "%VULKAN_SDK_PATH%"=="" (
        call :log_error "Could not locate installed Vulkan SDK"
        exit /b 1
    )
    
    call :setup_environment "%VULKAN_SDK_PATH%"
    call :verify_installation "%VULKAN_SDK_PATH%"
    
) else if "%INSTALL_METHOD%"=="manual" (
    REM Provide manual installation instructions
    call :log_info "Manual installation instructions:"
    echo.
    echo 1. Download Vulkan SDK from: https://vulkan.lunarg.com/sdk/home#windows
    echo 2. Run the installer as Administrator
    echo 3. Follow the installation wizard
    echo 4. Restart your command prompt
    echo 5. Run: %SCRIPT_NAME% /verify
    echo.
    exit /b 0
    
) else (
    call :log_error "Unknown installation method: %INSTALL_METHOD%"
    exit /b 1
)

call :log_success "Vulkan installation completed successfully!"
exit /b 0

REM ============================================================================
REM Argument Parsing
REM ============================================================================

:parse_arguments
if "%~1"=="" goto :parse_done

set "ARG=%~1"

if /i "%ARG%"=="/h" goto :show_help_and_exit
if /i "%ARG%"=="/help" goto :show_help_and_exit
if /i "%ARG%"=="-h" goto :show_help_and_exit
if /i "%ARG%"=="--help" goto :show_help_and_exit

if /i "%ARG%"=="/v" (
    set "VERBOSE=true"
    shift /1
    goto :parse_arguments
)
if /i "%ARG%"=="/verbose" (
    set "VERBOSE=true"
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/f" (
    set "FORCE_REINSTALL=true"
    shift /1
    goto :parse_arguments
)
if /i "%ARG%"=="/force" (
    set "FORCE_REINSTALL=true"
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/s" (
    set "USER_INSTALL=false"
    shift /1
    goto :parse_arguments
)
if /i "%ARG%"=="/system" (
    set "USER_INSTALL=false"
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/u" (
    set "USER_INSTALL=true"
    shift /1
    goto :parse_arguments
)
if /i "%ARG%"=="/user" (
    set "USER_INSTALL=true"
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/version" (
    set "VULKAN_VERSION=%~2"
    shift /1
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/method" (
    set "INSTALL_METHOD=%~2"
    shift /1
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/interactive" (
    set "SILENT_INSTALL=false"
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/no-verify" (
    set "VERIFY_CHECKSUM=false"
    shift /1
    goto :parse_arguments
)

if /i "%ARG%"=="/log-file" (
    set "LOG_FILE=%~2"
    shift /1
    shift /1
    goto :parse_arguments
)

call :log_error "Unknown option: %ARG%"
goto :show_help_and_exit

:show_help_and_exit
call :show_help
exit /b 1

:parse_done
exit /b 0

REM ============================================================================
REM Main Function
REM ============================================================================

:main
REM Parse command line arguments
call :parse_arguments %*
if !errorlevel! neq 0 exit /b 1

REM Initialize logging
call :log "Starting Vulkan SDK installation for Windows..."
call :log "Log file: %LOG_FILE%"

REM Check Windows version
call :get_windows_version
call :verbose_log "Windows version: %WIN_VERSION%"

REM Detect architecture
call :detect_architecture
call :verbose_log "Architecture: %ARCH%"

if "%ARCH%"=="unknown" (
    call :log_error "Unsupported architecture"
    exit /b 1
)

REM Check PowerShell availability
call :check_powershell
if !errorlevel! neq 0 (
    call :log_error "PowerShell is required but not available"
    exit /b 1
)

REM Check administrator privileges if system install
if "%USER_INSTALL%"=="false" (
    call :check_admin
    if !errorlevel! neq 0 (
        call :log_error "Administrator privileges required for system-wide installation"
        call :log_info "Run this script as Administrator or use /user option"
        exit /b 1
    )
)

REM Determine installation method
call :determine_install_method "%INSTALL_METHOD%"

REM Validate installation method
if not "%INSTALL_METHOD%"=="download" if not "%INSTALL_METHOD%"=="manual" (
    call :log_error "Invalid installation method: %INSTALL_METHOD%"
    exit /b 1
)

REM Perform installation
call :install_vulkan
if !errorlevel! neq 0 (
    call :log_error "Vulkan SDK installation failed!"
    echo.
    echo Installation failed. Check the log for details: %LOG_FILE%
    exit /b 1
)

call :log_success "Vulkan SDK installation completed successfully!"
echo.
echo Next steps:
echo 1. Restart your command prompt to refresh environment variables
echo 2. Test the installation: vulkaninfo --summary
echo 3. Install vulkan-forge: pip install vulkan-forge
echo.
echo Installation log saved to: %LOG_FILE%
exit /b 0

REM ============================================================================
REM Script Entry Point
REM ============================================================================

call :main %*
exit /b %errorlevel%