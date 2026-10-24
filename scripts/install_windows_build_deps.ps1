param([Parameter(Mandatory = $true)][string]$PackageDir)

$ErrorActionPreference = "Stop"

nuget install intelopenmp.devel.win -DirectDownload -NonInteractive
if ($LASTEXITCODE -ne 0) { throw "Intel OpenMP development package install failed" }
nuget install intelopenmp.redist.win -DirectDownload -NonInteractive
if ($LASTEXITCODE -ne 0) { throw "Intel OpenMP runtime package install failed" }

Copy-Item "intelopenmp.devel.win*/build/native/win-x64/libiomp5md.lib" "$PackageDir/src/deepwave/"
Copy-Item "intelopenmp.redist.win*/runtimes/win-x64/native/libiomp5md.dll" "$PackageDir/src/deepwave/"
New-Item -ItemType Directory -Force "$PackageDir/runtime-notices" | Out-Null
Copy-Item "intelopenmp.redist.win*/license.txt" "$PackageDir/runtime-notices/INTEL_OPENMP_LICENSE.txt"

$installer = Join-Path $PWD "cuda_13.3.1_windows_network.exe"
Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/13.3.1/network_installers/cuda_13.3.1_windows_network.exe" -OutFile $installer
# CUDA 13 packages the compiler headers and device compiler separately from
# nvcc. Selecting nvcc/cudart alone leaves CMake's compiler probe incomplete.
# https://docs.nvidia.com/cuda/archive/13.3.1/cuda-installation-guide-microsoft-windows/index.html
$components = @(
    "-s",
    "nvcc_13.3",
    "crt_13.3",
    "nvvm_13.3",
    "cudart_13.3",
    "documentation_13.3",
    "visual_studio_integration_13.3"
)
$process = Start-Process -FilePath $installer -ArgumentList $components -Wait -PassThru
if ($process.ExitCode -notin @(0, 3010)) {
    throw "CUDA installer failed with exit code $($process.ExitCode)"
}

$cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit/CUDA/v13.3"
foreach ($relativePath in @("include/crt/host_config.h", "nvvm/bin/cicc.exe", "nvvm/libdevice/libdevice.10.bc")) {
    $requiredFile = Join-Path $cudaRoot $relativePath
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "CUDA installation is incomplete: missing $requiredFile (required by crt_13.3/nvvm_13.3)"
    }
}
