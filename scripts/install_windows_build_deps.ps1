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

$installer = Join-Path $PWD "cuda_12.4.1_windows_network.exe"
Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/12.4.1/network_installers/cuda_12.4.1_windows_network.exe" -OutFile $installer
# https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-microsoft-windows/index.html
$components = @(
    "-s",
    "nvcc_12.4",
    "cudart_12.4",
    "documentation_12.4",
    "visual_studio_integration_12.4"
)
$process = Start-Process -FilePath $installer -ArgumentList $components -Wait -PassThru
if ($process.ExitCode -notin @(0, 3010)) {
    throw "CUDA installer failed with exit code $($process.ExitCode)"
}

$cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit/CUDA/v12.4"
foreach ($relativePath in @("include/crt/host_config.h", "nvvm/bin/cicc.exe", "nvvm/libdevice/libdevice.10.bc")) {
    $requiredFile = Join-Path $cudaRoot $relativePath
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "CUDA installation is incomplete: missing $requiredFile"
    }
}
