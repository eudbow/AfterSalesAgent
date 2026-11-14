# Explicit local preparation. Run from any directory; does not change system configuration.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot '.local/conda-envs/aftersales-rag'
$env:CONDA_PKGS_DIRS = Join-Path $projectRoot '.local/conda-pkgs'
$env:CONDA_ENVS_PATH = Join-Path $projectRoot '.local/conda-envs'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.local/pip-cache'
$env:PYTHONUTF8 = '1'
$env:CONDA_SSL_VERIFY = 'true'
if (-not (Test-Path -LiteralPath (Join-Path $environmentPath 'python.exe'))) {
    # conda-forge avoids requiring Anaconda default-channel terms. Disable optional plugins
    # because the host's ToS plugin cache is outside the writable workspace.
    conda --no-plugins create --solver classic --prefix $environmentPath python=3.12 pip -y --override-channels -c conda-forge
    if ($LASTEXITCODE -ne 0) { throw 'Conda creation failed' }
}
$pythonPath = Join-Path $environmentPath 'python.exe'
& $pythonPath -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $pythonPath -m pip install -r (Join-Path $projectRoot 'configs/e4/requirements-models.txt') --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Model dependency installation failed' }
& (Join-Path $projectRoot '.venv/Scripts/python.exe') -X utf8 (Join-Path $PSScriptRoot 'bootstrap_e4.py') models
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
& $pythonPath -m pip freeze | Set-Content -LiteralPath (Join-Path $projectRoot '.local/e4_artifacts/model-requirements.lock.txt') -Encoding utf8
