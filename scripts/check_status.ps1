$base = "c:/Users/rshit/Downloads/ExperimentIQ"
Write-Host "=== Project Status ===" -ForegroundColor Yellow

$dirs = @("app", "app/structured", "app/unstructured", "app/eval", "scripts", "tests")
foreach ($dir in $dirs) {
    $full = Join-Path $base $dir
    if (Test-Path $full) {
        Write-Host "  ✓ Directory: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Directory: $dir MISSING" -ForegroundColor Red
    }
}



$configs = @("pyproject.toml", "requirements.txt", "Dockerfile", "docker-compose.yml", "README.md")
Write-Host "`nConfig files:" -ForegroundColor Cyan
foreach ($file in $configs) {
    $full = Join-Path $base $file
    if (Test-Path $full) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file MISSING" -ForegroundColor Red
    }
}
