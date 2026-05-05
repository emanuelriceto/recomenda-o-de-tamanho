# =============================================================
# setup_ambiente.ps1
# Script de configuração do ambiente — rode UMA VEZ no início
# Execute no PowerShell dentro da pasta tcc_tamanho/:
#   .\setup_ambiente.ps1
# =============================================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TCC — Configuração do Ambiente Python" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar Python
Write-Host "[1/4] Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO: Python não encontrado. Baixe em https://python.org" -ForegroundColor Red
    exit 1
}
Write-Host "      OK: $pythonVersion" -ForegroundColor Green

# 2. Criar ambiente virtual
Write-Host "[2/4] Criando ambiente virtual..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "      Ambiente virtual já existe, pulando criação." -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "      OK: pasta venv/ criada" -ForegroundColor Green
}

# 3. Ativar ambiente virtual
Write-Host "[3/4] Ativando ambiente virtual..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "      OK: ambiente ativo" -ForegroundColor Green

# 4. Instalar dependências
Write-Host "[4/4] Instalando dependências (pode demorar 1-2 min)..." -ForegroundColor Yellow
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
Write-Host "      OK: dependências instaladas" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Ambiente configurado com sucesso!" -ForegroundColor Green
Write-Host ""
Write-Host "  Para ativar o ambiente nas próximas vezes:"
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  Para rodar a Fase 1 (se ainda não rodou):"
Write-Host "  python database\create_db.py" -ForegroundColor White
Write-Host "  python data\size_charts\size_charts_data.py" -ForegroundColor White
Write-Host "  python data\ansur\load_ansur.py" -ForegroundColor White
Write-Host "  python verificar_fase1.py" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
