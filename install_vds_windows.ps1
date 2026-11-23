# NBA 2K26 7/24 Ücretsiz Windows VDS Tek-Komut Kurulum Scripti (v7.1)
$ErrorActionPreference = "SilentlyContinue"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  NBA 2K26 7/24 VDS GECE OTO-PİLOT KURULUMU BAŞLIYOR..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

$BaseDir = "C:\NBA2K26_AutoPilot"
$ExtDir = "$BaseDir\extension"
New-Item -ItemType Directory -Force -Path $ExtDir | Out-Null

# 1. Chrome Eklenti Dosyalarını GitHub Buluttan İndir
$RepoRaw = "https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/vds_chrome_extension"
Invoke-WebRequest -Uri "$RepoRaw/manifest.json" -UseBasicParsing -OutFile "$ExtDir\manifest.json"
Invoke-WebRequest -Uri "$RepoRaw/content.js" -UseBasicParsing -OutFile "$ExtDir\content.js"
Write-Host "[1/4] NBA 2K26 v7.1 Oto-Pilot Chrome eklentisi indirildi: $ExtDir" -ForegroundColor Green

# 2. VDS Uyku ve Ekran Kapanmasını Devre Dışı Bırak
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
Write-Host "[2/4] VDS 7/24 açık kalma (uyku engelleme) ayarları yapıldı." -ForegroundColor Green

# 3. Google Chrome Kurulu Değilse Otomatik Kur
$ChromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $ChromePath)) {
    Write-Host "[3/4] Google Chrome indiriliyor ve kuruluyor..." -ForegroundColor Yellow
    $ChromeInstaller = "$env:TEMP\chrome_installer.exe"
    Invoke-WebRequest -Uri "https://dl.google.com/chrome/install/latest/chrome_installer.exe" -UseBasicParsing -OutFile $ChromeInstaller
    Start-Process -FilePath $ChromeInstaller -ArgumentList "/silent /install" -Wait
} else {
    Write-Host "[3/4] Google Chrome zaten kurulu." -ForegroundColor Green
}

# 4. Tailscale Kurulu Değilse İndir ve Kur
$TailscalePath = "C:\Program Files\Tailscale\tailscale-ipn.exe"
if (-not (Test-Path $TailscalePath)) {
    Write-Host "[4/4] Tailscale Ev/Telefon IP Köprüsü indiriliyor..." -ForegroundColor Yellow
    $TsInstaller = "$env:TEMP\tailscale-setup.exe"
    Invoke-WebRequest -Uri "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe" -UseBasicParsing -OutFile $TsInstaller
    Start-Process -FilePath $TsInstaller -ArgumentList "/quiet" -Wait
} else {
    Write-Host "[4/4] Tailscale zaten kurulu." -ForegroundColor Green
}

# Masaüstüne 7/24 Başlatıcı Kısayolu Oluştur
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$BatFile = "$DesktopPath\NBA2K26_7_24_OtoPilot_Baslat.bat"
$BatContent = @"
@echo off
title NBA 2K26 7/24 VDS Oto-Pilot v7.1
echo NBA 2K26 7/24 VDS Oto-Pilot Baslatiliyor...
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --user-data-dir="C:\NBA2K26_AutoPilot\chrome_profile" --load-extension="C:\NBA2K26_AutoPilot\extension" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows "https://streamlit.io"
"@
Set-Content -Path $BatFile -Value $BatContent -Encoding ASCII

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  KURULUM TAMAMLANDI! Masaüstündeki 'NBA2K26_7_24_OtoPilot_Baslat' " -ForegroundColor Green
Write-Host "  dosyasına çift tıklayarak Melbet/1xBet'e giriş yapabilirsiniz." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
