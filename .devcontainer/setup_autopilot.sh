#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  NBA 2K26 GITHUB CODESPACES SANAL EKRAN KURULUMU (v7.1)"
echo "=========================================================="

sudo apt-get update -y
sudo apt-get install -y wget curl unzip xdotool libnss3 libatk-bridge2.0-0 libgtk-3-0 libgbm1 libasound2

# Google Chrome Kurulumu
if ! command -v google-chrome >/dev/null 2>&1; then
    echo "[1/3] Google Chrome indiriliyor ve kuruluyor..."
    wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O /tmp/chrome.deb
    sudo apt-get install -y /tmp/chrome.deb || sudo apt-get -f install -y
    rm -f /tmp/chrome.deb
fi

# Tailscale Kurulumu (İsteğe Bağlı Ev/Telefon IP Köprüsü İçin)
if ! command -v tailscale >/dev/null 2>&1; then
    echo "[2/3] Tailscale indiriliyor..."
    curl -fsSL https://tailscale.com/install.sh | sh || true
fi

# Masaüstü Kısayolu Oluştur
REPO_DIR="$(pwd)"
mkdir -p ~/Desktop
cat << EOF > ~/Desktop/NBA2K26_OtoPilot_Baslat.sh
#!/usr/bin/env bash
export DISPLAY=:1
google-chrome --no-sandbox --disable-dev-shm-usage --user-data-dir="\$HOME/.nba2k26_chrome" --load-extension="$REPO_DIR/vds_chrome_extension" --disable-background-timer-throttling --disable-renderer-backgrounding --window-size=1280,800 "https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/live_bet_signal.json" &
EOF
chmod +x ~/Desktop/NBA2K26_OtoPilot_Baslat.sh

echo "[3/3] Kurulum tamamlandı!"
