#!/usr/bin/env bash
set -e
echo "=========================================================="
echo "  NBA 2K26 7/24 UBUNTU VDS OTO-PİLOT KURULUMU (v7.1)"
echo "=========================================================="

sudo apt-get update -y
sudo apt-get install -y curl wget unzip xfce4 xfce4-goodies xrdp dbus-x11

# XRDP Masaüstü Ayarı (Windows Uzak Masaüstü ile bağlanmak için)
echo "xfce4-session" > ~/.xsession
sudo systemctl enable xrdp
sudo systemctl restart xrdp

# Google Chrome Kurulumu
if ! command -v google-chrome >/dev/null 2>&1; then
    wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O /tmp/chrome.deb
    sudo apt-get install -y /tmp/chrome.deb || sudo apt-get -f install -y
fi

# Tailscale Ev/Telefon IP Köprüsü Kurulumu
if ! command -v tailscale >/dev/null 2>&1; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# NBA 2K26 v7.1 Oto-Pilot Chrome Eklentisini İndir
EXT_DIR="/opt/nba2k26_autopilot/extension"
sudo mkdir -p "$EXT_DIR"
REPO_RAW="https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/vds_chrome_extension"
sudo curl -fsSL "$REPO_RAW/manifest.json" -o "$EXT_DIR/manifest.json"
sudo curl -fsSL "$REPO_RAW/content.js" -o "$EXT_DIR/content.js"
sudo chmod -R 755 /opt/nba2k26_autopilot

# Masaüstü Başlatıcı Oluştur
mkdir -p ~/Desktop
cat << 'EOF' > ~/Desktop/NBA2K26_OtoPilot.sh
#!/usr/bin/env bash
google-chrome --no-sandbox --user-data-dir="$HOME/.nba2k26_chrome" --load-extension="/opt/nba2k26_autopilot/extension" --disable-background-timer-throttling --disable-renderer-backgrounding &
EOF
chmod +x ~/Desktop/NBA2K26_OtoPilot.sh

echo "=========================================================="
echo "  KURULUM TAMAMLANDI! Windows 'Uzak Masaüstü' ile VDS IP'nize"
echo "  bağlanıp Masaüstündeki NBA2K26_OtoPilot.sh ile başlatın."
echo "=========================================================="
