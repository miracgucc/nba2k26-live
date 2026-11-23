#!/usr/bin/env bash

REPO_DIR="$(pwd)"
export DISPLAY=:1

# Sanal ekranın (DISPLAY=:1) açılmasını bekle ve Chrome'u NBA 2K26 v7.1 Oto-Pilot Eklentisi ile otomatik başlat
(
    sleep 5
    if ! pgrep -f "nba2k26_chrome" >/dev/null 2>&1; then
        google-chrome \
            --no-sandbox \
            --disable-dev-shm-usage \
            --user-data-dir="$HOME/.nba2k26_chrome" \
            --load-extension="$REPO_DIR/vds_chrome_extension" \
            --disable-background-timer-throttling \
            --disable-renderer-backgrounding \
            --disable-backgrounding-occluded-windows \
            --window-size=1280,800 \
            "https://www.google.com" >/tmp/nba2k26_chrome.log 2>&1 &
    fi

    # Gece uyku engelleme döngüsü
    while true; do
        export DISPLAY=:1
        xdotool mousemove_relative 1 1 2>/dev/null || true
        sleep 45
        xdotool mousemove_relative -- -1 -1 2>/dev/null || true
        sleep 45
    done
) >/tmp/nba2k26_keepalive.log 2>&1 &

echo "✅ NBA 2K26 Sanal Ekran (Port 6080 - Şifre: nba2k26) ve Chrome Oto-Pilot başlatıldı!"
