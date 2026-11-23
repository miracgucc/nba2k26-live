// ==UserScript==
// @name         NBA 2K26 Gece Oto-Pilot v7.1 (Melbet & 1xBet Mobil - Bilgisayar Kapalı)
// @namespace    https://github.com/miracgucc/nba2k26-live
// @version      7.1.1
// @description  35. dakikada v7.1 seçici kurallara uyan NBA 2K26 maçlarına Tek Tıkla Bahis (1-Click Bet) ile otomatik basar. Otomatik Uyku Kilidi (WakeLock) ve OLED Simsiyah Gece Ekranı içerir.
// @author       NBA2K26-CloudBot
// @match        *://*melbet*/*
// @match        *://*1xbet*/*
// @match        *://*1x-bet*/*
// @grant        none
// ==/UserScript==

(function () {
    'use strict';

    const SIGNAL_URL = "https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/live_bet_signal.json";
    const STORAGE_KEY_PLAYED = "nba2k26_v71_played_signals";
    const STORAGE_KEY_ENABLED = "nba2k26_v71_autopilot_enabled";
    const STORAGE_KEY_MAX_BETS = "nba2k26_v71_max_night_bets";
    const STORAGE_KEY_LOG = "nba2k26_v71_bet_log";

    let wakeLockSentinel = null;

    async function requestScreenWakeLock() {
        try {
            if ('wakeLock' in navigator && isAutoEnabled() && !wakeLockSentinel) {
                wakeLockSentinel = await navigator.wakeLock.request('screen');
                wakeLockSentinel.addEventListener('release', () => {
                    wakeLockSentinel = null;
                });
            }
        } catch (e) {}
    }

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible' && isAutoEnabled()) {
            requestScreenWakeLock();
        }
    });

    function getPlayedSignals() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY_PLAYED) || "[]");
        } catch (e) {
            return [];
        }
    }

    function markSignalPlayed(sigId, infoText) {
        const list = getPlayedSignals();
        if (!list.includes(sigId)) {
            list.push(sigId);
            localStorage.setItem(STORAGE_KEY_PLAYED, JSON.stringify(list.slice(-100)));
        }
        try {
            const logs = JSON.parse(localStorage.getItem(STORAGE_KEY_LOG) || "[]");
            logs.unshift({ time: new Date().toLocaleTimeString("tr-TR"), text: infoText });
            localStorage.setItem(STORAGE_KEY_LOG, JSON.stringify(logs.slice(0, 25)));
        } catch (e) {}
    }

    function isAutoEnabled() {
        return localStorage.getItem(STORAGE_KEY_ENABLED) !== "false";
    }

    function getMaxBets() {
        return parseInt(localStorage.getItem(STORAGE_KEY_MAX_BETS) || "15", 10) || 15;
    }

    // OLED Pitch-Black Night Overlay (Screen looks 100% off while browser stays active)
    const blackOverlay = document.createElement("div");
    blackOverlay.id = "nba2k26-oled-blackout";
    blackOverlay.style.cssText = [
        "display:none",
        "position:fixed",
        "inset:0",
        "z-index:1000000",
        "background:#000000",
        "color:rgba(255,255,255,0.18)",
        "font-family:system-ui,sans-serif",
        "font-size:12px",
        "flex-direction:column",
        "justify-content:center",
        "align-items:center",
        "text-align:center",
        "user-select:none"
    ].join(";");
    blackOverlay.innerHTML = `
        <div id="nba2k26-oled-info" style="padding:16px;">
            🌙 NBA 2K26 Gece Oto-Pilot Çalışıyor (Simsiyah OLED Modu)<br/>
            <span id="nba2k26-oled-sub">Gece Oynanan Kupon: 0</span><br/><br/>
            <span style="font-size:11px;color:rgba(255,255,255,0.12);">(Ekranı açmak için iki kez dokunun)</span>
        </div>
    `;
    document.body.appendChild(blackOverlay);

    let lastTapTime = 0;
    blackOverlay.addEventListener("click", () => {
        const now = Date.now();
        if (now - lastTapTime < 450) {
            blackOverlay.style.display = "none";
        }
        lastTapTime = now;
    });

    // Floating Mobile UI Panel
    const panel = document.createElement("div");
    panel.id = "nba2k26-autopilot-panel";
    panel.style.cssText = [
        "position:fixed",
        "bottom:12px",
        "right:12px",
        "z-index:999999",
        "width:295px",
        "background:rgba(15,23,42,0.96)",
        "color:#f8fafc",
        "border:2px solid #10b981",
        "border-radius:12px",
        "padding:10px 12px",
        "font-family:system-ui,-apple-system,sans-serif",
        "font-size:12px",
        "box-shadow:0 10px 25px rgba(0,0,0,0.55)"
    ].join(";");

    panel.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <strong style="color:#34d399;font-size:13px;">🌙 NBA 2K26 Oto-Pilot v7.1</strong>
            <div style="display:flex;gap:4px;">
                <button id="nba2k26-dim-btn" title="Ekranı simsiyah yapar ama arka planda bahis almaya devam eder" style="background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:6px;padding:4px 7px;font-weight:600;font-size:11px;cursor:pointer;">🌑 Karart</button>
                <button id="nba2k26-toggle-btn" style="border:none;border-radius:6px;padding:4px 8px;font-weight:700;font-size:11px;cursor:pointer;"></button>
            </div>
        </div>
        <div id="nba2k26-status-box" style="background:#1e293b;padding:6px 8px;border-radius:8px;margin-bottom:6px;line-height:1.35;">
            ⏳ Bulut sinyal motoruna bağlanıyor...
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#94a3b8;">
            <span>Gece Kupon: <b id="nba2k26-count-lbl" style="color:#fff;">0</b> / <input id="nba2k26-max-inp" type="number" min="1" max="40" style="width:38px;background:#0f172a;color:#fff;border:1px solid #334155;border-radius:4px;padding:1px 3px;text-align:center;" /></span>
            <button id="nba2k26-reset-btn" style="background:#334155;color:#cbd5e1;border:none;border-radius:4px;padding:2px 6px;font-size:10px;cursor:pointer;">Sayacı Sıfırla</button>
        </div>
        <div id="nba2k26-last-bet" style="margin-top:5px;font-size:10px;color:#fbbf24;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></div>
    `;
    document.body.appendChild(panel);

    const toggleBtn = document.getElementById("nba2k26-toggle-btn");
    const dimBtn = document.getElementById("nba2k26-dim-btn");
    const statusBox = document.getElementById("nba2k26-status-box");
    const countLbl = document.getElementById("nba2k26-count-lbl");
    const maxInp = document.getElementById("nba2k26-max-inp");
    const resetBtn = document.getElementById("nba2k26-reset-btn");
    const lastBetBox = document.getElementById("nba2k26-last-bet");
    const oledSub = document.getElementById("nba2k26-oled-sub");

    maxInp.value = getMaxBets();
    maxInp.addEventListener("change", () => {
        localStorage.setItem(STORAGE_KEY_MAX_BETS, String(maxInp.value || 15));
    });

    resetBtn.addEventListener("click", () => {
        localStorage.setItem(STORAGE_KEY_PLAYED, "[]");
        updateUI();
    });

    dimBtn.addEventListener("click", () => {
        requestScreenWakeLock();
        blackOverlay.style.display = "flex";
    });

    toggleBtn.addEventListener("click", () => {
        const next = !isAutoEnabled();
        localStorage.setItem(STORAGE_KEY_ENABLED, next ? "true" : "false");
        if (next) requestScreenWakeLock();
        updateUI();
    });

    function updateUI() {
        const on = isAutoEnabled();
        toggleBtn.textContent = on ? "🟢 AÇIK" : "🔴 KAPALI";
        toggleBtn.style.background = on ? "#10b981" : "#ef4444";
        toggleBtn.style.color = "#ffffff";
        panel.style.borderColor = on ? "#10b981" : "#ef4444";
        const cnt = getPlayedSignals().length;
        countLbl.textContent = String(cnt);
        let lastStr = "";
        try {
            const logs = JSON.parse(localStorage.getItem(STORAGE_KEY_LOG) || "[]");
            if (logs.length > 0) {
                lastStr = "Son: " + logs[0].time + " • " + logs[0].text;
                lastBetBox.textContent = lastStr;
            }
        } catch (e) {}
        if (oledSub) {
            oledSub.textContent = `Gece Oynanan Kupon: ${cnt} ${lastStr ? '(' + lastStr + ')' : ''}`;
        }
    }
    updateUI();
    requestScreenWakeLock();

    function tryClickBaremOnPage(sig) {
        const trLabel = (sig.target_barem_tr || "").toLowerCase();
        const enLabel = (sig.target_barem_en || "").toLowerCase();
        const rawBarem = (sig.target_barem || "").toLowerCase();
        const minOdd = parseFloat(sig.min_acceptable_odd || 1.50);

        const wasBlackout = blackOverlay.style.display === "flex";
        if (wasBlackout) blackOverlay.style.display = "none";

        const candidates = document.querySelectorAll(
            ".c-bets__bet, .bet_type, .ui-market__toggle, [data-market], .market-odd, .c-bet-box, button, a, div"
        );

        for (const el of candidates) {
            const txt = (el.innerText || el.textContent || "").trim().toLowerCase();
            if (!txt || txt.length > 80) continue;

            const matchesBarem =
                (trLabel && txt.includes(trLabel)) ||
                (enLabel && txt.includes(enLabel)) ||
                (rawBarem && rawBarem.includes("-") && txt.includes(rawBarem));

            if (matchesBarem) {
                const container = el.closest(".c-bets__bet, .ui-market__item, .c-bet-box, li, tr, div") || el.parentElement;
                const fullTxt = (container ? container.innerText : txt).replace(",", ".");
                const nums = fullTxt.match(/\b([1-9]\.\d{1,3})\b/g);
                if (nums && nums.length > 0) {
                    const liveOdd = parseFloat(nums[nums.length - 1]);
                    if (liveOdd >= minOdd && liveOdd < 25.0) {
                        const clickTarget = container.querySelector(".c-bets__inner, .ui-market__value, .bet_type, button") || el;
                        clickTarget.click();
                        if (wasBlackout) blackOverlay.style.display = "flex";
                        return { clicked: true, liveOdd: liveOdd };
                    }
                }
            }
        }
        if (wasBlackout) blackOverlay.style.display = "flex";
        return { clicked: false, liveOdd: 0 };
    }

    async function pollCloudSignal() {
        updateUI();
        requestScreenWakeLock();
        try {
            const res = await fetch(SIGNAL_URL + "?t=" + Date.now(), { cache: "no-store" });
            if (!res.ok) return;
            const data = await res.json();
            const played = getPlayedSignals();

            if (data.has_active_play_signal && data.current_play_signal) {
                const sig = data.current_play_signal;
                const alreadyPlayed = played.includes(sig.signal_id);
                statusBox.innerHTML = `
                    <div style="color:#34d399;font-weight:700;">🚨 35.DK OYNA SİNYALİ!</div>
                    <div><b>${sig.game_name}</b> (${sig.current_score})</div>
                    <div>🎯 <b>${sig.target_barem_tr}</b> @ <b>${sig.target_odd}</b> (${sig.pick_type})</div>
                    <div style="color:${alreadyPlayed ? '#38bdf8' : '#fbbf24'};">
                        ${alreadyPlayed ? '✅ Bu maça kupon basıldı (Kilitli)' : '⚡ Tek Tıkla Bahis aranıyor...'}
                    </div>
                `;

                if (isAutoEnabled() && !alreadyPlayed && played.length < getMaxBets()) {
                    const curUrl = window.location.href;
                    if (sig.raw_game_id && !curUrl.includes(sig.raw_game_id)) {
                        const links = document.querySelectorAll("a[href]");
                        for (const a of links) {
                            const href = a.getAttribute("href") || "";
                            const aTxt = (a.innerText || "").toLowerCase();
                            if (href.includes(sig.raw_game_id) || (sig.home_team && aTxt.includes(sig.home_team.toLowerCase()))) {
                                a.click();
                                return;
                            }
                        }
                    }
                    const clickRes = tryClickBaremOnPage(sig);
                    if (clickRes.clicked) {
                        markSignalPlayed(sig.signal_id, `${sig.target_barem_tr} @ ${clickRes.liveOdd}`);
                        updateUI();
                    }
                }
            } else {
                const liveCount = (data.live_matches || []).length;
                const lastHist = (data.recent_history || [])[0];
                statusBox.innerHTML = `
                    <div>📡 Bulut Aktif (${data.updated_at_tr.slice(11)}) • Canlı: <b>${liveCount} maç</b></div>
                    <div style="color:#94a3b8;font-size:11px;">35. dakikada kurala uyan maç bekleniyor...</div>
                    ${lastHist ? `<div style="font-size:10px;color:#cbd5e1;margin-top:2px;">Son Karar: ${lastHist.game_name.slice(0, 22)} ➔ <b>${lastHist.decision}</b> ${lastHist.target_barem_tr || ''}</div>` : ''}
                `;
            }
        } catch (err) {
            statusBox.innerHTML = `<span style="color:#f87171;">⚠️ Bağlantı bekleniyor...</span>`;
        }
    }

    setInterval(pollCloudSignal, 10000);
    pollCloudSignal();
})();
