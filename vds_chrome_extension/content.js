(function () {
    'use strict';

    if (document.getElementById("nba2k26-autopilot-panel")) return;

    const SIGNAL_URL = "https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/live_bet_signal.json";
    const STORAGE_KEY_PLAYED = "nba2k26_v71_played_signals";
    const STORAGE_KEY_ENABLED = "nba2k26_v71_autopilot_enabled";
    const STORAGE_KEY_MAX_BETS = "nba2k26_v71_max_night_bets";
    const STORAGE_KEY_LOG = "nba2k26_v71_bet_log";

    let latestMelbetUrl = "https://melbet-56831.pro/tr/live/basketball/2935701-nba-2k26-cyber-league";
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
        return parseInt(localStorage.getItem(STORAGE_KEY_MAX_BETS) || "20", 10) || 20;
    }

    const panel = document.createElement("div");
    panel.id = "nba2k26-autopilot-panel";
    panel.style.cssText = [
        "position:fixed",
        "bottom:14px",
        "right:14px",
        "z-index:999999",
        "width:315px",
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
            <strong style="color:#34d399;font-size:13px;">☁️ NBA 2K26 VDS Oto-Pilot v7.1</strong>
            <button id="nba2k26-toggle-btn" style="border:none;border-radius:6px;padding:4px 8px;font-weight:700;font-size:11px;cursor:pointer;"></button>
        </div>
        <div id="nba2k26-status-box" style="background:#1e293b;padding:6px 8px;border-radius:8px;margin-bottom:6px;line-height:1.35;">
            ⏳ 7/24 Bulut sinyal motoruna bağlanıyor...
        </div>
        <div style="display:flex;gap:5px;align-items:center;margin-bottom:6px;">
            <button id="nba2k26-melbet-btn" style="flex:1;background:#2563eb;color:#fff;border:none;border-radius:6px;padding:5px 6px;font-weight:700;font-size:11px;cursor:pointer;">
                🚀 Güncel Melbet'e Git
            </button>
            <button id="nba2k26-test-btn" title="0 TL ile açık maçtaki baremi kupona eklemeyi test eder" style="background:#d97706;color:#fff;border:none;border-radius:6px;padding:5px 7px;font-weight:700;font-size:11px;cursor:pointer;">
                🧪 Test (0 TL)
            </button>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#94a3b8;">
            <span>Oynanan Kupon: <b id="nba2k26-count-lbl" style="color:#fff;">0</b> / <input id="nba2k26-max-inp" type="number" min="1" max="50" style="width:40px;background:#0f172a;color:#fff;border:1px solid #334155;border-radius:4px;padding:1px 3px;text-align:center;" /></span>
            <button id="nba2k26-reset-btn" style="background:#334155;color:#cbd5e1;border:none;border-radius:4px;padding:2px 6px;font-size:10px;cursor:pointer;">Sayacı Sıfırla</button>
        </div>
        <div id="nba2k26-last-bet" style="margin-top:5px;font-size:10px;color:#fbbf24;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></div>
    `;
    document.body.appendChild(panel);

    const toggleBtn = document.getElementById("nba2k26-toggle-btn");
    const melbetBtn = document.getElementById("nba2k26-melbet-btn");
    const testBtn = document.getElementById("nba2k26-test-btn");
    const statusBox = document.getElementById("nba2k26-status-box");
    const countLbl = document.getElementById("nba2k26-count-lbl");
    const maxInp = document.getElementById("nba2k26-max-inp");
    const resetBtn = document.getElementById("nba2k26-reset-btn");
    const lastBetBox = document.getElementById("nba2k26-last-bet");

    maxInp.value = getMaxBets();
    maxInp.addEventListener("change", () => {
        localStorage.setItem(STORAGE_KEY_MAX_BETS, String(maxInp.value || 20));
    });

    resetBtn.addEventListener("click", () => {
        localStorage.setItem(STORAGE_KEY_PLAYED, "[]");
        updateUI();
    });

    melbetBtn.addEventListener("click", () => {
        window.location.href = latestMelbetUrl;
    });

    testBtn.addEventListener("click", () => {
        const testBarems = ["201-210", "211-220", "191-200", "221-230", "181-190", "171-180", "231+"];
        let found = null;
        for (const b of testBarems) {
            const res = tryClickBaremOnPage({
                target_barem: b,
                target_barem_tr: b,
                target_barem_en: b,
                min_acceptable_odd: 1.05
            });
            if (res.clicked) {
                found = { barem: b, odd: res.liveOdd, txt: res.matchedText || b };
                break;
            }
        }
        if (found) {
            statusBox.innerHTML = `<div style="color:#34d399;font-weight:700;">✅ TEST BAŞARILI (0 TL)!</div><div>🎯 <b>${found.txt}</b> (Oran: <b>${found.odd}</b>) tıklandı! Sağdaki kupona eklendiyse sistem %100 tamamdır.</div>`;
        } else {
            const pageTxt = (document.body ? document.body.innerText : "");
            const sample = pageTxt.match(/.{0,20}(201|211|191|Aralı|aralı|Toplam).{0,25}/i);
            statusBox.innerHTML = `<div style="color:#fbbf24;font-weight:700;">⚠️ Barem Ekranda Görünmüyor!</div><div>Maçın içine girip aşağı kaydırarak 'Sayı Aralığı' başlığını açın ve tekrar Test'e basın.<br/><span style="font-size:10px;color:#94a3b8;">Bulunan ipucu: ${sample ? sample[0].replace(/\s+/g,' ') : 'Maç içi sayfa değil'}</span></div>`;
        }
    });

    toggleBtn.addEventListener("click", () => {
        const next = !isAutoEnabled();
        localStorage.setItem(STORAGE_KEY_ENABLED, next ? "true" : "false");
        if (next) requestScreenWakeLock();
        updateUI();
    });

    function updateUI() {
        const on = isAutoEnabled();
        toggleBtn.textContent = on ? "🟢 7/24 AKTİF" : "🔴 DURAKLATILDI";
        toggleBtn.style.background = on ? "#10b981" : "#ef4444";
        toggleBtn.style.color = "#ffffff";
        panel.style.borderColor = on ? "#10b981" : "#ef4444";
        countLbl.textContent = String(getPlayedSignals().length);
        try {
            const logs = JSON.parse(localStorage.getItem(STORAGE_KEY_LOG) || "[]");
            if (logs.length > 0) {
                lastBetBox.textContent = "Son: " + logs[0].time + " • " + logs[0].text;
            }
        } catch (e) {}
    }
    updateUI();
    requestScreenWakeLock();

    function tryClickBaremOnPage(sig) {
        const rawBarem = (sig.target_barem || "").trim().toLowerCase();
        const minOdd = parseFloat(sig.min_acceptable_odd || 1.05);

        let matcher = null;
        if (rawBarem.includes("-")) {
            const parts = rawBarem.split("-");
            const lo = parts[0].trim();
            const hi = parts[1].trim();
            matcher = new RegExp("\\b" + lo + "\\b.{0,22}\\b" + hi + "\\b", "i");
        } else if (rawBarem.includes("160")) {
            matcher = /\b160\b.{0,25}(az|alt|less|under|veya|kadar|<=)/i;
        } else if (rawBarem.includes("231")) {
            matcher = /\b231\b.{0,25}(fazla|üst|üzeri|more|over|\+|veya|>=)/i;
        }
        if (!matcher) return { clicked: false, liveOdd: 0 };

        // Scan clickable market elements (Melbet v3 + 1xBet Desktop & Mobile)
        const candidates = document.querySelectorAll(
            "button, [role='button'], .ui-market, .c-bets__bet, .bet_type, .market-odd, .c-bet-box, [class*='market'], [class*='bet'], [class*='coef'], li, a, div, span"
        );

        let bestMatch = null;
        for (const el of candidates) {
            if (!el.offsetParent) continue; // skip hidden elements
            const txt = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
            if (!txt || txt.length > 90) continue;

            if (matcher.test(txt)) {
                // Find the smallest clickable parent containing both the barem label AND the odd value
                let box = el;
                for (let depth = 0; depth < 4 && box; depth++) {
                    const bTxt = (box.innerText || "").replace(/\s+/g, " ").replace(",", ".").trim();
                    if (bTxt.length <= 120) {
                        // Remove barem numbers (160..231) before searching for odd (e.g. 1.85, 3.45, 13)
                        const withoutRange = bTxt.replace(/\b(1[5-9]\d|2[0-4]\d)\b/g, " ");
                        const nums = withoutRange.match(/\b([1-9]\d?(?:\.\d{1,3})?)\b/g);
                        if (nums && nums.length > 0) {
                            const liveOdd = parseFloat(nums[nums.length - 1]);
                            if (liveOdd >= minOdd && liveOdd < 35.0) {
                                // Prefer the most specific (shortest text) element
                                if (!bestMatch || bTxt.length < bestMatch.len) {
                                    bestMatch = { el: box, liveOdd: liveOdd, len: bTxt.length, text: bTxt };
                                }
                                break;
                            }
                        }
                    }
                    box = box.parentElement;
                }
            }
        }

        if (bestMatch) {
            bestMatch.el.scrollIntoView({ behavior: "instant", block: "center" });
            bestMatch.el.click();
            return { clicked: true, liveOdd: bestMatch.liveOdd, matchedText: bestMatch.text };
        }
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

            if (data.active_melbet_nba2k26_url) {
                latestMelbetUrl = data.active_melbet_nba2k26_url;
                melbetBtn.textContent = "🚀 Güncel Melbet'e Git (" + (data.active_melbet_domain || "").replace("https://", "") + ")";
            }

            // Auto domain-hop if current Melbet domain got BTK-blocked during the night
            const curHost = (window.location.hostname || "").toLowerCase();
            const bodyTxt = (document.body ? document.body.innerText : "").toLowerCase();
            const isBlockedPage =
                bodyTxt.includes("erisime_engellenmis") ||
                bodyTxt.includes("ülkenizde kullanılamamaktadır") ||
                bodyTxt.includes("err_connection") ||
                bodyTxt.includes("bu siteye ulaşılamıyor");
            if (curHost.includes("melbet") && isBlockedPage && data.active_melbet_domain && ! window.location.href.startsWith(data.active_melbet_domain)) {
                window.location.href = latestMelbetUrl;
                return;
            }

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
                        // Direct jump to the exact Melbet/1xBet match URL by game_id (works from any page!)
                        if (window.location.hostname.toLowerCase().includes("melbet") || window.location.hostname.toLowerCase().includes("1xbet")) {
                            window.location.href = window.location.origin + "/tr/live/basketball/2935701-nba-2k26-cyber-league/" + sig.raw_game_id;
                            return;
                        }
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
