# Entry point alias for Streamlit Community Cloud default path (streamlit_app.py -> app.py)
# v4.5-pro-suite: Includes automatic GitHub Raw code hot-sync + module reload so Streamlit Cloud never gets stuck on stale code.
import importlib
import os
import sys
import time
import urllib.request

_LAST_CODE_SYNC = {"ts": 0.0}


def _hot_sync_latest_code():
    now = time.time()
    if now - _LAST_CODE_SYNC["ts"] < 45.0:
        return
    _LAST_CODE_SYNC["ts"] = now
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bucket = int(now // 45)
    updated_any = False
    for fname in ("period_predictor.py", "app.py"):
        try:
            url = f"https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/{fname}?cb={bucket}"
            req = urllib.request.Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                remote_code = resp.read().decode("utf-8")
            if len(remote_code) > 5000:
                fpath = os.path.join(base_dir, fname)
                cur_code = ""
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as rf:
                        cur_code = rf.read()
                if remote_code != cur_code:
                    tmp_p = fpath + ".tmp"
                    with open(tmp_p, "w", encoding="utf-8") as wf:
                        wf.write(remote_code)
                    os.replace(tmp_p, fpath)
                    updated_any = True
        except Exception:
            pass
    if updated_any and "period_predictor" in sys.modules:
        try:
            importlib.reload(sys.modules["period_predictor"])
        except Exception:
            pass


_hot_sync_latest_code()
if "period_predictor" in sys.modules and not hasattr(sys.modules["period_predictor"], "analyze_matchup_live_summary"):
    try:
        importlib.reload(sys.modules["period_predictor"])
    except Exception:
        pass

with open("app.py", "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), "app.py", "exec"))
