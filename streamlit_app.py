# Entry point alias for Streamlit Community Cloud default path (streamlit_app.py -> app.py)
# NOTE: Never overwrite git-tracked .py files on disk at runtime, as it blocks Streamlit Cloud's git pull daemon and triggers endless file-watcher restarts.
with open("app.py", "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), "app.py", "exec"))
