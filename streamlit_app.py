# Entry point alias for Streamlit Community Cloud default path (streamlit_app.py -> app.py)
with open("app.py", "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), "app.py", "exec"))
