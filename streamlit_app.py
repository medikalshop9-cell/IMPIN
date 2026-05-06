# Entry point for Streamlit Community Cloud
# Delegates to the main dashboard

import runpy, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
runpy.run_path(str(pathlib.Path(__file__).parent / "dashboard" / "app.py"), run_name="__main__")
