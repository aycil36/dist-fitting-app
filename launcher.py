import subprocess
import sys
import threading
import time
import webbrowser
import os

def open_browser():
    time.sleep(4)
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    t = threading.Thread(target=open_browser)
    t.daemon = True
    t.start()
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(os.path.dirname(__file__), "app.py"),
        "--server.headless", "true",
        "--server.port", "8501"
    ])
