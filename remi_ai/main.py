"""Start Remi AI.

    python -m remi_ai              desktop window (Windows: Edge WebView2, built into Windows 10/11)
    python -m remi_ai --browser    same app in the default web browser, for development
"""
import argparse
import os
import sys

from . import __version__
from .api import Api

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def main():
    ap = argparse.ArgumentParser(prog="remi_ai", description="Remi AI foot X-ray assistant")
    ap.add_argument("--data", help="folder for patient files (default: %%APPDATA%%\\Remi AI)")
    ap.add_argument("--models", help="folder with the AI models (default: ai_models next to the app)")
    ap.add_argument("--browser", action="store_true", help="open in a web browser instead of a window")
    ap.add_argument("--port", type=int, default=8765, help="port for --browser")
    args = ap.parse_args()

    if args.browser:
        from .server import serve
        serve(Api(args.data, args.models), WEB, args.port)
        return

    try:
        import webview
    except ImportError:
        sys.exit("pywebview is not installed. Run: pip install -r requirements-app.txt")
    api = Api(args.data, args.models)
    window = webview.create_window(f"Remi AI {__version__}", os.path.join(WEB, "index.html"), js_api=api,
                                   width=1280, height=800, min_size=(1024, 640), maximized=True,
                                   background_color="#070708")
    # Some pywebview versions ignore maximized=True on Windows, so maximize once the window is shown.
    window.events.shown += window.maximize
    webview.start()
