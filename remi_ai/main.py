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


ICON = os.path.join(WEB, "img", "remi.ico")


def _windows_icon(window):
    """Show the Remi logo in the title bar and taskbar instead of Python's icon."""
    import ctypes

    try:  # group the taskbar button under Remi AI, not python.exe
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RemiAI.Desktop")
    except Exception:
        pass

    def set_icon():
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.FindWindowW.restype = wintypes.HWND
            user32.LoadImageW.restype = wintypes.HANDLE
            user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                          ctypes.c_int, ctypes.c_int, wintypes.UINT]
            user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            hwnd = user32.FindWindowW(None, window.title)
            for kind, size in ((0, 16), (1, 32)):  # WM_SETICON: ICON_SMALL, ICON_BIG
                h = user32.LoadImageW(None, ICON, 1, size, size, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
                if hwnd and h:
                    user32.SendMessageW(hwnd, 0x80, kind, h)
        except Exception:
            pass

    window.events.shown += set_icon


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
    if sys.platform.startswith("win"):
        _windows_icon(window)
    webview.start(icon=ICON)
