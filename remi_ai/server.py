"""Development server: the same page and Api in a normal browser, on this computer only.

The page calls POST /api/<method> with a JSON list of arguments, which mirrors
window.pywebview.api.<method>(...) in the desktop window.
"""
import json
import mimetypes
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve(api, web_dir, port=8765, open_browser=True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            name = self.path.split("?")[0].lstrip("/") or "index.html"
            path = os.path.normpath(os.path.join(web_dir, name))
            if not path.startswith(os.path.normpath(web_dir)) or not os.path.isfile(path):
                return self._send(404, b"not found", "text/plain")
            with open(path, "rb") as f:
                body = f.read()
            if name == "index.html":  # tell the page to call /api/ instead of waiting for the window bridge
                body = body.replace(b"<head>", b"<head><script>window.REMI_DEV = true</script>", 1)
            self._send(200, body, mimetypes.guess_type(path)[0] or "application/octet-stream")

        def do_POST(self):
            name = self.path.rsplit("/", 1)[-1]
            fn = getattr(api, name, None)
            if not self.path.startswith("/api/") or name.startswith("_") or not callable(fn):
                return self._send(404, b"{}", "application/json")
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"[]")
            self._send(200, json.dumps(fn(*args)).encode(), "application/json")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Remi AI running at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
