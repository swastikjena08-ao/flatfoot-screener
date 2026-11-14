"""Development server: the same page and Api in a normal browser, on this computer only.

The page calls POST /api/<method> with a JSON list of arguments, which mirrors
window.pywebview.api.<method>(...) in the desktop window.

Other websites open in the same browser must not be able to drive it, so:
  - every request must name this server in its Host header (stops DNS-rebinding pages), and
  - every API call must carry this run's random token in an X-Remi-Token header. The token is
    only written into the page this server serves, and other sites can't set that header
    without a CORS preflight, which this server never answers.
"""
import hmac
import json
import mimetypes
import os
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve(api, web_dir, port=8765, open_browser=True):
    token = secrets.token_urlsafe(32)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            if self.headers.get("Host", "") in hosts:
                return True
            self._send(403, b"forbidden", "text/plain")
            return False

        def do_GET(self):
            if not self._host_ok():
                return
            name = self.path.split("?")[0].lstrip("/") or "index.html"
            path = os.path.normpath(os.path.join(web_dir, name))
            if not path.startswith(os.path.normpath(web_dir)) or not os.path.isfile(path):
                return self._send(404, b"not found", "text/plain")
            with open(path, "rb") as f:
                body = f.read()
            if name == "index.html":  # tell the page to call /api/ instead of waiting for the window bridge
                script = f"<head><script>window.REMI_DEV = true; window.REMI_TOKEN = {json.dumps(token)}</script>"
                body = body.replace(b"<head>", script.encode(), 1)
            self._send(200, body, mimetypes.guess_type(path)[0] or "application/octet-stream")

        def do_POST(self):
            if not self._host_ok():
                return
            if not hmac.compare_digest(self.headers.get("X-Remi-Token", "").encode(), token.encode()):
                return self._send(403, b'{"error": "Forbidden"}', "application/json")
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
