"""Serve web/ locally without caching:  python serve.py  ->  http://localhost:8765"""
import functools
import http.server


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    handler = functools.partial(Handler, directory="web")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8765), handler).serve_forever()
