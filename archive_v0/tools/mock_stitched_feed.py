from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def mock_svg() -> bytes:
    timestamp = datetime.now().strftime("%H:%M:%S")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
  <defs>
    <style>
      @keyframes pulse {{ 0%, 100% {{ opacity: .35; }} 50% {{ opacity: 1; }} }}
      @keyframes scan {{ 0% {{ transform: translateX(-180px); }} 100% {{ transform: translateX(960px); }} }}
      @keyframes walk {{ 0% {{ transform: translate(0,0); }} 50% {{ transform: translate(28px,-16px); }} 100% {{ transform: translate(0,0); }} }}
      .pulse {{ animation: pulse 1.1s ease-in-out infinite; }}
      .scan {{ animation: scan 3s linear infinite; }}
      .walker {{ animation: walk 2.4s ease-in-out infinite; }}
      text {{ font-family: Inter, Arial, sans-serif; }}
    </style>
  </defs>
  <rect width="960" height="540" fill="#0f172a"/>
  <rect x="24" y="24" width="912" height="492" rx="26" fill="#111827" stroke="#334155" stroke-width="2"/>
  <text x="52" y="70" fill="#f8fafc" font-size="28" font-weight="800">Mock Stitched Camera Feed</text>
  <circle class="pulse" cx="830" cy="58" r="9" fill="#22c55e"/>
  <text x="850" y="65" fill="#bbf7d0" font-size="18" font-weight="800">LIVE {timestamp}</text>

  <g stroke="#475569" stroke-width="2">
    <line x1="480" y1="102" x2="480" y2="484"/>
    <line x1="52" y1="293" x2="908" y2="293"/>
  </g>

  <g>
    <rect x="64" y="118" width="388" height="150" rx="16" fill="#1e293b" stroke="#475569"/>
    <rect x="508" y="118" width="388" height="150" rx="16" fill="#1e293b" stroke="#475569"/>
    <rect x="64" y="320" width="388" height="150" rx="16" fill="#1e293b" stroke="#475569"/>
    <rect x="508" y="320" width="388" height="150" rx="16" fill="#1e293b" stroke="#475569"/>
  </g>

  <g fill="#94a3b8" font-size="17" font-weight="800">
    <text x="86" y="150">CAM 1</text>
    <text x="530" y="150">CAM 2</text>
    <text x="86" y="352">CAM 3</text>
    <text x="530" y="352">CAM 4</text>
  </g>

  <g fill="#38bdf8" opacity=".95">
    <circle class="walker" cx="170" cy="205" r="13"/>
    <circle class="walker" cx="302" cy="222" r="11" style="animation-delay: .4s"/>
    <circle class="walker" cx="610" cy="204" r="12" style="animation-delay: .7s"/>
    <circle class="walker" cx="760" cy="410" r="13" style="animation-delay: .2s"/>
    <circle class="walker" cx="215" cy="405" r="10" style="animation-delay: .9s"/>
  </g>

  <rect class="scan" x="0" y="104" width="150" height="386" fill="#22c55e" opacity=".08"/>
  <text x="52" y="506" fill="#cbd5e1" font-size="15" font-weight="700">
    Test URL for dashboard Settings: http://localhost:8080/stitched_feed
  </text>
</svg>"""
    return svg.encode("utf-8")


class MockFeedHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/stitched_feed"}:
            body = mock_svg()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        print(f"[mock-feed] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a mock stitched camera feed for dashboard testing.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MockFeedHandler)
    print(f"Mock stitched feed running at http://{args.host}:{args.port}/stitched_feed")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping mock stitched feed.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
