from __future__ import annotations

import argparse
import getpass
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlparse, urlunparse

import cv2
from ultralytics import YOLO


class YoloCameraStream:
    def __init__(
        self,
        rtsp_url: str,
        model_path: str,
        confidence: float,
        width: int,
        height: int,
        jpeg_quality: int,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.close()
        self.capture = cv2.VideoCapture(self.rtsp_url)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open the RTSP camera stream. Check IP, network, username, and password.")

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def jpeg_frame(self) -> bytes | None:
        if self.capture is None or not self.capture.isOpened():
            self.open()

        assert self.capture is not None
        success, frame = self.capture.read()
        if not success:
            self.close()
            time.sleep(1)
            return None

        results = self.model(frame, classes=[0], conf=self.confidence, verbose=False)
        annotated_frame = results[0].plot()

        if self.width > 0 and self.height > 0:
            annotated_frame = cv2.resize(annotated_frame, (self.width, self.height))

        ok, buffer = cv2.imencode(
            ".jpg",
            annotated_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None
        return buffer.tobytes()


def mask_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.username or parsed.password:
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        masked_netloc = host
        if parsed.username:
            masked_netloc = f"{parsed.username}:***@{host}"
        return urlunparse(parsed._replace(netloc=masked_netloc))
    return raw_url


def build_rtsp_url(args: argparse.Namespace) -> str:
    if args.rtsp_url:
        return args.rtsp_url

    camera_ip = args.camera_ip or os.getenv("CAG_CAMERA_IP", "")
    username = args.username or os.getenv("CAG_CAMERA_USERNAME", "admin")
    password = args.password or os.getenv("CAG_CAMERA_PASSWORD", "")
    if not camera_ip:
        raise SystemExit("Missing camera IP. Set $env:CAG_CAMERA_IP or pass --camera-ip.")
    if not password:
        password = getpass.getpass("Camera password: ")

    safe_username = quote(username, safe="")
    safe_password = quote(password, safe="")
    channel = str(args.channel).strip("/")
    return f"rtsp://{safe_username}:{safe_password}@{camera_ip}:554/Streaming/Channels/{channel}"


def make_handler(stream: YoloCameraStream) -> type[BaseHTTPRequestHandler]:
    class StreamHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
                return

            if path == "/":
                body = (
                    b"<html><body style='margin:0;background:#0f172a;'>"
                    b"<img src='/stitched_feed' style='width:100%;height:auto;display:block;' />"
                    b"</body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path != "/stitched_feed":
                self.send_error(404, "Not found")
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            while True:
                try:
                    frame = stream.jpeg_frame()
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception as error:
                    print(f"[stream] {error}")
                    time.sleep(1)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[stream] {self.address_string()} - {format % args}")

    return StreamHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Hikvision RTSP + YOLO stream for the Streamlit dashboard.")
    parser.add_argument(
        "--rtsp-url",
        default=os.getenv("CAG_RTSP_URL", ""),
        help="Full RTSP URL. If your password has special characters like @, prefer --camera-ip with CAG_CAMERA_PASSWORD.",
    )
    parser.add_argument("--camera-ip", default=os.getenv("CAG_CAMERA_IP", ""), help="Camera IP address, for example 172.20.10.5.")
    parser.add_argument("--username", default=os.getenv("CAG_CAMERA_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("CAG_CAMERA_PASSWORD", ""), help="Camera password. Prefer CAG_CAMERA_PASSWORD.")
    parser.add_argument("--channel", default=os.getenv("CAG_CAMERA_CHANNEL", "101"), help="Hikvision channel, usually 101 or 102.")
    parser.add_argument("--model", default=os.getenv("CAG_YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rtsp_url = build_rtsp_url(args)

    stream = YoloCameraStream(
        rtsp_url=rtsp_url,
        model_path=args.model,
        confidence=args.conf,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"Opening RTSP stream: {mask_url(rtsp_url)}")
    stream.open()
    print("Camera stream opened.")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(stream))
    print(f"Dashboard feed URL: http://{args.host}:{args.port}/stitched_feed")
    print("Open that URL in Chrome first, then paste it into Dashboard > Settings > Stitched feed URL.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping stream server.")
    finally:
        server.server_close()
        stream.close()


if __name__ == "__main__":
    main()
