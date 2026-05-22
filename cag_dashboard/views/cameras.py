from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from cag_dashboard import config

def render_stitched_feed(feed_url: str) -> None:
    st.subheader("Live Stitched Feed")
    url = feed_url.strip()
    if not url:
        st.html(
            """
            <div class="stitched-feed-card stitched-feed-empty">
              <strong>Waiting for stitched stream</strong>
              <span>Add the stream URL in Settings once the stitching server is running.</span>
            </div>
            """
        )
        return

    if not url.lower().startswith(("http://", "https://")):
        st.html(
            """
            <div class="stitched-feed-card stitched-feed-empty">
              <strong>Unsupported stream URL</strong>
              <span>Use an HTTP or HTTPS stream URL, for example http://localhost:8080/stitched_feed.</span>
            </div>
            """
        )
        return

    escaped_url = escape(url, quote=True)
    lower_url = url.lower().split("?", 1)[0]
    if lower_url.endswith((".mp4", ".webm", ".ogg")):
        media_html = (
            f'<video class="stitched-media" src="{escaped_url}" '
            "autoplay muted playsinline controls></video>"
        )
    else:
        media_html = (
            f'<img class="stitched-media" src="{escaped_url}" '
            'alt="Live stitched camera feed" />'
        )

    st.html(
        f"""
        <div class="stitched-feed-card">
          {media_html}
          <div class="feed-source">Source: {escaped_url}</div>
        </div>
        """
    )


def render_camera_panel(data: dict[str, Any]) -> None:
    st.subheader("Camera Coverage")
    cameras = data.get("cameras", [])
    if not cameras:
        st.info("No camera data available.")
        return

    tiles = []
    for index, camera in enumerate(cameras):
        camera_id = escape(str(camera.get("camera_id", f"cam_{index + 1}")))
        status = str(camera.get("status", "unknown")).lower()
        state = "normal" if status == "online" else "critical"
        colour = config.COLOURS[state]
        count = "N/A" if camera.get("count") is None else str(camera.get("count"))
        fps = "0" if camera.get("fps") in (None, "") else str(camera.get("fps"))
        visibility = str(camera.get("visibility_status", "unknown")).replace("_", " ").title()
        tiles.append(
            f"""
            <div class="camera-tile" style="border-left-color:{colour};">
              <div class="tile-top">
                <strong>{camera_id}</strong>
                <span style="color:{colour}; background:{colour}1f;">{escape(status.title())}</span>
              </div>
              <div class="tile-stats">
                <div><span>Count</span><strong>{escape(count)}</strong></div>
                <div><span>FPS</span><strong>{escape(fps)}</strong></div>
              </div>
              <div class="tile-note">Visibility: {escape(visibility)}</div>
            </div>
            """
        )
    st.html(f"<div class='tile-stack'>{''.join(tiles)}</div>")
