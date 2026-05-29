from __future__ import annotations

from html import escape
from math import ceil, sqrt
from typing import Any

import streamlit as st

from cag_dashboard import config
from cag_dashboard.logic import (
    camera_colour,
    camera_positions,
    safe_float,
    safe_int,
    zone_capacity,
    zone_state,
)

def render_floorplan(data: dict[str, Any], deltas: dict[str, int]) -> None:
    zones = data.get("zones", [])
    cameras = data.get("cameras", [])
    capacity = safe_int(data.get("capacity_limit"))
    st.subheader("Enhanced Tent Schematic")
    if not zones:
        st.info("No zone data available for the floorplan.")
        return

    zone_count = len(zones)
    fallback_capacity = capacity / zone_count if zone_count and capacity else 0
    columns = zone_count if zone_count <= 3 else 2 if zone_count <= 4 else min(4, ceil(sqrt(zone_count)))
    rows = ceil(zone_count / columns)

    view_w, view_h = 1120, 620
    tent_x, tent_y, tent_w, tent_h = 92, 112, 936, 354
    zone_gap = 16
    zone_x, zone_y = tent_x + 74, tent_y + 78
    zone_w = (tent_w - 148 - zone_gap * (columns - 1)) / columns
    zone_h = (tent_h - 156 - zone_gap * (rows - 1)) / rows
    count_size = 42 if rows == 1 else 32 if rows == 2 else 26

    zone_svg = []
    for index, zone in enumerate(zones):
        zone_id = str(zone.get("zone_id", index + 1))
        count = safe_int(zone.get("count"))
        cap = zone_capacity(zone, fallback_capacity)
        percent = (count / cap * 100) if cap else 0
        state = zone_state(percent)
        colour = config.COLOURS[state]
        row = index // columns
        col = index % columns
        x = zone_x + col * (zone_w + zone_gap)
        y = zone_y + row * (zone_h + zone_gap)
        cx = x + zone_w / 2
        delta = deltas.get(zone_id)
        delta_text = "" if delta is None or delta == 0 else f"{delta:+d}"
        zone_svg.append(
            f"""
            <g>
              <rect x="{x:.1f}" y="{y:.1f}" width="{zone_w:.1f}" height="{zone_h:.1f}" rx="18"
                    fill="{colour}" fill-opacity="0.18" stroke="{colour}" stroke-width="2.5" />
              <text x="{cx:.1f}" y="{y + 31:.1f}" text-anchor="middle"
                    font-size="18" font-weight="800" fill="{config.COLOURS['ink']}">Zone {escape(zone_id)}</text>
              <text x="{cx:.1f}" y="{y + zone_h / 2 + count_size / 3:.1f}" text-anchor="middle"
                    font-size="{count_size}" font-weight="850" fill="{config.COLOURS['ink']}">{count}</text>
              <text x="{cx:.1f}" y="{y + zone_h - 29:.1f}" text-anchor="middle"
                    font-size="13" font-weight="700" fill="{config.COLOURS['muted']}">{cap:.0f} cap | {percent:.0f}% | {state.title()}</text>
              <text x="{x + zone_w - 18:.1f}" y="{y + 28:.1f}" text-anchor="end"
                    font-size="13" font-weight="800" fill="{colour}">{escape(delta_text)}</text>
            </g>
            """
        )

    camera_svg = []
    positions = camera_positions(len(cameras), tent_x, tent_y, tent_w, tent_h)
    for index, camera in enumerate(cameras[: len(positions)]):
        x, y = positions[index]
        colour = camera_colour(camera)
        camera_id = escape(str(camera.get("camera_id", f"cam_{index + 1}")))
        status = escape(str(camera.get("status", "unknown")).title())
        label_y = y - 32 if y > tent_y + tent_h / 2 else y + 48
        camera_svg.append(
            f"""
            <g>
              <circle cx="{x:.1f}" cy="{y:.1f}" r="18" fill="{config.COLOURS['panel']}" stroke="{colour}" stroke-width="3" />
              <rect x="{x - 10:.1f}" y="{y - 7:.1f}" width="15" height="13" rx="3" fill="{colour}" />
              <path d="M{x + 5:.1f} {y - 4:.1f} L{x + 14:.1f} {y - 9:.1f} V{y + 9:.1f} L{x + 5:.1f} {y + 4:.1f} Z"
                    fill="{colour}" />
              <rect x="{x - 56:.1f}" y="{label_y - 17:.1f}" width="112" height="38" rx="11"
                    fill="{config.COLOURS['panel']}" stroke="{config.COLOURS['line']}" stroke-width="1" />
              <text x="{x:.1f}" y="{label_y - 1:.1f}" text-anchor="middle"
                    font-size="12" font-weight="800" fill="{config.COLOURS['ink']}">{camera_id}</text>
              <text x="{x:.1f}" y="{label_y + 14:.1f}" text-anchor="middle"
                    font-size="11" fill="{config.COLOURS['muted']}">{status}</text>
            </g>
            """
        )

    svg = f"""
    <svg viewBox="0 0 {view_w} {view_h}" style="width:100%;height:auto;display:block;max-width:100%;"
         xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Enhanced schematic tent floorplan">
      <rect x="0" y="0" width="{view_w}" height="{view_h}" rx="22" fill="{config.COLOURS['panel']}" />
      <rect x="26" y="26" width="{view_w - 52}" height="{view_h - 52}" rx="20"
            fill="{config.COLOURS['surface']}" stroke="{config.COLOURS['line']}" stroke-width="1.5" />
      <text x="54" y="66" font-size="19" font-weight="850" fill="{config.COLOURS['ink']}">Spatial Occupancy Map</text>
      <text x="54" y="91" font-size="13" fill="{config.COLOURS['muted']}">Tent zones, movement corridor, entry/exit flow, and camera coverage markers</text>

      <rect x="{tent_x}" y="{tent_y}" width="{tent_w}" height="{tent_h}" rx="28"
            fill="{config.COLOURS['map_fill']}" stroke="{config.COLOURS['muted']}" stroke-width="3" />
      <rect x="{tent_x + 16}" y="{tent_y + 16}" width="{tent_w - 32}" height="{tent_h - 32}" rx="22"
            fill="none" stroke="{config.COLOURS['line']}" stroke-width="2" stroke-dasharray="9 9" />
      <text x="{tent_x + 30}" y="{tent_y + 42}" font-size="15" font-weight="850" fill="{config.COLOURS['ink']}">Monitored Tent Boundary</text>

      <rect x="{tent_x - 46}" y="{tent_y + tent_h / 2 - 46}" width="82" height="92" rx="14"
            fill="#e0f2fe" stroke="#0284c7" stroke-width="2" />
      <text x="{tent_x - 5}" y="{tent_y + tent_h / 2 - 9}" text-anchor="middle"
            font-size="13" font-weight="850" fill="#075985">ENTRY</text>
      <path d="M{tent_x - 25} {tent_y + tent_h / 2 + 14} H{tent_x + 42}"
            stroke="#0284c7" stroke-width="3" stroke-linecap="round" />
      <path d="M{tent_x + 42} {tent_y + tent_h / 2 + 14} l-11 -8 v16 z" fill="#0284c7" />

      <rect x="{tent_x + tent_w - 36}" y="{tent_y + tent_h / 2 - 46}" width="82" height="92" rx="14"
            fill="#ecfdf5" stroke="#16a34a" stroke-width="2" />
      <text x="{tent_x + tent_w + 5}" y="{tent_y + tent_h / 2 - 9}" text-anchor="middle"
            font-size="13" font-weight="850" fill="#166534">EXIT</text>
      <path d="M{tent_x + tent_w - 42} {tent_y + tent_h / 2 + 14} H{tent_x + tent_w + 25}"
            stroke="#16a34a" stroke-width="3" stroke-linecap="round" />
      <path d="M{tent_x + tent_w + 25} {tent_y + tent_h / 2 + 14} l-11 -8 v16 z" fill="#16a34a" />

      <line x1="{tent_x + 54}" y1="{tent_y + tent_h / 2}" x2="{tent_x + tent_w - 54}" y2="{tent_y + tent_h / 2}"
            stroke="{config.COLOURS['muted']}" stroke-width="2" stroke-dasharray="10 10" opacity="0.85" />
      <text x="{tent_x + tent_w / 2}" y="{tent_y + tent_h / 2 - 14}" text-anchor="middle"
            font-size="12" font-weight="750" fill="{config.COLOURS['muted']}">Primary movement corridor</text>

      {''.join(zone_svg)}
      {''.join(camera_svg)}

      <line x1="54" y1="522" x2="{view_w - 54}" y2="522" stroke="{config.COLOURS['line']}" stroke-width="1" />
      <text x="54" y="558" font-size="13" font-weight="850" fill="{config.COLOURS['ink']}">Legend</text>
      <rect x="144" y="540" width="28" height="18" rx="5" fill="{config.COLOURS['normal']}" fill-opacity="0.18" stroke="{config.COLOURS['normal']}" />
      <text x="182" y="554" font-size="13" font-weight="700" fill="{config.COLOURS['ink']}">Normal</text>
      <rect x="282" y="540" width="28" height="18" rx="5" fill="{config.COLOURS['warning']}" fill-opacity="0.18" stroke="{config.COLOURS['warning']}" />
      <text x="320" y="554" font-size="13" font-weight="700" fill="{config.COLOURS['ink']}">Warning</text>
      <rect x="430" y="540" width="28" height="18" rx="5" fill="{config.COLOURS['critical']}" fill-opacity="0.18" stroke="{config.COLOURS['critical']}" />
      <text x="468" y="554" font-size="13" font-weight="700" fill="{config.COLOURS['ink']}">Critical</text>
      <circle cx="608" cy="549" r="10" fill="{config.COLOURS['panel']}" stroke="{config.COLOURS['normal']}" stroke-width="3" />
      <text x="626" y="554" font-size="13" font-weight="700" fill="{config.COLOURS['ink']}">Camera marker</text>
    </svg>
    """
    st.html(f"<div class='map-shell'>{svg}</div>")
