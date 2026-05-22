from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cag_dashboard import config
from cag_dashboard.components import render_light_table
from cag_dashboard.logic import data_age_label, format_timestamp

def render_system_health(data: dict[str, Any], age: float | None) -> None:
    cameras = data.get("cameras", [])
    online = sum(1 for cam in cameras if str(cam.get("status", "")).lower() == "online")
    total = len(cameras)
    health = data.get("system_health", {})
    items = [
        ("Cameras", f"{online}/{total} Online"),
        ("Fusion", health.get("fusion_engine", "Unknown")),
        ("Network", health.get("network", "Unknown")),
        ("Power", health.get("power", "Unknown")),
        ("Last Data", data_age_label(age)),
    ]
    cards = "".join(
        f"""
        <div class="mini-card">
          <span>{escape(label)}</span>
          <strong>{escape(str(value))}</strong>
        </div>
        """
        for label, value in items
    )
    st.subheader("System Health")
    st.html(f"<div class='health-grid'>{cards}</div>")


def render_count_chart(data: dict[str, Any]) -> None:
    history = pd.DataFrame(data.get("count_history", []))
    st.subheader("Count Over Time")
    if history.empty:
        st.info("No count history available.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["total_count"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8, color="#2563eb"),
            hovertemplate="%{x}<br>%{y} people<extra></extra>",
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=52, r=18, t=24, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOURS["panel"],
        font=dict(color=config.COLOURS["ink"], size=13),
        xaxis=dict(
            showgrid=False,
            title="",
            tickfont=dict(color=config.COLOURS["ink"], size=13),
            linecolor=config.COLOURS["muted"],
            linewidth=1,
            ticks="outside",
            tickcolor=config.COLOURS["muted"],
        ),
        yaxis=dict(
            gridcolor=config.COLOURS["grid"],
            title=dict(text="People", font=dict(color=config.COLOURS["muted"], size=14)),
            tickfont=dict(color=config.COLOURS["ink"], size=13),
            linecolor=config.COLOURS["muted"],
            linewidth=1,
            ticks="outside",
            tickcolor=config.COLOURS["muted"],
        ),
    )
    st.plotly_chart(fig, width="stretch")


def render_events(data: dict[str, Any]) -> None:
    events = data.get("event_log") or data.get("alerts", [])
    st.subheader("Latest Events")
    if not events:
        st.success("No events logged.")
        return
    rows = []
    for event in events[-5:]:
        rows.append(
            {
                "Time": format_timestamp(event.get("timestamp")),
                "Level": str(event.get("level", "info")).title(),
                "Event": event.get("message", ""),
            }
        )
    render_light_table(rows)
