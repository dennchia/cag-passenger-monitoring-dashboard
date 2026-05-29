from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from cag_dashboard import config
from cag_dashboard.components import badge
from cag_dashboard.logic import (
    active_alert,
    capacity_state,
    reliability_note,
    safe_int,
    trend_summary,
    update_critical_timer,
)

def render_primary_status(
    data: dict[str, Any],
    state: str,
    warning_threshold: float,
    critical_threshold: float,
    demo_mode: bool,
    age: float | None,
) -> None:
    total = safe_int(data.get("total_count"))
    capacity = safe_int(data.get("capacity_limit"))
    ratio, capacity_status = capacity_state(total, capacity, warning_threshold, critical_threshold)
    percent = ratio * 100
    trend, trend_detail, eta = trend_summary(data.get("count_history", []), capacity)
    timer = update_critical_timer(state)
    note = reliability_note(data, state, age, demo_mode)
    bar_width = min(percent, 100)
    colour = config.COLOURS.get(state if state == "offline" else capacity_status, config.COLOURS["normal"])

    st.html(
        f"""
        <section class="primary-grid">
          <div class="count-card">
            <div class="metric-label">Total Fused Count</div>
            <div class="count-number">{total}</div>
            <div class="count-caption">Current people detected inside monitored tent area</div>
          </div>
          <div class="capacity-card">
            <div class="metric-label">Capacity Used</div>
            <div class="capacity-number">{percent:.0f}%</div>
            <div class="capacity-sub">{total}/{capacity} planned capacity</div>
            <div class="capacity-track">
              <div style="width:{bar_width:.1f}%; background:{colour};"></div>
            </div>
            <div class="threshold-row">
              <span>Warning {warning_threshold:.0%}</span>
              <span>Critical {critical_threshold:.0%}</span>
            </div>
          </div>
          <div class="ops-brief-card">
            <div class="metric-label">Operational Brief</div>
            <div class="brief-row"><span>Trend</span><strong>{escape(trend)}</strong></div>
            <div class="brief-row"><span>Change</span><strong>{escape(trend_detail)}</strong></div>
            <div class="brief-row"><span>Capacity ETA</span><strong>{escape(eta)}</strong></div>
            <div class="brief-row"><span>Critical Timer</span><strong>{escape(timer)}</strong></div>
            <p>{escape(note)}</p>
          </div>
        </section>
        """,
    )


def render_alert_strip(data: dict[str, Any], state: str) -> None:
    total = safe_int(data.get("total_count"))
    capacity = safe_int(data.get("capacity_limit"))
    level, message = active_alert(data, state, total, capacity)
    colour = config.COLOURS.get(level, config.COLOURS["info"])
    st.html(
        f"""
        <div class="alert-strip" style="border-left-color:{colour}; background:{colour}1c;">
          <div class="alert-level" style="background:{colour};">{escape(level.title())}</div>
          <div>{escape(message)}</div>
        </div>
        """,
    )
