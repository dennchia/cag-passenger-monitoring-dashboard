from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from cag_dashboard.components import badge
from cag_dashboard.logic import data_age_label, format_timestamp

def render_header(data: dict[str, Any], state: str, age: float | None) -> None:
    state_label = state.title()
    st.html(
        f"""
        <div class="ops-header">
          <div>
            <div class="eyebrow">Real-Time Passenger Counting and Monitoring System</div>
            <h1>CAG Passenger Monitoring - Live Dashboard</h1>
          </div>
          <div class="header-meta">
            <div><span>Run ID</span><strong>{escape(str(data.get("run_id", "N/A")))}</strong></div>
            <div><span>Last Update</span><strong>{escape(format_timestamp(data.get("timestamp")))}</strong></div>
            <div><span>Data Age</span><strong>{escape(data_age_label(age))}</strong></div>
            <div><span>System</span>{badge(state_label, state)}</div>
          </div>
        </div>
        """,
    )
