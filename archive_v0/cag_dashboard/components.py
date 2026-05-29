from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from cag_dashboard import config
from cag_dashboard.theme import active_theme

def badge(label: str, state: str) -> str:
    colour = config.COLOURS.get(state, config.COLOURS["info"])
    return (
        f"<span class='status-badge' style='background:{colour};color:#ffffff !important;'>"
        f"{escape(label)}</span>"
    )


def render_theme_toggle() -> None:
    mode = active_theme()
    next_mode = "dark" if mode == "light" else "light"
    label = "Dark mode" if mode == "light" else "Light mode"
    title = f"Switch to {next_mode} mode"
    st.markdown(
        f"""
        <div class="theme-toggle-row">
          <a class="theme-toggle-link" href="?theme={next_mode}" target="_self" title="{escape(title)}">
            {escape(label)}
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_light_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No rows available.")
        return

    columns = list(rows[0].keys())
    header_html = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        row_html.append(f"<tr>{cells}</tr>")

    st.html(
        f"""
        <div class="light-table-wrap">
          <table class="light-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(row_html)}</tbody>
          </table>
        </div>
        """
    )
