from __future__ import annotations

import streamlit as st

from cag_dashboard import config

def active_theme() -> str:
    query_theme = st.query_params.get("theme")
    if isinstance(query_theme, list):
        query_theme = query_theme[0] if query_theme else None
    if query_theme in config.THEMES:
        st.session_state.theme_mode = query_theme
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    return str(st.session_state.theme_mode)


def theme_colours() -> dict[str, str]:
    return {**config.BASE_COLOURS, **config.THEMES.get(active_theme(), config.THEMES["light"])}


def set_active_theme() -> None:
    st.session_state.theme_mode = "dark" if active_theme() == "light" else "light"
