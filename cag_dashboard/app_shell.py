from __future__ import annotations

import time

import streamlit as st

from cag_dashboard import config
from cag_dashboard.components import render_theme_toggle
from cag_dashboard.controls import settings_controls
from cag_dashboard.data import read_dashboard_data
from cag_dashboard.logic import display_state, zone_deltas
from cag_dashboard.styles import inject_css, render_css_components
from cag_dashboard.theme import theme_colours
from cag_dashboard.views import (
    render_alert_strip,
    render_camera_panel,
    render_count_chart,
    render_events,
    render_floorplan,
    render_header,
    render_primary_status,
    render_registered_persons,
    render_stitched_feed,
    render_system_health,
    render_technical_view,
)


def main() -> None:
    st.set_page_config(
        page_title="CAG Passenger Monitoring",
        page_icon=":airplane:",
        layout="wide",
    )

    config.COLOURS = theme_colours()
    inject_css(config.COLOURS)
    render_css_components()

    _, settings_col, theme_col = st.columns([1, 0.16, 0.16])
    with settings_col:
        (
            data_path,
            demo_mode,
            auto_refresh,
            show_technical,
            warning_threshold,
            critical_threshold,
            refresh_seconds,
            stitched_feed_url,
        ) = settings_controls()
    with theme_col:
        render_theme_toggle()

    try:
        data = read_dashboard_data(data_path)
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    state, age = display_state(data, demo_mode, warning_threshold, critical_threshold)
    deltas = zone_deltas(data.get("zones", []))

    render_header(data, state, age)

    tabs = ["Operations Overview", "Registered Persons"]
    if show_technical:
        tabs.append("Technical View")
    tab_objects = st.tabs(tabs)

    with tab_objects[0]:
        render_primary_status(data, state, warning_threshold, critical_threshold, demo_mode, age)
        render_alert_strip(data, state)

        map_col, side_col = st.columns([1.45, 0.85])
        with map_col:
            render_floorplan(data, deltas)
        with side_col:
            render_stitched_feed(stitched_feed_url)
            render_camera_panel(data)

        chart_col, health_col = st.columns([1.15, 0.85])
        with chart_col:
            render_count_chart(data)
        with health_col:
            render_system_health(data, age)
            render_events(data)

    with tab_objects[1]:
        render_registered_persons(data)

    if show_technical and len(tab_objects) > 2:
        with tab_objects[2]:
            render_technical_view(data)

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()
