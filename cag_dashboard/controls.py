from __future__ import annotations

from pathlib import Path

import streamlit as st

from cag_dashboard import config

def settings_controls() -> tuple[Path, bool, bool, bool, float, float, int, str]:
    with st.popover(
        "Settings",
        icon=":material/settings:",
        use_container_width=True,
    ):
        st.markdown("**Data Source**")
        selected = st.selectbox("Demo scenario", list(config.SCENARIO_PATHS.keys()))
        selected_path = config.SCENARIO_PATHS[selected]
        if selected_path is None:
            data_path = Path(
                st.text_input(
                    "Fusion output JSON",
                    "data/live_fusion_output.json",
                    help="Use this for live fusion output from teammates.",
                )
            )
        else:
            data_path = selected_path
            st.text_input("Fusion output JSON", str(data_path), disabled=True)

        st.markdown("**Live Video**")
        stitched_feed_url = st.text_input(
            "Stitched feed URL",
            "",
            help="Use the HTTP stream from the stitching server, for example http://localhost:8080/stitched_feed.",
        )

        st.markdown("**Runtime**")
        demo_mode = st.toggle(
            "Demo mode",
            value=True,
            help="On: treats fixed scenario timestamps as live. Off: checks stale/live data normally.",
        )
        auto_refresh = st.toggle("Auto-refresh dashboard", value=False)
        refresh_seconds = st.selectbox("Refresh interval", [1, 2, 5, 10], index=1)
        show_technical = st.toggle("Show technical tools", value=False)

        st.markdown("**Capacity Thresholds**")
        warning_threshold = st.slider("Warning threshold", 0.30, 0.90, 0.60, 0.05)
        critical_threshold = st.slider("Critical threshold", 0.50, 1.00, 0.85, 0.05)
        if critical_threshold <= warning_threshold:
            st.warning("Critical threshold should be above warning threshold.")
            critical_threshold = min(1.0, warning_threshold + 0.05)

    return (
        data_path,
        demo_mode,
        auto_refresh,
        show_technical,
        warning_threshold,
        critical_threshold,
        int(refresh_seconds),
        stitched_feed_url.strip(),
    )
