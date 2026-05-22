from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

def render_technical_view(data: dict[str, Any]) -> None:
    st.subheader("Technical View")
    st.caption("Debug details for integration. Keep this hidden for the main operations demo unless needed.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Cameras**")
        st.dataframe(pd.DataFrame(data.get("cameras", [])), hide_index=True, width="stretch")
        st.markdown("**Zones**")
        st.dataframe(pd.DataFrame(data.get("zones", [])), hide_index=True, width="stretch")
    with right:
        st.markdown("**Alerts**")
        st.dataframe(pd.DataFrame(data.get("alerts", [])), hide_index=True, width="stretch")
        st.markdown("**Raw JSON**")
        st.json(data)
