from __future__ import annotations

from typing import Any

import streamlit as st

from cag_dashboard import config
from cag_dashboard.components import render_light_table
from cag_dashboard.logic import format_timestamp, safe_float

def person_display_status(person: dict[str, Any]) -> str:
    status = str(person.get("status", "not_detected")).lower()
    confidence = safe_float(person.get("confidence"))
    if status == "inside" and confidence < config.CONFIDENCE_THRESHOLD:
        return "Possible Match"
    return status.replace("_", " ").title()


def render_registered_persons(data: dict[str, Any]) -> None:
    st.subheader("Registered Persons")
    st.caption(
        "Optional display-only module for consented test participants. The dashboard does not perform face recognition."
    )
    people = data.get("recognized_persons", [])
    if not people:
        st.info("No registered person data available.")
        return
    rows = []
    for person in people:
        rows.append(
            {
                "Person ID": person.get("person_id", "N/A"),
                "Label": person.get("label", "N/A"),
                "Status": person_display_status(person),
                "Last Seen": format_timestamp(person.get("last_seen")),
                "Camera": person.get("camera_id", "N/A"),
                "Confidence": f"{safe_float(person.get('confidence')):.0%}",
            }
        )
    render_light_table(rows)
    st.html(
        """
        <div class="privacy-note">
          <strong>Privacy boundary:</strong>
          use consented test participants only, prefer anonymised labels, and manually verify low-confidence matches.
        </div>
        """
    )
