from __future__ import annotations

import streamlit as st

from cag_dashboard import config

def inject_css(colours: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: {colours['app_bg']};
            color: {colours['ink']};
          }}
          header[data-testid="stHeader"],
          div[data-testid="stToolbar"],
          div[data-testid="stDecoration"],
          div[data-testid="stStatusWidget"],
          #MainMenu {{
            visibility: hidden;
            height: 0;
          }}
          .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1480px;
            overflow-x: hidden;
          }}
          section.main,
          .stApp {{
            overflow-x: hidden;
          }}
          div[data-testid="stMetric"] {{
            background: {colours['panel']};
            border: 1px solid {colours['line']};
            border-radius: 0.5rem;
            padding: 0.8rem 0.9rem;
          }}
          div[data-testid="stMetric"] label {{
            color: {colours['muted']};
            font-weight: 700;
          }}
          .stTabs [data-baseweb="tab-list"] {{
            gap: 0.25rem;
          }}
          .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            padding-left: 1rem;
            padding-right: 1rem;
          }}
          .stTabs [data-baseweb="tab"] p {{
            color: {colours['ink']} !important;
            font-weight: 800 !important;
          }}
          .stTabs [aria-selected="true"] p {{
            color: #c53030 !important;
          }}
          .stTabs [data-baseweb="tab-highlight"] {{
            background-color: #c53030 !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_css_components() -> None:
    st.markdown(
        f"""
        <style>
          .ops-header {{
            display: grid;
            grid-template-columns: minmax(300px, 1fr) auto;
            gap: 1rem;
            align-items: end;
            padding: 1rem 0 0.75rem;
          }}
          .eyebrow {{
            color: {config.COLOURS['muted']};
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
          }}
          .ops-header h1 {{
            margin: 0.15rem 0 0;
            font-size: clamp(1.7rem, 3vw, 2.8rem);
            line-height: 1.05;
            letter-spacing: 0;
            color: {config.COLOURS['ink']};
          }}
          .header-meta {{
            display: grid;
            grid-template-columns: repeat(4, minmax(118px, auto));
            gap: 0.55rem;
          }}
          .header-meta > div {{
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.5rem;
            padding: 0.58rem 0.72rem;
            min-height: 58px;
          }}
          .header-meta span,
          .metric-label,
          .mini-card span,
          .tile-stats span {{
            display: block;
            color: {config.COLOURS['muted']};
            font-size: 0.76rem;
            font-weight: 800;
          }}
          .header-meta strong {{
            display: block;
            color: {config.COLOURS['ink']};
            font-size: 0.95rem;
            margin-top: 0.22rem;
            white-space: nowrap;
          }}
          .status-badge {{
            display: inline-flex;
            color: #ffffff !important;
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 850;
            margin-top: 0.18rem;
          }}
          .primary-grid {{
            display: grid;
            grid-template-columns: minmax(280px, 1.08fr) minmax(260px, 0.72fr) minmax(300px, 0.9fr);
            gap: 1rem;
            align-items: stretch;
            margin-top: 0.6rem;
          }}
          .count-card,
          .capacity-card,
          .ops-brief-card {{
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.65rem;
            padding: 1rem;
          }}
          .count-number {{
            font-size: clamp(5rem, 9vw, 8.4rem);
            line-height: 0.9;
            letter-spacing: 0;
            font-weight: 900;
            color: {config.COLOURS['ink']};
            margin-top: 0.35rem;
          }}
          .count-caption,
          .capacity-sub,
          .ops-brief-card p,
          .threshold-row {{
            color: {config.COLOURS['muted']};
            font-size: 0.9rem;
          }}
          .capacity-number {{
            font-size: clamp(3.2rem, 5.8vw, 5.2rem);
            line-height: 1;
            font-weight: 900;
            color: {config.COLOURS['ink']};
            margin-top: 0.45rem;
          }}
          .capacity-track {{
            height: 0.78rem;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 1rem;
          }}
          .capacity-track > div {{
            height: 100%;
            border-radius: 999px;
          }}
          .threshold-row {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 0.45rem;
            font-weight: 700;
          }}
          .brief-row {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.45rem 0;
            border-bottom: 1px solid #e2e8f0;
          }}
          .brief-row span {{
            color: {config.COLOURS['muted']};
            font-weight: 750;
          }}
          .brief-row strong {{
            color: {config.COLOURS['ink']};
            text-align: right;
          }}
          .alert-strip {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            border-left: 6px solid;
            border-radius: 0.55rem;
            padding: 0.85rem 1rem;
            margin: 1rem 0;
            font-weight: 750;
          }}
          .alert-level {{
            color: #ffffff;
            border-radius: 999px;
            padding: 0.18rem 0.58rem;
            font-size: 0.78rem;
            font-weight: 850;
            flex: 0 0 auto;
          }}
          .map-shell svg {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 0.75rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
          }}
          .stitched-feed-card {{
            background: {config.COLOURS['panel']};
            border: 1px solid {config.COLOURS['line']};
            border-radius: 0.65rem;
            padding: 0.7rem;
            margin-bottom: 0.9rem;
          }}
          .stitched-media {{
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
            border-radius: 0.48rem;
            border: 1px solid {config.COLOURS['line']};
            background: {config.COLOURS['panel_soft']};
          }}
          .stitched-feed-empty {{
            min-height: 190px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 0.45rem;
            text-align: center;
            background:
              linear-gradient(135deg, transparent 0 48%, {config.COLOURS['line']} 49% 51%, transparent 52% 100%),
              {config.COLOURS['panel_soft']};
          }}
          .stitched-feed-empty strong {{
            color: {config.COLOURS['ink']};
            font-size: 0.98rem;
          }}
          .stitched-feed-empty span,
          .feed-source {{
            color: {config.COLOURS['muted']};
            font-size: 0.78rem;
            font-weight: 700;
          }}
          .feed-source {{
            margin-top: 0.45rem;
            overflow-wrap: anywhere;
          }}
          .tile-stack {{
            display: grid;
            gap: 0.65rem;
          }}
          .camera-tile {{
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-left: 5px solid;
            border-radius: 0.55rem;
            padding: 0.78rem 0.85rem;
          }}
          .tile-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
          }}
          .tile-top strong {{
            color: {config.COLOURS['ink']};
          }}
          .tile-top span {{
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            font-size: 0.73rem;
            font-weight: 850;
            white-space: nowrap;
          }}
          .tile-stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.7rem;
            margin-top: 0.65rem;
          }}
          .tile-stats strong {{
            color: {config.COLOURS['ink']};
            font-size: 1.4rem;
            line-height: 1.05;
          }}
          .tile-note {{
            margin-top: 0.45rem;
            color: {config.COLOURS['muted']};
            font-size: 0.8rem;
            font-weight: 650;
          }}
          .health-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
            gap: 0.65rem;
          }}
          .mini-card {{
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.55rem;
            padding: 0.78rem 0.85rem;
          }}
          .mini-card strong {{
            display: block;
            margin-top: 0.25rem;
            color: {config.COLOURS['ink']};
            font-size: 1.1rem;
          }}
          .privacy-note {{
            margin-top: 1rem;
            border: 1px solid #d97706;
            border-left: 6px solid #b7791f;
            border-radius: 0.6rem;
            background: #fffbeb;
            color: #713f12;
            padding: 0.9rem 1rem;
            font-size: 1rem;
            font-weight: 650;
          }}
          .privacy-note strong {{
            color: #713f12;
            font-weight: 850;
          }}
          .light-table-wrap {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid rgba(148, 163, 184, 0.32);
            border-radius: 0.65rem;
            background: #ffffff;
          }}
          .light-table {{
            width: 100%;
            border-collapse: collapse;
            color: {config.COLOURS['ink']};
            background: #ffffff;
            font-size: 0.95rem;
          }}
          .light-table th {{
            text-align: left;
            color: {config.COLOURS['muted']};
            background: #f8fafc;
            font-weight: 850;
            padding: 0.72rem 0.8rem;
            border-bottom: 1px solid #dbe3ee;
          }}
          .light-table td {{
            padding: 0.72rem 0.8rem;
            border-bottom: 1px solid #edf2f7;
            font-weight: 650;
          }}
          .light-table tbody tr:last-child td {{
            border-bottom: 0;
          }}
          .count-card,
          .capacity-card,
          .ops-brief-card,
          .header-meta > div,
          .camera-tile,
          .mini-card,
          .light-table-wrap,
          .light-table {{
            background: {config.COLOURS['panel']} !important;
            border-color: {config.COLOURS['line']} !important;
          }}
          .light-table th {{
            background: {config.COLOURS['table_header']} !important;
            color: {config.COLOURS['muted']} !important;
            border-color: {config.COLOURS['line']} !important;
          }}
          .light-table td,
          .brief-row {{
            border-color: {config.COLOURS['line']} !important;
          }}
          .capacity-track {{
            background: {config.COLOURS['line']} !important;
          }}
          .privacy-note {{
            background: {config.COLOURS['privacy_bg']} !important;
            color: {config.COLOURS['privacy_text']} !important;
            border-color: {config.COLOURS['privacy_border']} !important;
          }}
          .privacy-note strong {{
            color: {config.COLOURS['privacy_text']} !important;
          }}
          .theme-toggle-row {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            min-height: 2.8rem;
          }}
          .theme-toggle-link {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.5rem;
            border-radius: 0.7rem;
            border: 1px solid {config.COLOURS['theme_button_border']} !important;
            background: {config.COLOURS['theme_button_bg']} !important;
            background-color: {config.COLOURS['theme_button_bg']} !important;
            color: {config.COLOURS['theme_button_text']} !important;
            text-decoration: none !important;
            font-weight: 850;
            line-height: 1;
            padding: 0.55rem 1.05rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
            transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
          }}
          .theme-toggle-link:hover {{
            transform: translateY(-1px);
            background: {config.COLOURS['theme_button_hover']} !important;
            background-color: {config.COLOURS['theme_button_hover']} !important;
            border-color: {config.COLOURS['info']} !important;
            color: {config.COLOURS['theme_button_text']} !important;
            text-decoration: none !important;
          }}
          div[data-testid="stPopover"] button {{
            min-height: 2.5rem;
            height: 2.5rem;
            width: 100%;
            border-radius: 0.7rem;
            border: 1px solid {config.COLOURS['theme_button_border']} !important;
            background: {config.COLOURS['theme_button_bg']} !important;
            background-color: {config.COLOURS['theme_button_bg']} !important;
            background-image: none !important;
            color: {config.COLOURS['theme_button_text']} !important;
            font-weight: 850;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
          }}
          div[data-testid="stPopover"] button *,
          div[data-testid="stPopover"] button p {{
            color: {config.COLOURS['theme_button_text']} !important;
          }}
          div[data-testid="stPopover"] button:hover {{
            background: {config.COLOURS['theme_button_hover']} !important;
            background-color: {config.COLOURS['theme_button_hover']} !important;
            border-color: {config.COLOURS['info']} !important;
            color: {config.COLOURS['theme_button_text']} !important;
          }}
          @media (max-width: 980px) {{
            .ops-header,
            .primary-grid {{
              grid-template-columns: 1fr;
            }}
            .header-meta {{
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
