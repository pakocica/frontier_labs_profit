"""The frozen top bar (D-043): app title + segmented mode switch ("Point forecast" |
"Monte Carlo") + level selector. Sticky via theme.inject_layout_css; the same graph options
render in both modes — the switch only decides point trajectory vs MC fan per tile.
"""
import streamlit as st

from .levels import LEVEL_LABELS
from .state import _reg


def render():
    """Render the bar — the top strip of the MIDDLE area (D-048): level selector at the LEFT
    edge, the mode switch at the RIGHT edge, the small title in between. Both widgets bind to
    plain session keys ("mode", "level") that state.mc_active()/state.level() read at the top
    of the run."""
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([1.5, 2.2, 1.7], vertical_alignment="center")
        # the level selector writes st.session_state["level"]; the sidebar reads it (top of run)
        c1.selectbox("Level", LEVEL_LABELS, key="level", label_visibility="collapsed",
                     help="The explorer is layered: raise the level to add one mechanism at a "
                          "time. The sidebar, equations, and tiles grow with the level.")
        c2.markdown("<div style='text-align:center;font-size:0.95rem;font-weight:600;"
                    "opacity:0.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
                    "Frontier-AI-lab competition — explorer</div>", unsafe_allow_html=True)
        _reg("mode", "Point forecast")
        c3.segmented_control("Mode", ["Point forecast", "Monte Carlo"], key="mode",
                             label_visibility="collapsed",
                             help="**Point forecast** shows the single trajectory at the spot "
                                  "values. **Monte Carlo** shows the forecast fan across the "
                                  "sampling ranges — it keeps accumulating in the background "
                                  "either way, so switching is instant.")
