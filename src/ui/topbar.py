"""The frozen top bar (D-043): app title + segmented mode switch ("Point forecast" |
"Monte Carlo") + level selector. Sticky via theme.inject_layout_css; the same graph options
render in both modes — the switch only decides point trajectory vs MC fan per tile.
"""
import streamlit as st

from .levels import LEVEL_LABELS
from .state import _reg


def render():
    """Render the bar. Both widgets bind to plain session keys ("mode", "level") that
    state.mc_active()/state.level() read at the top of the run."""
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([2.6, 2.2, 1.6], vertical_alignment="center")
        c1.markdown("##### Frontier-AI-lab competition — explorer")
        _reg("mode", "Point forecast")
        c2.segmented_control("Mode", ["Point forecast", "Monte Carlo"], key="mode",
                             label_visibility="collapsed",
                             help="**Point forecast** shows the single trajectory at the spot "
                                  "values. **Monte Carlo** shows the forecast fan across the "
                                  "sampling ranges — it keeps accumulating in the background "
                                  "either way, so switching is instant.")
        # the level selector writes st.session_state["level"]; the sidebar read it (top of run)
        c3.selectbox("Level", LEVEL_LABELS, key="level", label_visibility="collapsed",
                     help="The explorer is layered: raise the level to add one mechanism at a "
                          "time. The sidebar, equations, and tiles grow with the level.")
