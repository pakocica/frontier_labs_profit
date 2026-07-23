"""The page header + frozen top strip (D-043 / D-051).

`render_title()` draws the page's main heading (spanning the middle pane, all responsive modes);
`render()` draws the sticky top strip — a "Model:" label + the complexity-level selector (with an
explainer: a hover tooltip on desktop, an ⓘ popover on narrow/phone) at the LEFT, the "Point
forecast | Monte Carlo" mode switch at the RIGHT. `render_footer()` draws the quiet author
attribution. Both widgets bind to plain session keys ("mode", "level") read at the top of the run.
"""
import streamlit as st

from .levels import LEVEL_LABELS
from .state import _reg

# ---- one-line constant swaps (Pavel owns the wording) --------------------------------------
TITLE_MAIN = "Will Frontier AI Labs Be Profitable?"
TITLE_SUB = "an interactive model of frontier-AI-lab competition"

# the complexity-level concept, drafted from the level-ladder spirit (kept crisp)
LEVEL_EXPLAINER = (
    "The explorer is **layered**: start simple and raise the level to add one mechanism at a "
    "time. Each level switches on exactly one more block of the model — a new equation and its "
    "parameters — so you can isolate what each assumption changes."
)

# author footer (ships with the widget → also appears on the web version at the next sync)
FOOTER_HTML = (
    'Built by <a href="https://pkocourek.com" target="_blank" rel="noopener">Pavel Kocourek</a>'
    '<span class="appfooter-sep">&middot;</span>work in progress'
    '<span class="appfooter-sep">&middot;</span>developed under the '
    '<a href="https://safe.ai" target="_blank" rel="noopener">CAIS Fellowship</a>'
)


def render_title():
    """The page's main heading, above the top strip; spans the middle pane in every mode (the
    theme CSS scales it with the fluid font and shrinks it in phone mode)."""
    with st.container(key="apptitle"):
        st.markdown(
            f"<div class='apptitle-main'>{TITLE_MAIN}</div>"
            f"<div class='apptitle-sub'>{TITLE_SUB}</div>",
            unsafe_allow_html=True,
        )


def render():
    """The sticky top strip: [Model: · level selector · ⓘ] at the left, mode switch flush right."""
    with st.container(key="topbar"):
        # ratios: label+picker+ⓘ hug the LEFT edge (picker width-capped in theme CSS), the
        # wide last column pushes the mode switch to the RIGHT edge (flex-end there)
        c_lab, c_sel, c_info, c_mode = st.columns([0.42, 1.15, 0.22, 2.6],
                                                   vertical_alignment="center")
        c_lab.markdown("<div class='model-label'>Model:</div>", unsafe_allow_html=True)
        # the level selector writes st.session_state["level"]; the sidebar reads it (top of run).
        # `help` is the DESKTOP hover tooltip; the ⓘ popover (narrow/phone) carries the same text.
        c_sel.selectbox("Level", LEVEL_LABELS, key="level", label_visibility="collapsed",
                        help=LEVEL_EXPLAINER)
        with c_info:
            # hover doesn't exist on touch, so narrow/phone get a tappable ⓘ; hidden in wide by
            # theme CSS (.st-key-levelinfo), where the selectbox tooltip already covers it
            with st.container(key="levelinfo"):
                with st.popover("ⓘ", help="what the complexity level means"):
                    st.markdown(LEVEL_EXPLAINER)
        _reg("mode", "Point forecast")
        c_mode.segmented_control("Mode", ["Point forecast", "Monte Carlo"], key="mode",
                                 label_visibility="collapsed",
                                 help="**Point forecast** shows the single trajectory at the spot "
                                      "values. **Monte Carlo** shows the forecast fan across the "
                                      "sampling ranges — it keeps accumulating in the background "
                                      "either way, so switching is instant.")


def render_footer():
    """The quiet author attribution at the bottom of the middle pane (all modes)."""
    with st.container(key="appfooter"):
        st.markdown(f"<div class='appfooter-inner'>{FOOTER_HTML}</div>", unsafe_allow_html=True)
