"""The page header + frozen top strip (D-043 / D-051).

`render_title()` draws the page's main heading (spanning the middle pane, all responsive modes);
`render()` draws the sticky top strip — a "Model:" label + the complexity-level selector (with an
explainer: a hover tooltip on desktop, an ⓘ popover on narrow/phone). The "Point forecast |
Monte Carlo" mode switch moved to the top of the CHARTS panel (Pavel, round 2: "it relates to
the graphs") — see views._charts_column; it still binds the plain "mode" session key read at
the top of the run. `render_footer()` draws the quiet author attribution.
"""
import streamlit as st

from .levels import LEVEL_LABELS
from .state import close_cal

# ---- one-line constant swaps (Pavel owns the wording) --------------------------------------
TITLE_MAIN = "Will Frontier AI Labs Be Profitable?"
TITLE_SUB = "an interactive model of frontier-AI-lab competition"

# the complexity-level concept, drafted from the level-ladder spirit (kept crisp).
# D-081: "one more mechanism" became "the next block of mechanisms" — the merged Level 2
# switches on the whole dynamics package (slowdown + ℓ + RSI engine + saturation) at once.
LEVEL_EXPLAINER = (
    "The explorer is **layered**: Level 1 is the bare steady-growth model, and each higher "
    "level switches on the next block of mechanisms — new equations and their parameters. "
    "Picking a level changes which parts of the model are active; every level is a superset of "
    "the ones below, so raising it only adds, never removes."
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
    """The sticky top strip: [Model: · level selector · ⓘ] at the left; the wide last column
    is EMPTY (the mode switch lives in the charts panel now) but keeps the first three
    left-pinned."""
    with st.container(key="topbar"):
        c_lab, c_sel, c_info, _fill = st.columns([0.42, 1.15, 0.22, 2.6],
                                                 vertical_alignment="center")
        # help= here renders the visible desktop tooltip icon next to "Model:" — the selectbox's
        # own help never shows because its label is collapsed. NOTE: the label text must stay
        # plain markdown — with unsafe_allow_html=True Streamlit leaves the help directive
        # unparsed and ":help[]" leaks into the page. Styling comes from the container key.
        with c_lab:
            with st.container(key="modellabel"):
                st.markdown("Model:", help=LEVEL_EXPLAINER)
        # the level selector writes st.session_state["level"]; the sidebar reads it (top of run).
        # `help` is the DESKTOP hover tooltip; the ⓘ popover (narrow/phone) carries the same text.
        # Switching level closes the docked calibration panel: the panel shows a parameter that
        # the new level may not even mount, and Pavel wants a level switch to read as a fresh view.
        c_sel.selectbox("Level", LEVEL_LABELS, key="level", label_visibility="collapsed",
                        help=LEVEL_EXPLAINER, on_change=close_cal)
        with c_info:
            # hover doesn't exist on touch, so narrow/phone get a tappable ⓘ; hidden in wide by
            # theme CSS (.st-key-levelinfo), where the selectbox tooltip already covers it
            with st.container(key="levelinfo"):
                with st.popover("ⓘ", help="what the complexity level means"):
                    st.markdown(LEVEL_EXPLAINER)


def render_footer():
    """The quiet author attribution at the bottom of the middle pane (all modes)."""
    with st.container(key="appfooter"):
        st.markdown(f"<div class='appfooter-inner'>{FOOTER_HTML}</div>", unsafe_allow_html=True)
