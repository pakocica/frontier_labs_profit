"""Streamlit widget — frontier-AI-lab competition (draft v2), phase 2.

This app contains ZERO model math. Every model function, dataclass, and constant is imported at
startup from `model.py` (the single source of truth, D-025; a plain module since D-085). The UI
lives in the `ui/` package — see `ui/__init__.py` for the module map — and this file only
orchestrates one run:

    top bar (mode + level)  →  sidebar (effective dict d)  →  simulate  →  main area
    (docked calibration panel? | Equations pane | pinned chart tiles)

Layout & interaction spec: D-043 (+ amendments) and D-044 in Notes/decision_log.md.

Run:  uv run streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="AI-lab competition widget", layout="wide")

from ui import sidebar, simcache, state, theme, topbar, views  # noqa: E402
from ui.model_access import m  # noqa: E402

theme.inject_base_css()
theme.inject_layout_css()
theme.inject_frontend_js()  # D-049/D-050: level-selector shim + panel drag-collapse (every run)

# (the D-042 MC-only slider-fill de-emphasis injector is gone — round 2 made the playhead a
# uniform rail in BOTH modes, in the static layout CSS)
LEVEL = state.level()

topbar.render_title()   # page heading, above the sticky top strip (D-051)
topbar.render()
d = sidebar.render(LEVEL)

items = simcache._items(d)
sim = simcache.sim_cached(items)
p = m.Params(**d)
hl = m.headline(sim, p)

views.render_main(d, items, sim, hl, p, LEVEL)
topbar.render_footer()   # quiet author attribution at the bottom of the middle pane (D-051)
