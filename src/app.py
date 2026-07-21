"""Streamlit widget — frontier-AI-lab competition (draft v2), phase 2.

This app contains ZERO model math. Every model function, dataclass, and constant is loaded at
startup from `model_notebook.ipynb` via `notebook_loader.load_model` (the notebook is the single
source of truth, D-025). The UI lives in the `ui/` package — see `ui/__init__.py` for the module
map — and this file only orchestrates one run:

    top bar (mode + level)  →  sidebar (effective dict d)  →  simulate  →  main area
    (docked calibration panel? | Introduction/Equations pane | pinned chart tiles)

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

LEVEL = state.level()
if state.mc_active():
    theme.inject_mc_slider_css()

topbar.render()
d = sidebar.render(LEVEL)

items = simcache._items(d)
sim = simcache.sim_cached(items)
p = m.Params(**d)
hl = m.headline(sim, p)

views.render_main(d, items, sim, hl, p, LEVEL)
