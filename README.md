# Frontier-AI-lab competition — interactive explorer

Interactive explorer for an economic model of frontier-AI-lab competition
(leader vs. open-source follower: capability race, catch-up dynamics, and
leader profitability), with point-forecast and Monte-Carlo modes.

**Live app:** https://pkocourek.com/frontier_labs_profit/

This is a static [stlite](https://github.com/whitphx/stlite) build: the
Streamlit app runs entirely in the browser via Pyodide (Python compiled to
WebAssembly). Nothing runs on a server; the first load downloads the Python
runtime and takes ~30–60 seconds.

## Layout

- `index.html` — the stlite host page (loads `@stlite/browser` from jsdelivr,
  mounts the files listed in `app_files.js`, boots `app.py`).
- `app_files.js` — generated manifest of the mounted source files.
- `src/` — the widget sources, synced verbatim from the research repo
  (`app.py`, `ui/` modules, `model_notebook.ipynb` — the single source of
  truth for the model code — and the `mc_component/` custom component).
- `deploy/` — the sync script and update instructions (see
  [`deploy/README.md`](deploy/README.md)).

The app itself is developed in the (private) research repository; this repo
only hosts the deployable snapshot. To update it, re-run `deploy/sync.sh`
and push (details in `deploy/README.md`).
