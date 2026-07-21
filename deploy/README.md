# Deploying / updating the explorer

The live app (https://pkocourek.com/frontier_labs_profit/) is a static
[stlite](https://github.com/whitphx/stlite) page served by GitHub Pages from
this repo (`pakocica/frontier_labs_profit`, branch `main`, path `/`). There is
no build step: `index.html` mounts the files under `src/` into an in-browser
Python (Pyodide) filesystem and runs the Streamlit app there.

## Updating after changes to the widget

From a clone of this repo, run the sync script and push:

```bash
deploy/sync.sh            # copies widget sources from the model repo (default path)
git add -A
git commit -m "sync widget"
git push
```

`sync.sh` takes an optional argument if the model repo lives elsewhere:

```bash
deploy/sync.sh "/path/to/model/widget"
```

It copies `app.py`, `notebook_loader.py`, `model_notebook.ipynb`, `ui/*.py`
and `mc_component/index.html` into `src/`, and regenerates `app_files.js`
(the manifest `index.html` mounts). New `ui/*.py` modules are picked up
automatically; if the widget ever grows files *outside* those locations
(new top-level modules, new asset dirs), add a `cp` line to `sync.sh`.

GitHub Pages redeploys automatically on push (usually < 1 minute). Hard-reload
the page (Cmd-Shift-R) to bypass cached assets when checking.

## What to check after an update

Load the live URL, wait out the ~30–60 s Pyodide boot, then click through:
level switch, Point forecast ↔ Monte Carlo, a slider move (charts react),
a ⓘ calibration panel, and let the Monte-Carlo accumulation reach a
milestone (bands appear at n = 10). The browser console should show no
Python tracebacks (one "Custom Component ui.mc.mc_panel fetch error" line is
expected and harmless — it is Streamlit's health-check fetch, which stlite
serves through its own channel).

## Known web-build divergences from the native app

- Everything runs client-side; the first load downloads the Pyodide runtime
  (~30–60 s; cached afterwards).
- Monte-Carlo accumulation is ~2× slower than native (WASM): the full 100
  draws take ~1.5–2 min instead of ~40 s. Bands appear from n = 10, and the
  UI stays responsive throughout, so no cap was reduced for the web build.
- Pinned versions: `@stlite/browser@1.8.1` (bundles Streamlit 1.57 on
  Python 3.13) in `index.html`. The native app runs Streamlit 1.59; no
  1.58/1.59-only APIs are used by the widget.
