"""App-side UI package for the widget (phase-1 refactor, D-043 companion).

Module map (all 13 modules on disk; keep this list and widget/README.md's Layout block in step):
  model_access — imports the model module (D-025/D-110) + process-wide derived constants
                 (m, P0, TDEF)
  theme        — two-hue palette (D-040), CSS injections, plotly figure helpers
  content      — ALL parameter/level prose & metadata dicts (labels, grades, captions)
  levels       — progressive-level ladder: labels, ranged keys, pins, level card
  state        — session-state infrastructure: reset registry, dual-mode (range/spot)
                 stores, MC range overrides, distribution bounds; level()/mc_active()
  simcache     — st.cache_data wrappers around `model.simulate` / `model.delay`
  mc           — live Monte-Carlo engine + the bidirectional panel component (D-042)
  calibration  — per-parameter calibration cards + details dialog (sources, ranges)
  calpanel     — the DOCKED calibration panel (D-043 variant A2): source cards, mini rails,
                 grade/basis chips, methodology
  equations    — equations-&-calibration panel + the base-model profit condition
  sidebar      — the whole sidebar: target/param rows (D-037/D-041) -> effective dict d
  topbar       — page header, the frozen level-selector strip and the footer (D-043/D-051)
  views        — the five views (paths/finance/delay/MC/hood); entry point `render_main`
"""
