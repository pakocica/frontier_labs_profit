"""App-side UI package for the widget (phase-1 refactor, D-043 companion).

Module map:
  model_access — notebook loading (D-025) + process-wide derived constants (m, P0, TDEF)
  theme        — two-hue palette (D-040), CSS injections, plotly figure helpers
  content      — ALL parameter/level prose & metadata dicts (labels, grades, captions)
  levels       — progressive-level ladder: labels, ranged keys, pins, level card
  state        — session-state infrastructure: reset registry, dual-mode (range/spot)
                 stores, MC range overrides, distribution bounds; level()/mc_active()
  simcache     — st.cache_data wrappers around the notebook's simulate/delay functions
  mc           — live Monte-Carlo engine + the bidirectional panel component (D-042)
  calibration  — per-parameter calibration cards + details dialog (sources, ranges)
  equations    — equations-&-calibration panel + the base-model profit condition
  sidebar      — the whole sidebar: target/param rows (D-037/D-041) -> effective dict d
  views        — header, level selector, and the five views (paths/finance/delay/MC/hood)
"""
