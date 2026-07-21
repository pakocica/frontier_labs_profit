"""Live Monte-Carlo engine (D-042): heartbeat-driven accumulation, milestone-frozen
snapshots, and the bidirectional panel component (mc_component/index.html).
"""
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from .model_access import m
from .theme import (C_COST, C_GAP, C_GAP_MED, C_PROFIT, C_REV, C_SAMPLE, _rgba)

# ======================================================================= Monte Carlo (live)
MC_BATCH = 4           # draws added per heartbeat tick (the component ticks every ~1.2s)
MC_MIN_GAP = 0.9       # min seconds between draws (so button clicks don't trigger a fresh batch)
MC_CAP = 100           # stop drawing at this many draws (kept small so the sample browser —
                       # which appears only once accumulation is done — is reachable in seconds)
# Bands and headline stats refresh only when n crosses these counts (then freeze until the next
# one) — so the charts sit still instead of blinking on every tick.
MC_MILESTONES = (10, 20, 50, 100)


def _mc_theme():
    """Best-effort theme for the raw-iframe component. Defaults to the app's dark look; a
    light-theme user sees dark charts (acceptable per the task — the component is self-contained
    and can't inherit Streamlit's theme CSS)."""
    base = None
    try:
        t = st.context.theme
        base = getattr(t, "type", None)
    except Exception:
        base = None
    if base == "light":
        return dict(fg="#1f1f1f", grid="rgba(0,0,0,0.08)", zero="rgba(0,0,0,0.18)",
                    panel="rgba(0,0,0,0)", btn_bg="#e9e9ee", btn_fg="#1f1f1f",
                    btn_border="#c9c9d2", muted="#5a5a5a")
    return dict(fg="#d8d8d2", grid="rgba(255,255,255,0.08)", zero="rgba(255,255,255,0.16)",
                panel="rgba(0,0,0,0)", btn_bg="#2a2d34", btn_fg="#e8e8e8",
                btn_border="#464a52", muted="#9a9a94")


def _rl(a):
    """Round an array to 4 significant figures as a plain float list (None for non-finite); keeps
    the embedded JSON small enough for the iframe srcdoc."""
    return [None if not np.isfinite(x) else float(f"{x:.4g}") for x in np.asarray(a, float)]


def _mc_refresh_snapshot(store, show_blowup, show_horizon=True):
    """Recompute the FROZEN snapshot the component renders from: percentile bands, headline stats,
    and the K=min(n,80) most-recent draws (each with its plotted series and its sampled-parameter
    strip). Called only at milestone crossings, so the panel's contents change only at round counts."""
    draws = store["draws"]
    n = len(draws)
    if n == 0:
        return
    t = np.asarray(draws[0]["t"], float)
    cr = np.array([dd["crossing"] for dd in draws], float)
    n_blow = int(sum(dd["blowup"] for dd in draws))
    # D-047: the ONE stat the Finance tile shows — share of draws with positive profit at the
    # horizon T (the end of each draw's profit path)
    p_pos_T = float(np.mean([float(np.asarray(dd["profit"], float)[-1]) > 0.0
                             for dd in draws]))
    stats = {"p_prof": float(np.isfinite(cr).mean()),
             "med_cr": (float(np.nanmedian(cr[np.isfinite(cr)])) if np.isfinite(cr).any()
                        else float("nan")),
             "n_blow": n_blow, "p_pos_T": p_pos_T}
    # Two-hue scheme: each panel's bands are light/mid shades of the series' OWN hue; the median
    # is the saturated hue and its color never changes per draw. Gap = derived → grey family;
    # cost shares its identity with the Finance panel (lighter blue, dashed median).
    CH = [("profit", "Profit flow  Π  ($B/yr)", "$/yr  ($B)", C_PROFIT, C_PROFIT, "solid"),
          ("Delta", "Capability gap  Δ = xᴸ − xᶠ  (OOM)", "OOM", C_GAP_MED, C_GAP, "solid"),
          ("revenue", "Revenue  (gap rent, $B/yr)", "$/yr  ($B)", C_REV, C_REV, "solid"),
          ("cost", "Cost  ($B/yr)", "$/yr  ($B)", C_COST, C_COST, "dash")]
    charts = []
    for key, title, ylab, color, bhue, mdash in CH:
        A = np.array([dd[key] for dd in draws], float)
        A = np.where(np.isfinite(A), A, np.nan)
        lo5, lo25, med, hi75, hi95 = np.nanpercentile(A, [5, 25, 50, 75, 95], axis=0)
        # y-range fits the dark 25-75% band + median (upper-tail draws sit off-frame), as before.
        fin = np.concatenate([lo25[np.isfinite(lo25)], hi75[np.isfinite(hi75)],
                              med[np.isfinite(med)]])
        if fin.size:
            span = float(fin.max() - fin.min()) or 1.0
            yr = [float(fin.min()) - 0.08 * span, float(fin.max()) + 0.08 * span]
        else:
            yr = [0.0, 1.0]
        if key == "profit":
            # 0 must stay inside the profit panel's range (the axis zeroline marks break-even;
            # no plotted zero line) even when every band is single-signed
            yr = [min(yr[0], 0.0), max(yr[1], 0.0)]
        charts.append(dict(key=key, title=title, ylab=ylab, color=color, mdash=mdash,
                           band_light=_rgba(bhue, 0.12), band_mid=_rgba(bhue, 0.30), yrange=yr,
                           lo5=_rl(lo5), lo25=_rl(lo25), med=_rl(med), hi75=_rl(hi75),
                           hi95=_rl(hi95)))
    # (no per-draw embedding: the inspected sample's series are injected at render time by
    # mc_render, and its parameter values surface as dashed ticks on the sidebar range controls)
    next_ms = next((msn for msn in MC_MILESTONES if msn > n), None)
    store["snapshot"] = dict(t=_rl(t), charts=charts, n=n, next_ms=next_ms,
                             stopped=store["stopped"], stats=stats, show_blowup=bool(show_blowup),
                             show_horizon=bool(show_horizon))
    store["band_n"] = n
    store["_next_ms"] = next_ms


# The panel is ONE bidirectional custom component (widget/mc_component/index.html): the four
# Plotly charts plus the minimal in-panel corner controls (D-042) — a faint "n = k out of 100"
# count while accumulating, replaced by the [◀][▶][⊙ sample] pills when done (arrows only while
# the sample is shown; no per-draw caption). Clicks come back through the Streamlit component
# protocol as {show, idx, epoch}, so the server knows the inspected draw and the sidebar range
# controls can render its sampled values as dashed ticks. Between milestones Streamlit never
# re-renders the component (see mc_accumulate); when args do change, the iframe PERSISTS (stable
# key) and the component restyles the existing charts in place (Plotly.react) — nothing blinks.
_MC_PANEL = components.declare_component(
    "mc_panel", path=str(Path(__file__).resolve().parent.parent / "mc_component"))


def _mc_inspection(store):
    """(show, idx): the component's last-reported inspection state, gated to the CURRENT
    accumulation epoch (a restart invalidates the previous run's inspection) and to accumulation
    being done."""
    val = st.session_state.get("_mc_panel_val") or {}
    draws = store.get("draws") or []
    if not (store.get("stopped") and draws and val.get("show")
            and val.get("epoch") == store.get("epoch")):
        return False, 0
    return True, int(np.clip(int(val.get("idx", len(draws) - 1)), 0, len(draws) - 1))


def _inspected_params():
    """Raw sampled values of the currently inspected MC draw (empty when not inspecting) — the
    sidebar renders these as dashed ticks on the range controls."""
    store = st.session_state.get("_mc_store") or {}
    show, i = _mc_inspection(store)
    return dict(store["draws"][i]["params"]) if show else {}


# Tile order (D-043 + Pavel's HARD RULE: never place graphs side by side): the FINANCE
# instance stacks profit, revenue and cost VERTICALLY at full column width — the old
# revenue|cost row collided titles/axes at narrow widths. The MODEL-PATH instance shows the
# capability-gap fan. Indices refer to the snapshot's chart list (profit, Delta, revenue,
# cost — built in _mc_refresh_snapshot).
FIN_GRID = [[0], [2], [3]]
PATH_GRID = [[1]]


def _mc_store(mc_key):
    """The accumulation store, but only if it belongs to the CURRENT effective key."""
    store = st.session_state.get("_mc_store")
    return store if store is not None and store.get("key") == mc_key else None


def _mc_payload(store, snap, *, grid, corner, heartbeat, visible):
    """Args for one panel instance. Both instances share the frozen snapshot; they differ in
    which charts they show (grid), whether they own the corner UI + heartbeat, and visibility
    (the finance instance stays mounted HIDDEN in Point-forecast mode — background precalc)."""
    show, i = _mc_inspection(store)
    sampled = None
    if show:
        dd = store["draws"][i]
        sampled = {k: _rl(dd[k]) for k in ("profit", "Delta", "revenue", "cost")}
    return dict(t=(snap["t"] if snap else []), charts=(snap["charts"] if snap else []),
                grid=grid, corner=bool(corner), heartbeat=bool(heartbeat),
                visible=bool(visible and snap is not None),
                n=int(snap["n"]) if snap else 0, cap=int(MC_CAP),
                n_live=int(len(store.get("draws") or [])),
                done=bool(store.get("stopped")), theme=_mc_theme(), sample=C_SAMPLE,
                sampled=sampled, idx=int(i), epoch=int(store["epoch"]))


def mc_headline(mc_key, show_blowup=True):
    """The ONE compact stat the Finance tile shows in Monte-Carlo mode (D-047 declutter):
    the share of draws with positive profit at the horizon T. Frozen at milestone counts
    like the bands; everything else the tile used to show is gone — the graphs carry it."""
    store = _mc_store(mc_key)
    snap = store.get("snapshot") if store else None
    if snap is None:
        st.info("Starting the Monte-Carlo draws…")   # mc_accumulate owns store creation
        return
    pct = snap["stats"].get("p_pos_T", float("nan")) * 100.0
    if np.isfinite(pct):
        st.markdown(f"**{pct:.0f}% of draws profitable at the horizon $T$**",
                    help="Share of draws whose profit flow is positive at the end of the "
                         "horizon. Freezes at round draw counts (10, 20, 50, 100) like the "
                         "bands, so it doesn't flicker while drawing.")


def mc_panel_fin(mc_key, visible, show_blowup=True):
    """The finance panel instance (profit + revenue|cost fans, corner UI, HEARTBEAT). Mounted
    on EVERY run — with visible=False in Point-forecast mode, where it draws nothing and takes
    no space but keeps ticking, so accumulation continues in the background and switching to
    Monte Carlo is instant (D-043). The sample browser lives in the component's top-right
    corner; each corner click reports {show, idx} back. Returns True when a corner click
    landed (the CALLER reruns once so the sidebar ticks — rendered before this component —
    follow the newly inspected draw)."""
    S = st.session_state
    store = _mc_store(mc_key)
    if store is None:
        return False   # mc_accumulate creates the store; the panel mounts next run
    store["show_blowup"] = bool(show_blowup)
    snap = store.get("snapshot")
    if snap is None and visible:
        st.info("Starting the Monte-Carlo draws…")
    payload = _mc_payload(store, snap, grid=FIN_GRID, corner=True, heartbeat=True,
                          visible=visible)
    val = _MC_PANEL(data=payload, key="mc_panel", default=None)
    norm = None
    if isinstance(val, dict):
        # `tick` (the accumulation heartbeat) is deliberately EXCLUDED: its arrival alone
        # already reran the app; only inspection changes need the extra propagation rerun.
        norm = {"show": bool(val.get("show")), "idx": int(val.get("idx", 0)),
                "epoch": int(val.get("epoch", -1))}
    if norm != S.get("_mc_panel_val"):
        S["_mc_panel_val"] = norm
        return True
    return False


def mc_panel_path(mc_key):
    """The model-path panel instance (capability-gap fan): display only — no corner UI, no
    heartbeat, mounted only while the Monte-Carlo mode is visible."""
    store = _mc_store(mc_key)
    snap = store.get("snapshot") if store else None
    if snap is None:
        st.info("Starting the Monte-Carlo draws…")
        return
    _MC_PANEL(data=_mc_payload(store, snap, grid=PATH_GRID, corner=False, heartbeat=False,
                               visible=True),
              key="mc_panel_path", default=None)


def mc_accumulate(params_items, mc_key, sample_keys=None, merge_delta=False, show_blowup=True,
                  show_horizon=True, target_ranges=None, param_ranges=None):
    """Draw worker, HEARTBEAT-driven (D-042): the component emits a ~1.2s tick while
    accumulation is unfinished, and every tick reruns the app — each run of this function adds
    one batch (time-gated) and rebuilds the frozen snapshot at milestone crossings. Runs BEFORE
    mc_render in the view, so the freshly drawn state renders on the same run. Replaces the
    st.fragment(run_every=…) worker, whose frontend auto-refresh timer could be orphaned by a
    full-app rerun aborting mid-script (a permanently stalled accumulation)."""
    import time
    S = st.session_state
    store = S.get("_mc_store")
    if store is None or store.get("key") != mc_key:
        S["_mc_epoch"] = int(S.get("_mc_epoch", 0)) + 1   # invalidates stale inspection state
        store = {"key": mc_key, "draws": [], "batches": 0, "stopped": False, "last": 0.0,
                 "band_n": 0, "snapshot": None, "_next_ms": MC_MILESTONES[0],
                 "show_blowup": bool(show_blowup), "epoch": S["_mc_epoch"]}
        S["_mc_store"] = store
    now = time.time()
    if (not store["stopped"] and len(store["draws"]) < MC_CAP
            and now - store["last"] >= MC_MIN_GAP):
        p = m.Params(**dict(params_items))
        store["draws"].extend(m.mc_draw_batch(p, MC_BATCH, seed=store["batches"],
                                              sample_keys=list(sample_keys) if sample_keys else None,
                                              merge_delta=merge_delta,
                                              target_ranges=target_ranges,
                                              param_ranges=param_ranges))
        store["batches"] += 1
        store["last"] = now
        if len(store["draws"]) >= MC_CAP:
            store["stopped"] = True
    n = len(store["draws"])
    if n == 0:
        return
    nxt = store.get("_next_ms")
    # refresh at milestones, and ALWAYS once more when accumulation stops (so the final snapshot
    # renders the done state — sample controls appear — even if the cap isn't itself a milestone)
    if (store["band_n"] == 0 or (nxt is not None and n >= nxt)
            or (store["stopped"] and store["band_n"] < n)):
        _mc_refresh_snapshot(store, show_blowup, show_horizon)


def mc_prepare(d, LEVEL):
    """The effective Monte-Carlo context for this run — computed on EVERY run (D-043: the
    accumulation runs in the background whatever the mode). The cache key is the FULL
    effective tuple — level, every effective parameter, the active sampling ranges, and the
    sampled set — so ANY change (spot values, range ends, mode-of-record toggles, [choose],
    resets, level switches) restarts the accumulation from n=0.

    Returns (mc_key, sample_keys, merge_delta, target_ranges, param_ranges)."""
    from .levels import level_sample_keys
    from .state import _active_ranges, _active_rng, _mc_dim_editable, _range_mode

    def _pinned_dim(k):
        """A dimension leaves the sampled set in SPOT mode, or when its D-042 default is a
        POINT that the user hasn't widened."""
        if not _range_mode(k):
            return True
        return _active_rng(k)[0] is None and _mc_dim_editable(k)   # POINT, no user range yet

    sample_keys = [k for k in level_sample_keys(LEVEL) if not _pinned_dim(k)]
    merge_delta = LEVEL <= 5
    _over = st.session_state.get("_range_over", {})
    tro, pro = _active_ranges()
    mc_key = ((LEVEL,) + tuple(sorted(d.items()))
              + ("RANGES",) + tuple(sorted(_over.items()))
              + ("KEYS",) + tuple(sample_keys))
    return mc_key, sample_keys, merge_delta, tro, pro
