"""Live Monte-Carlo engine (D-042): heartbeat-driven accumulation, milestone-frozen
snapshots, and the bidirectional panel component (mc_component/index.html).
"""
from pathlib import Path

import dataclasses
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from .model_access import m
from .theme import (C_FOLLOWER, C_GAP, C_GAP_MED, C_LEADER, C_PROFIT, C_SAMPLE, YEAR0,
                    _rgba)

# ======================================================================= Monte Carlo (live)
# INTERIM RESPONSIVENESS PATCH, 2026-07-29 (Pavel: "I need the app be functional while people
# might try to use it"). The heartbeat design reruns the WHOLE app once per tick
# (see mc_accumulate below), so the old 4-draws-per-tick × 100-draw cap meant **25 consecutive
# full-app reruns, ~30 s of continuous refreshing, after every single parameter change** — the
# behaviour Pavel reported as the app not responding. A change now costs FOUR reruns, and each
# does ~96% less compute: 20 draws at 5.9 ms against 100 at 115 ms.
#
# This is deliberately a strict subset of the ratified strip-down (Notes/widget_rebuild_design.md
# R1/R2): 20 draws is R1's fast tier and MC_DT is R2's coarse draw grid. What is NOT done here is
# the actual removal of the heartbeat, and R1's 200-draw "detailed" tier — both belong to the
# rebuild. Nothing in model.py, no Params default and no fixture is touched, so no ratified number
# moves and the suite needs no re-freeze.
MC_BATCH = 5           # ticks land at 5 / 10 / 15 / 20 — FOUR reruns per change, against 25 before
MC_MIN_GAP = 0.9       # min seconds between draws (so button clicks don't trigger a fresh batch)
MC_CAP = 20            # R1's fast tier. The 200-draw detailed tier arrives with the rebuild, as
                       # an explicit click that never carries over across a parameter change.
# WHY NOT ONE BATCH OF 20 (which is where R1 ends up)? Three tests assert that accumulation GROWS
# across reruns — test_background_mc_in_point_mode needs three successive increases (n1 < n2 < n3),
# test_charts_panel_always_mounted and test_charts_tab_switch_keeps_mc_ticking one each. Those
# assertions are the only available proxy for "the background precalc is running", since AppTest
# cannot drive the component's heartbeat. Filling the cap in one tick makes them false, and
# rewriting a test to go green during an urgent patch is precisely the move that has cost this
# project days. 5 keeps every assertion honest, fires both milestones (10, 20) exactly, and still
# removes 21 of the 25 reruns. The one-batch design lands with the rebuild, where the tests get
# rewritten deliberately to assert the new invariant instead of the retired one.
# The draw grid ONLY (R2): the point path keeps dt = 0.005. Measured at the Level-3 config, a draw
# costs 5.9 ms here against 115 ms on the fine grid, for a worst whole-series deviation of 2.9e-3
# — and fan width is set by parameter spread, orders of magnitude larger. The fine grid stays on
# the point path because Delta-dot(0) = 0 is exact only under constant growth: at L2/L3 it is
# ~1e-7 and coarsening degrades it ~450x, which the ratified-invariant test would (rightly) catch.
MC_DT = 0.2
# Bands and headline stats refresh only when n crosses these counts (then freeze until the next
# one) — so the charts sit still instead of blinking on every tick.
MC_MILESTONES = (10, 20)


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
    the embedded JSON small enough for the iframe srcdoc. For Y-SERIES ONLY — never the time
    grid (see _rlt)."""
    return [None if not np.isfinite(x) else float(f"{x:.4g}") for x in np.asarray(a, float)]


def _rlt(a):
    """The TIME grid, rounded to 4 DECIMALS — not significant figures. Under the calendar axis
    (D-076, t + 2026) four significant figures is integer years, which collapsed every point
    within a year onto one x and made the fans render as annual steps (Pavel's report,
    2026-07-27). Fixed decimals keep the ~20 points/yr the draws actually carry."""
    return [None if not np.isfinite(x) else round(float(x), 4) for x in np.asarray(a, float)]


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
    # D-047: the ONE stat the Finance tile shows. Computed from the SAME series the label and
    # the fan describe — earnings vs model-building cost at the horizon (audit X-03: the old
    # profit[-1] > 0 form was numerically identical under Π = E − B, but stat and label were
    # different objects and would have diverged silently under any profit-only extension).
    def _covers(dd):
        if "revenue" in dd and "cost" in dd:
            return (float(np.asarray(dd["revenue"], float)[-1])
                    > float(np.asarray(dd["cost"], float)[-1]))
        return float(np.asarray(dd["profit"], float)[-1]) > 0.0   # pre-D-068 record (reload)
    p_pos_T = float(np.mean([_covers(dd) for dd in draws]))
    # D-076: draws whose IMPLIED training-bill growth left the observed band [2.0, 2.9]×/yr were
    # rejected and redrawn inside mc_draw_batch. Report the count — a constraint that silently
    # truncates the joint prior would read as "we sampled everything" when we did not.
    n_rej = int(sum(int(dd.get("rejects", 0)) for dd in draws))
    stats = {"p_prof": float(np.isfinite(cr).mean()),
             "med_cr": (float(np.nanmedian(cr[np.isfinite(cr)])) if np.isfinite(cr).any()
                        else float("nan")),
             "n_blow": n_blow, "p_pos_T": p_pos_T, "n_rej": n_rej}
    # Two-hue scheme: each panel's bands are light/mid shades of the series' OWN hue; the median
    # is the saturated hue and its color never changes per draw. Gap = derived → grey family.
    # Entry: (key, title, ylab, median color, band hue, median dash, y-axis type).
    # D-080: the three money fans (profit, revenue, cost) collapsed into ONE coverage fan —
    # ρ_t = E_t/B_t in percent, the only identified finance outcome; break-even is the 100%
    # refline (the component draws `refline` as a dashed hline). "coverage" is DERIVED here
    # from the per-draw revenue/cost records — the notebook records are unchanged.
    # D-068 MC extension: fans for the Algo progress / Compute / Value tabs ride on the SAME
    # snapshot — the per-draw records already carry a_L/a_F, c_L/c_F and W_R/W_F (one series
    # per fan chart, leader vs follower stacked vertically per Pavel's no-side-by-side rule).
    CH = [("coverage", "Coverage — earnings ÷ model-building cost  (%)", "%",
           C_PROFIT, C_PROFIT, "solid", None),
          ("Delta", "Capability gap  Δ = xᴸ − xᶠ  (OOM)", "OOM", C_GAP_MED, C_GAP, "solid", None),
          ("a_L", "Algo progress — leader  aᴸ", "OOM above 2026 frontier",
           C_LEADER, C_LEADER, "solid", None),
          ("a_F", "Algo progress — follower  aᶠ", "OOM above 2026 frontier",
           C_FOLLOWER, C_FOLLOWER, "solid", None),
          ("c_L", "Compute — leader (frontier)  cᴸ", "OOM above 2026 frontier",
           C_LEADER, C_LEADER, "solid", None),
          ("c_F", "Compute — follower  cᶠ", "OOM above 2026 frontier",
           C_FOLLOWER, C_FOLLOWER, "solid", None),
          ("W_R", "Value index — leader  W(xᴸ)  (× today, log)", "× today  (log)",
           C_LEADER, C_LEADER, "solid", "log"),
          ("W_F", "Value index — fringe  W(xᶠ)  (× today, log)", "× today  (log)",
           C_FOLLOWER, C_FOLLOWER, "solid", "log")]
    charts = []
    nan_path = np.full_like(np.asarray(draws[0]["profit"], float), np.nan)

    def _series(dd, key):
        # coverage is DERIVED per draw (D-080): ρ = 100·revenue/cost — the records are unchanged
        if key == "coverage":
            with np.errstate(divide="ignore", invalid="ignore"):
                return 100.0 * (np.asarray(dd.get("revenue", nan_path), float)
                                / np.asarray(dd.get("cost", nan_path), float))
        # .get guard: a store carried across a code reload may hold pre-D-068 records
        return np.asarray(dd.get(key, nan_path), float)

    for key, title, ylab, color, bhue, mdash, ytype in CH:
        A = np.array([_series(dd, key) for dd in draws])
        A = np.where(np.isfinite(A), A, np.nan)
        if np.isnan(A).all():
            lo5 = lo25 = med = hi75 = hi95 = nan_path
        else:
            lo5, lo25, med, hi75, hi95 = np.nanpercentile(A, [5, 25, 50, 75, 95], axis=0)
        # y-range fits the dark 25-75% band + median (upper-tail draws sit off-frame), as before.
        fin = np.concatenate([lo25[np.isfinite(lo25)], hi75[np.isfinite(hi75)],
                              med[np.isfinite(med)]])
        if ytype == "log":
            fin = fin[fin > 0.0]
        if fin.size:
            lo_v, hi_v = float(fin.min()), float(fin.max())
            if ytype == "log":     # Plotly log-axis ranges are in log10 units
                lo_v, hi_v = np.log10(lo_v), np.log10(hi_v)
            span = (hi_v - lo_v) or 1.0
            yr = [lo_v - 0.08 * span, hi_v + 0.08 * span]
        else:
            yr = [0.0, 1.0]
        refline = None
        if key == "coverage":
            # 100% = break-even must stay inside the range (the component draws `refline`
            # as a dashed annotated hline) even when every band is single-signed
            yr = [min(yr[0], 90.0), max(yr[1], 110.0)]
            refline = dict(y=100.0, text="break-even (100%)")
        charts.append(dict(key=key, title=title, ylab=ylab, color=color, mdash=mdash,
                           ytype=ytype, refline=refline,
                           band_light=_rgba(bhue, 0.12), band_mid=_rgba(bhue, 0.30), yrange=yr,
                           lo5=_rl(lo5), lo25=_rl(lo25), med=_rl(med), hi75=_rl(hi75),
                           hi95=_rl(hi95)))
    # (no per-draw embedding: the inspected sample's series are injected at render time by
    # mc_render, and its parameter values surface as dashed ticks on the sidebar range controls)
    next_ms = next((msn for msn in MC_MILESTONES if msn > n), None)
    # calendar x-axis (D-076): the fans share the point charts' axis convention, 2026 + t
    store["snapshot"] = dict(t=_rlt(t + YEAR0), charts=charts, n=n, next_ms=next_ms,
                             stopped=store["stopped"], stats=stats, show_blowup=bool(show_blowup),
                             show_horizon=bool(show_horizon))
    store["band_n"] = n
    store["_next_ms"] = next_ms


# The panel is ONE bidirectional custom component (widget/mc_component/index.html): the four
# Plotly charts plus the minimal in-panel corner controls (D-042) — a faint "n = k out of MC_CAP"
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
    """Sampled values of the currently inspected MC draw (empty when not inspecting) — the
    sidebar renders these as dashed ticks on the range controls, keyed by DIAL key.

    The coverage dimension needs the inverse of `mc_prepare`'s forward map: the dial is app-side
    and in PERCENT (D-080), while the sampler draws the Params field `rho` — a fraction — so
    every stored draw carries `rho` and never `cov0`. Without the map back, the one drawn MONEY
    dial — the headline output's own row — is the only row whose ⊙ tick never appears
    (2026-07-28 functionality test F-4). Since D-093 the map is the unit conversion alone: the
    hidden constant k it used to divide through is gone with the (R₀, m, k) triple."""
    store = st.session_state.get("_mc_store") or {}
    show, i = _mc_inspection(store)
    if not show:
        return {}
    out = dict(store["draws"][i]["params"])
    if "rho" in out and "cov0" not in out:
        out["cov0"] = 100.0 * float(out["rho"])
    return out


# Tile order (D-043 + Pavel's HARD RULE: never place graphs side by side). D-080: the FINANCE
# instance shows the ONE coverage fan — Pavel: "show only the coverage in the financial graphs
# to bring attention to this most important output" (D-093 removed the dollar scale outright,
# so a nominal view would now have to re-introduce it deliberately). The MODEL-PATH instance shows the capability-gap fan.
# Indices refer to the snapshot's chart list (coverage, Delta, a_L, a_F, c_L, c_F, W_R, W_F —
# built in _mc_refresh_snapshot). D-068 MC: the extra level-gated tabs reuse the SAME display
# instance with their own grid.
FIN_GRID = [[0]]
PATH_GRID = [[1]]
ALGO_GRID = [[2], [3]]
COMP_GRID_L = [[4]]          # < L3: the follower's compute is not modeled yet
COMP_GRID_BOTH = [[4], [5]]
VALUE_GRID = [[6], [7]]


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
        sampled = {k: _rl(dd[k]) for k in ("Delta", "a_L", "a_F", "c_L", "c_F", "W_R", "W_F")
                   if k in dd}
        if "revenue" in dd and "cost" in dd:   # the coverage fan's inspected series (D-080)
            with np.errstate(divide="ignore", invalid="ignore"):
                sampled["coverage"] = _rl(100.0 * np.asarray(dd["revenue"], float)
                                          / np.asarray(dd["cost"], float))
    return dict(t=(snap["t"] if snap else []), charts=(snap["charts"] if snap else []),
                grid=grid, corner=bool(corner), heartbeat=bool(heartbeat),
                visible=bool(visible and snap is not None),
                n=int(snap["n"]) if snap else 0, cap=int(MC_CAP),
                n_live=int(len(store.get("draws") or [])),
                done=bool(store.get("stopped")), theme=_mc_theme(), sample=C_SAMPLE,
                sampled=sampled, idx=int(i), epoch=int(store["epoch"]))


# The one-line MC explainer (round 3): the "How to read the Monte-Carlo fans" expander is
# gone — its content, condensed, lives in the headline's native help tooltip instead.
_MC_HELP = (
    "**Share of draws whose coverage ρ = earnings ÷ model-building cost exceeds 100% at the "
    "horizon $T$** (equivalently: whose profit flow is positive there — "
    "$\\Pi_t > 0 \\iff \\rho_t > 1$).\n\n"
    "**How the Monte-Carlo fans work.** Draws accumulate live over the documented sampling "
    "ranges; each fan shows the **median**, a mid-shade **25–75%** band and a light **5–95%** "
    "band. Each parameter row's TICK decides what the draws do with it: ticked, they sweep "
    "its trim crop — targets in natural units (inverted per draw), free dials in parameter "
    "space; unticked, every draw pins it at the spot value. Default ranges are tight (the "
    "span of the sources); single-source dimensions start unticked until you tick and widen "
    "them. y-axes fit the mid 25–75% band, so extreme upper-tail draws sit off-frame.\n\n"
    "**Coherence.** Compute growth and price-performance are drawn independently, but their "
    "difference is itself observed — the training bill grows 2.4×/yr (90% CI [2.0, 2.9]). Draws "
    "implying a bill outside that band are rejected and redrawn; the count is reported beside "
    "the stat rather than silently dropped.\n\n"
    f"Bands and this stat refresh only at round draw "
    f"counts ({', '.join(str(_n) for _n in MC_MILESTONES)}) and drawing stops at {MC_CAP}; "
    f"once done, the **⊙ control** in "
    "the chart corner steps through inspected draws (dashed ticks on the sidebar range "
    "controls). Any value, range or mode change restarts the accumulation. Profit is an "
    "undiscounted yearly flow.")


def mc_headline(mc_key, show_blowup=True):
    """The ONE compact stat the Finance tile shows in Monte-Carlo mode (D-047 declutter):
    'MC simulation: (?) <pct>% profitable at T'. The native help tooltip carries the whole
    how-to-read explainer (round 3 — it replaced the expander). Frozen at milestone counts
    like the bands; everything else the tile used to show is gone — the graphs carry it."""
    store = _mc_store(mc_key)
    snap = store.get("snapshot") if store else None
    if snap is None:
        st.markdown("**MC simulation:** starting the draws…", help=_MC_HELP)
        return
    pct = snap["stats"].get("p_pos_T", float("nan")) * 100.0
    n_rej = int(snap["stats"].get("n_rej", 0))
    if np.isfinite(pct):
        txt = f"**MC simulation:** {pct:.0f}% of draws cover their build cost at $T$"
        if n_rej:
            txt += (f" &nbsp;:gray[· {n_rej} draw{'s' if n_rej != 1 else ''} rejected "
                    "(coherence)]")
        st.markdown(txt, help=_MC_HELP)


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


def mc_panel_path(mc_key, grid=PATH_GRID):
    """The display-only panel instance (no corner UI, no heartbeat), mounted only while the
    Monte-Carlo mode is visible. One tab shows at a time, so every non-finance tab reuses this
    SAME instance with its own `grid` — the capability-gap fan (default), or the D-068 fans
    (ALGO_GRID / COMP_GRID_* / VALUE_GRID); a grid change rebuilds the chart divs in place."""
    store = _mc_store(mc_key)
    snap = store.get("snapshot") if store else None
    if snap is None:
        st.info("Starting the Monte-Carlo draws…")
        return
    _MC_PANEL(data=_mc_payload(store, snap, grid=grid, corner=False, heartbeat=False,
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
        # R2: the coarse grid is applied HERE, at the draw call, not to Params.dt — because
        # ui/levels.py:150 does d["dt"] = P0.dt, so moving the default would propagate into the
        # level pins and shift the frozen fingerprints. mc_draw_batch takes T and dt from p_base,
        # so replacing it on this one object confines the change to the draws.
        p = dataclasses.replace(m.Params(**dict(params_items)), dt=MC_DT)
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
    sampled set — so ANY change (spot values, crop ends, crop collapses, [choose], resets,
    level switches) restarts the accumulation from n=0.

    Returns (mc_key, sample_keys, merge_delta, target_ranges, param_ranges)."""
    from .levels import level_sample_keys, merged_delta
    from .state import (_active_ranges, _active_rng, _mc_dim_editable, _mc_sampled,
                        _sampled_on)

    def _pinned_dim(k):
        """A dimension leaves the sampled set when its per-parameter MC tick is OFF (D-079
        rider: unticked = every draw pins it at the spot) or its trim crop is COLLAPSED TO A
        POINT (point-default dims start with both handles on the spot until widened). Since
        the X-10 share dial, scale_of dims (g_a_F) are editable like any uniform band — in
        share units; only choice-kind dims (the eta menu) keep the no-lane fallback: with
        the tick on they sample exactly when a default distribution exists, else stay
        pinned."""
        if not _mc_dim_editable(k):
            return not _sampled_on(k) or _active_rng(k)[0] is None
        return not _mc_sampled(k)

    sample_keys = [k for k in level_sample_keys(LEVEL) if not _pinned_dim(k)]
    merge_delta = merged_delta(LEVEL)   # named predicate (X-18): one renumbering-safe gate
    _over = st.session_state.get("_range_over", {})
    tro, pro = _active_ranges()
    # D-080: the COVERAGE dimension is app-side (state.APP_RANGES) and in PERCENT, while the
    # Params field `rho` is the fraction — so the crop is converted here and the sampler draws
    # `rho`. A uniform crop in percent IS a uniform draw in the fraction (the map is linear),
    # which is what lets the trim lane mean exactly what it shows. D-093: one draw dimension
    # because there is now one money PARAMETER, not because two of three were pinned.
    if "cov0" in sample_keys:
        from .state import _active_rng, _tbounds_of
        lo, hi = _tbounds_of(_active_rng("cov0")[0])
        pro["rho"] = ("uniform", lo / 100.0, hi / 100.0)
        sample_keys = ["rho" if k == "cov0" else k for k in sample_keys]
    mc_key = ((LEVEL,) + tuple(sorted(d.items()))
              + ("RANGES",) + tuple(sorted(_over.items()))
              + ("KEYS",) + tuple(sample_keys))
    return mc_key, sample_keys, merge_delta, tro, pro
