"""The D-043 main area. Layout left→right (after the native sidebar):

    [docked calibration panel (when open)] | tabbed pane [Introduction | Equations]
                                             (folds into a thin strip while the panel is open)
                                           | PINNED chart tiles — Finance FIRST, Model path
                                             second, then the level-gated extras.

The chart tiles render the point trajectory or the Monte-Carlo fan per the top-bar mode; the
Monte-Carlo accumulation runs on EVERY run (background precalc — the finance panel component
stays mounted hidden in Point-forecast mode), so switching modes is instant.
D-044: the release-delay and under-the-hood sections are unreachable while MAX_LEVEL = 6 but
kept intact (reversible removal).
"""
import re
from dataclasses import fields

import numpy as np
import pandas as pd
import streamlit as st

import notebook_loader

from . import calpanel, theme
from .content import GRADES, NOTATION_SECTIONS
from .equations import equations_panel
from .levels import level_card
from .mc import (MC_CAP, mc_accumulate, mc_headline, mc_panel_fin, mc_panel_path, mc_prepare)
from .model_access import NB, m
from .simcache import delay_cached, sim_window_cached
from .state import _reg, cal_open, close_cal, mc_active
from .theme import (C_COST, C_FOLLOWER, C_GAP, C_LEADER, C_PROFIT, C_PSI, C_REV,
                    C_SERVED, NEUTRAL, PAL, TAU_RAMP, WINDOW_COLS, fig_base, line,
                    show, tab_intro)


# ======================================================================= diagnostics
def _warnings(sim, LEVEL):
    """Blow-up / consistency-cap diagnostics (top of the charts column)."""
    cap_ok = bool(sim["cap_xF_le_xR"].all() and sim["cap_W"].all())
    warns = []
    if LEVEL >= 3:   # γ (and thus the ψ blow-up) can only be non-trivial with the growth engine
        xL_max = float(np.nanmax(sim["x_L"]))
        if xL_max > 25.0:
            blow_t = float(sim["t"][np.argmax(sim["x_L"] > 25.0)])
            warns.append(f"BLOW-UP: the leader path passes +25 OOM at t = {blow_t:.1f} yr — the "
                         f"$\\psi$ feedback has gone super-exponential (finite-time singularity, "
                         f"spec N4). Curves beyond that point are meaningless; lower "
                         f"$\\gamma$/$\\rho_0$ or freeze AI assistance.")
    if not cap_ok:
        warns.append("Consistency cap hit (spec N2): somewhere the follower's capability $x^F$ "
                     "exceeds the model the leader serves, or the served model is worth less than "
                     "the follower's — cases the model isn't meant to cover. The affected terms "
                     "are floored where this bites, so read those stretches with caution.")
    for w in warns:
        st.warning(w)


# (the Point-mode verdict/metric cards and the sanity-check line were removed by D-047 —
# the graphs carry the profitability story; the notebook's headline() is still computed for
# the level-gated under-the-hood section)


# ======================================================================= tabbed left pane
def _tabbed_pane(d, p, LEVEL):
    """Full-height left pane (D-043 amendment): [Introduction | Equations] tabs — no vertical
    stacking that pushes the equations below the fold; the charts stay pinned on the right."""
    with st.container(key="mainpane"):
        # the widget key is GC'd while the pane is folded behind the calibration panel, so the
        # active tab is mirrored in a plain shadow key and reseeded from it on reopen (QA N2)
        _reg("pane_tab", st.session_state.get("_pane_tab_mem", "Introduction"))
        tab = st.segmented_control("Pane", ["Introduction", "Equations"], key="pane_tab",
                                   label_visibility="collapsed") or "Introduction"
        st.session_state["_pane_tab_mem"] = tab
        if tab == "Introduction":
            st.caption("A **leader** (the frontier lab(s)) races ahead while a **follower** "
                       "(open-source / competitive fringe) catches up. **This explorer is "
                       "layered:** raise the level (top bar) to add one mechanism at a time.")
            level_card(LEVEL, d)
            with st.expander("Notation & conventions — grows with the level", expanded=False):
                st.markdown("\n\n".join(NOTATION_SECTIONS[L] for L in range(1, LEVEL + 1)))
        else:
            equations_panel(LEVEL, d, p)


def _pane_strip():
    """The thin vertical strip the tabbed pane folds into while the calibration panel is open
    (variant A2); clicking it closes the panel and reopens the pane."""
    with st.container(key="eqstrip"):
        st.button("Introduction · Equations ▸", key="strip_btn", on_click=close_cal,
                  help="close the calibration panel and reopen this pane")


# ======================================================================= chart tiles
def _finance_tile(d, sim, hl, LEVEL, mode_mc, mc_key):
    """Finance FIRST (D-043 — it is the final output). Point trajectory or MC fan per mode.
    Returns True when the MC corner reported an inspection change (caller reruns)."""
    need_rerun = False
    with st.container(border=False):
        # D-054 (round 2): no section header (the group switch names it) and no how-to-read —
        # each chart title is self-explanatory instead. D-056: border=False — the fixed panel
        # already frames the graphs; the bordered card was a redundant box (and inset padding).
        if mode_mc:
            mc_headline(mc_key, show_blowup=(LEVEL >= 3))
            need_rerun = mc_panel_fin(mc_key, visible=True, show_blowup=(LEVEL >= 3))
        else:
            f = fig_base("Leader profit — revenue minus cost  Π  ($B/yr)", "year",
                         "$/yr  ($B)", height=230)
            line(f, sim["t"], sim["profit"], "profit  Π", C_PROFIT)
            # no plotted break-even line — the axis zeroline marks 0; keep 0 inside the y-range
            if float(np.min(sim["profit"])) >= 0.0 or float(np.max(sim["profit"])) <= 0.0:
                f.update_yaxes(rangemode="tozero")
            show(f, key="pt_profit")
            f = fig_base("Revenue (gap rent) vs compute cost  ($B/yr)", "year", "$/yr  ($B)",
                         height=230)
            line(f, sim["t"], sim["revenue"], "revenue  (gap rent θ·ΔW)", C_REV)
            line(f, sim["t"], sim["cost"], "cost", C_COST, dash="dash")
            show(f, key="pt_revcost")
    return need_rerun


def _capability_tile(d, sim, LEVEL, mode_mc, mc_key):
    """Capability paths + the gap Δ (and, with the growth engine, the RSI-feedback share ψ that
    rides with the gap). Point paths, or the capability-gap fan in Monte-Carlo mode.
    D-068: the algo-progress and compute graphs moved out to their own level-gated tabs."""
    served = d["tau"] > 0.0                 # x^R differs from x^L only under a release delay
    show_growth = LEVEL >= 3
    with st.container(border=False):
        # D-054 (round 2): no section header (the tab switch names it) and no how-to-read —
        # each chart title is self-explanatory instead. D-056: border=False (see _finance_tile).
        if mode_mc:
            mc_panel_path(mc_key)
            st.caption("The gap $\\Delta$ is what the leader earns rent on — its forecast fan "
                       "drives the Finance fans above. y-axis in OOM above today's frontier.")
            return

        cap_ttl = ("Capability over time — developed, served & follower  x" if served
                   else "Capability over time — leader vs follower  x")
        f = fig_base(cap_ttl, "year", "OOM above 2026 frontier", height=230)
        line(f, sim["t"], sim["x_L"], "leader" + (" developed" if served else "") + "  xᴸ",
             C_LEADER)
        if served:
            line(f, sim["t"], sim["x_R"], "leader served  xᴿ", C_SERVED, dash="dash")
        line(f, sim["t"], sim["x_F"], "follower  xᶠ", C_FOLLOWER)
        show(f, key="pt_cap")

        gap_ttl = ("Capability gap  Δ  &  RSI-feedback share  ψ" if show_growth
                   else "Capability gap — how far ahead the leader is  Δ = xᴸ − xᶠ")
        gap_ylab = "OOM (gap)   /   share (ψ)" if show_growth else "OOM"
        f = fig_base(gap_ttl, "year", gap_ylab, height=230)
        # anchor at 0 so a (near-)constant gap reads flat instead of auto-zooming into
        # integrator-precision noise on the y-axis
        f.update_yaxes(rangemode="tozero")
        line(f, sim["t"], sim["Delta"], "gap  Δ = xᴸ − xᶠ  (OOM)", C_GAP)
        if show_growth:
            line(f, sim["t"], sim["psi_share"], "ψ-share (fraction of algo progress from RSI)",
                 C_PSI, dash="dot")
            f.add_hline(y=0.25, line=dict(color=NEUTRAL, dash="dash", width=1),
                        annotation_text="ψ-share 25% (feedback no longer small)",
                        annotation_position="top left", annotation_font_size=10)
        show(f, key="pt_gap")


def _algo_tile(sim):
    """Algorithmic-progress paths a(t) — its own tab from L3 (D-068: moved out of Capability,
    where it crowded the gap graph). Both actors, unchanged data: catch-up flows through the
    algorithmic channel, so the follower's a can overtake the leader's while its total x trails."""
    with st.container(border=False):
        f = fig_base("Algorithmic progress — leader vs follower  a(t)", "year",
                     "OOM above 2026 frontier", height=230)
        line(f, sim["t"], sim["a_L"], "leader  aᴸ", C_LEADER)
        line(f, sim["t"], sim["a_F"], "follower  aᶠ", C_FOLLOWER)
        show(f, key="pt_algo")
        st.caption("Catch-up flows through the *algorithmic* channel, so the follower's $a$ can "
                   "overtake the leader's $a$ while its total capability $x$ still trails — the "
                   "compute deficit is what keeps the gap open.")


def _compute_tile(sim, LEVEL):
    """Compute paths c(t) — its own tab from L4 (D-068). The follower's compute is NOT modeled
    before L6, so L4–L5 plots ONLY the leader's frontier compute; L6 (catch-up channels) adds
    the follower's compute line alongside the leader's (Pavel's explicit instruction)."""
    both = LEVEL >= 6
    with st.container(border=False):
        ttl = ("Compute — leader vs follower  c(t)" if both
               else "Compute — leader (frontier)  c(t)")
        f = fig_base(ttl, "year", "OOM above 2026 frontier", height=230)
        line(f, sim["t"], sim["c_L"], "leader  cᴸ", C_LEADER)
        if both:
            line(f, sim["t"], sim["c_F"], "follower  cᶠ", C_FOLLOWER)
        show(f, key="pt_comp")
        st.caption(
            "Compute is the capital-intensive engine behind capability. "
            + ("The follower's own compute enters the model at this level, plotted alongside "
               "the leader's frontier compute." if both
               else "The follower's own compute is not modeled until the catch-up-channels "
                    "level (6); here only the leader's frontier compute is shown."))


def _value_tile(sim):
    """Value flows W over the horizon — its own tab from L5 (D-068). LOG y-axis: W(x) grows
    ~exponentially with capability, so on a log scale the value levels read off as slopes.
    x-axis is the horizon (year) — the quantity the model exposes cleanly (W_R, W_F are already
    integrated per t), and it keeps the tab consistent with the other time-series tabs. The two
    lines are the served-leader value W(xᴿ) and the follower value W(xᶠ), whose gap the leader
    earns rent on."""
    with st.container(border=False):
        f = fig_base("Value over time — leader served vs follower  W  ($B/yr, log)",
                     "year", "$/yr  ($B, log scale)", height=230)
        line(f, sim["t"], sim["W_R"], "leader served  W(xᴿ)", C_LEADER)
        line(f, sim["t"], sim["W_F"], "follower  W(xᶠ)", C_FOLLOWER)
        f.update_yaxes(type="log")
        show(f, key="pt_value")
        st.caption("Each actor's capability commands a dollar value $W(x)$ (\\$B/yr); the leader "
                   "earns rent on the **gap** $W(x^R) - W(x^F)$ between the two lines. Log "
                   "y-axis: near-exponential value growth reads as straight-line slopes.")


def _delay_section(d, items, LEVEL):
    """Release delay (Level 7 — UNREACHABLE while MAX_LEVEL = 6, kept for the D-044 revert).
    Same graph options in both modes; computed at the spot values either way."""
    with st.expander("Release delay — does withholding the frontier model pay?",
                     expanded=False):
        tab_intro(
            "The leader can serve an older model than it has (a release delay $\\tau$). "
            "Withholding forgoes revenue now but slows the follower's distillation. These "
            "profit paths show the trade-off for a constant delay (left) and for a one-month "
            "delay switched on only inside a window (right).",
            "Curves are **undiscounted profit per year**, not a single NPV number. The default "
            "**relative** view plots each delay's profit *minus* the no-delay run — the "
            "informative quantity, since delays move profit by a few \\$B against a ~\\$1000B "
            "absolute level (switch to **Absolute** for context). Below the zero line the "
            "delay is costing money; a rise back above it (the spike at a window's end) is the "
            "withheld model being released. Delay is capped at 3 months (the policy-relevant "
            "range); the horizon follows the toggle at the top of the sidebar.")
        if d["delta_rel"] == 0.0:
            st.info("$\\delta_{rel} = 0$ → **distillation disabled**: the released-model "
                    "channel is off, so delaying release has no effect on the follower and "
                    "every curve below coincides. This happens when the follower's own engine "
                    "already covers the leader's speed (no wedge left for the release channel) "
                    "— lower the follower's own rates or the lag to make the trade-off bite.")

        view = st.radio("View", ["Relative to no delay", "Absolute"], horizontal=True,
                        key="delay_view",
                        help="Delays change profit by only a few \\$B against a ~\\$1000B "
                             "absolute level, so the absolute curves overlap. The relative "
                             "view plots the difference from the no-delay run — the "
                             "informative quantity.")
        rel = view == "Relative to no delay"
        ylab = "profit vs no-delay  ($B/yr)" if rel else "profit Π  ($B/yr)"
        zero_txt = "no-delay baseline" if rel else "break-even"

        taus = (0.0, 1 / 12, 2 / 12, 3 / 12)
        dc = delay_cached(items, taus)
        base0 = np.asarray(dc["paths"][0])        # the τ = 0 profit path — the no-delay baseline
        # the two figures stack VERTICALLY (Pavel's hard rule: never place graphs side by side)
        ttl = ("(a) Constant delay: profit vs no delay" if rel
               else "(a) Constant delay: profit Π")
        f = fig_base(ttl, "year", ylab)
        for path, tau, col in zip(dc["paths"], taus, TAU_RAMP):
            if rel and tau == 0.0:
                continue   # identically zero in the relative view — the baseline shows it
            label = ("τ = 0 (release now)" if tau == 0
                     else f"τ = {int(round(tau * 12))} mo delay")
            y = (np.asarray(path) - base0) if rel else path
            line(f, dc["t"], y, label, col)
        f.add_hline(y=0, line=dict(color=PAL["red"], dash="dash", width=1),
                    annotation_text=zero_txt, annotation_position="bottom left",
                    annotation_font_size=10)
        show(f, key="pt_delay_a")
        ttl = ("(b) One-month delay in a window: profit vs no delay" if rel
               else "(b) One-month delay in a window: profit Π")
        f = fig_base(ttl, "year", ylab)
        if not rel:
            line(f, dc["t"], base0, "no delay (baseline)", NEUTRAL, dash="dash")
        for t0, col in zip((0.0, 2.0, 5.0), WINDOW_COLS):
            sw = sim_window_cached(items, t0, 2.0, 1.0)
            y = (np.asarray(sw["profit"]) - base0) if rel else sw["profit"]
            line(f, sw["t"], y, f"delay in years {t0:.0f}–{t0 + 2:.0f}", col)
            f.add_vrect(x0=t0, x1=min(t0 + 2, d["T"]), line_width=0, fillcolor=col,
                        opacity=0.06)
        f.add_hline(y=0, line=dict(color=PAL["red"], dash="dash", width=1),
                    annotation_text=zero_txt, annotation_position="bottom left",
                    annotation_font_size=10)
        f.update_xaxes(range=[0, d["T"]])  # keep the window figure on the shared horizon
        show(f, key="pt_delay_b")
        st.caption("(a) A constant delay shifts the whole revenue schedule down, because "
                   "revenue is earned on the *served* model $x^R$, not the developed model "
                   "$x^L$. (b) The same one-month delay costs most when imposed **early** "
                   "(capability is scarcest and the follower is closest); when each shaded "
                   "window ends, the withheld capability is released and profit briefly spikes "
                   "— the catch-up release.")


def _hood_section(p, hl):
    """Under the hood (Level 9 — UNREACHABLE while MAX_LEVEL = 6, kept for the D-044 revert):
    the live loaded notebook sources. Not an expander — its content nests expanders."""
    st.subheader("Under the hood")
    st.caption("All model code is loaded live from `model_notebook.ipynb` (the single source "
               "of truth). Sources of every exported cell, in order:")
    with st.expander("Parameter table (grades from calibration_master)", expanded=False):
        rows = []
        for fld in fields(m.Params):
            val = getattr(p, fld.name)
            rows.append(dict(param=fld.name,
                             current=f"{val:g}" if isinstance(val, float) else str(val),
                             grade=GRADES.get(fld.name, "-")))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320,
                     hide_index=True)
    st.markdown(f"**Implied t=0 operating profit** $\\theta\\,\\Delta W$ = "
                f"\\${hl['op_profit_t0']:,.0f}B/yr vs observed ~\\$40–60B/yr.")
    for idx, src in notebook_loader.export_sources(NB):
        names = []   # top-level def/class/CONSTANT names, so the label says what the cell holds
        for ln in src.splitlines():
            if ln.startswith(("def ", "class ")):
                names.append(ln.split()[1].split("(")[0].rstrip(":"))
            elif re.match(r"^[A-Z][A-Z0-9_]*\s*=", ln):
                names.append(ln.split("=")[0].strip())
        label = f"export cell (notebook position {idx})"
        if names:
            label += "  —  " + ", ".join(names)
        with st.expander(label):
            st.code(src, language="python")


def _charts_column(d, items, sim, hl, p, LEVEL, mode_mc, mc_key, sample_keys):
    """The right chart panel: a fixed drag-collapsible column (D-050 — the divider, the » and
    the collapsed strip are CLIENT-side, injected by theme.inject_frontend_js; the server
    always renders this column, so collapsing only CSS-hides it and the MC finance component
    stays mounted with its background accumulation ticking). Content: a Finance | Model path
    switch (one tile group at a time — Pavel; charts inside a group still stack vertically),
    with warnings on top and the level-gated extras under Finance."""
    with st.container(key="chartscol"):
        # ---- the tab switch (same segmented idiom as the middle pane's tabs); the shadow
        # mem key is belt-and-braces for widget GC, mirroring the pane_tab pattern.
        # D-068: five tabs, introduced progressively by level. Financial + Capability are always
        # present; Algo progress from L3, Compute from L4, Value from L5 (Compute gains the
        # follower line at L6). D-054 (round 2): the switch already names the visible tab, so
        # there are no per-tab section headers.
        tab_labels = ["Financial", "Capability"]
        if LEVEL >= 3:
            tab_labels.append("Algo progress")
        if LEVEL >= 4:
            tab_labels.append("Compute")
        if LEVEL >= 5:
            tab_labels.append("Value")
        default_tab = st.session_state.get("_charts_tab_mem", "Financial")
        if default_tab not in tab_labels:
            default_tab = "Financial"
        _reg("charts_tab", default_tab)
        # a level DROP can leave the persisted widget value pointing at a now-hidden tab, which
        # would make st.segmented_control raise — reseed it to a still-valid tab first
        if st.session_state.get("charts_tab") not in tab_labels:
            st.session_state["charts_tab"] = default_tab
        tab = st.segmented_control("Charts", tab_labels, key="charts_tab",
                                   label_visibility="collapsed",
                                   help="One graph tab at a time; more tabs unlock as the level "
                                        "rises. The Monte-Carlo accumulation keeps running "
                                        "whichever is shown.") \
            or default_tab
        st.session_state["_charts_tab_mem"] = tab
        fin_vis = tab == "Financial"
        _warnings(sim, LEVEL)
        if mode_mc:
            with st.expander("How to read the Monte-Carlo fans", expanded=False):
                st.markdown(
                    "Draws accumulate live across the sampling ranges and are summarised as a "
                    "**median line with a mid-shade 25–75% band and a light 5–95% band**. At "
                    f"**Level {LEVEL}** the **{len(sample_keys)}** dimension(s) with a "
                    "documented range are sampled — targets in natural units (inverted per "
                    "draw), free dials in parameter space; everything else is pinned at its "
                    "current value. **Default sampling ranges are tight** (the span of the "
                    "documented sources); single-source dimensions start **pinned at a point** "
                    "until you widen their range. y-axes fit the mid 25–75% band, so extreme "
                    "upper-tail draws sit off-frame. Bands and stats **refresh only at round "
                    f"draw counts** (10, 20, 50, 100); drawing stops at {MC_CAP}. Once done, "
                    "the **⊙ control** in the finance panel's corner steps through inspected "
                    "draws (dashed ticks on the sidebar range controls). Any value/range/mode "
                    "change restarts the accumulation. Profit is an undiscounted yearly flow.")
        need_rerun = False
        if fin_vis:
            need_rerun = _finance_tile(d, sim, hl, LEVEL, mode_mc, mc_key)
            if LEVEL >= 7:
                _delay_section(d, items, LEVEL)   # profit trade-off → rides with Finance
            if LEVEL >= 9:
                _hood_section(p, hl)
        elif tab == "Capability":
            _capability_tile(d, sim, LEVEL, mode_mc, mc_key)
        elif tab == "Algo progress":
            _algo_tile(sim)                       # point paths (no MC fan for a(t))
        elif tab == "Compute":
            _compute_tile(sim, LEVEL)             # leader only < L6; +follower at L6
        elif tab == "Value":
            _value_tile(sim)                      # W over the horizon, log y-axis
        if not (mode_mc and fin_vis):
            # CRITICAL invariant: exactly ONE finance-component mount per run. It is visible
            # only on (MC mode ∧ Finance tab); in every other state it mounts HIDDEN so its
            # heartbeat keeps the background accumulation ticking — switching tab or mode
            # then shows a ready fan instantly.
            need_rerun = mc_panel_fin(mc_key, visible=False,
                                      show_blowup=(LEVEL >= 3)) or need_rerun
        if need_rerun:
            st.rerun()   # corner click: propagate the inspected draw to the sidebar ticks


# ======================================================================= main entry
def render_main(d, items, sim, hl, p, LEVEL):
    """Everything below the frozen top bar (D-047 layout): [calibration panel + fold strip
    (when open) | middle tabbed pane (flex)] | right chart panel (fixed, collapsible).
    Column ratios are cosmetic — the theme CSS pins the real widths."""
    mode_mc = mc_active()
    open_key = cal_open()
    # ---- background Monte-Carlo accumulation runs on EVERY run (D-043) ---------------------
    mc_key, sample_keys, merge_delta, tro, pro = mc_prepare(d, LEVEL)
    mc_accumulate(items, mc_key, tuple(sample_keys), merge_delta, show_blowup=(LEVEL >= 3),
                  show_horizon=(LEVEL >= 5), target_ranges=tro, param_ranges=pro)
    if open_key:
        theme.inject_cal_emphasis_css(calpanel.param_row_key(open_key))
        pcol, scol, ccol = st.columns([1.0, 0.12, 0.5], gap="small")
        with pcol:
            calpanel.render(d, p)
        with scol:
            _pane_strip()
    else:
        # (D-055) the equations→parameters highlight is now HOVER-driven and fully client-side
        # (desktop only; see theme.inject_frontend_js), so there is no per-run CSS injection or
        # autoscroll here — and `hl` stays the headline dict the chart panel needs.
        lcol, ccol = st.columns([1.0, 0.5], gap="small")
        with lcol:
            _tabbed_pane(d, p, LEVEL)
    with ccol:
        _charts_column(d, items, sim, hl, p, LEVEL, mode_mc, mc_key, sample_keys)
