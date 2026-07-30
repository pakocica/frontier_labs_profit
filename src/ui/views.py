"""The D-043 main area. Layout left→right (after the native sidebar):

    [docked calibration panel (when open)] | Equations pane (Introduction tab retired —
                                             Pavel, round 2; folds into a thin strip while
                                             the panel is open)
                                           | PINNED chart tiles — Finance FIRST, Model path
                                             second, then the level-gated extras.

The chart tiles render the point trajectory or the Monte-Carlo fan per the top-bar mode; the
Monte-Carlo accumulation runs on EVERY run (background precalc — the finance panel component
stays mounted hidden in Point-forecast mode), so switching modes is instant.
D-081 ladder amendment (Pavel, 2026-07-27): the ladder is exactly THREE levels — the old
release-delay / cost-mechanism / extensions tail is RETIRED from the widget (not parked); the
release-delay and under-the-hood sections were deleted here, their model content parked in
the spec.
"""
import numpy as np
import streamlit as st

from . import calpanel, theme
from .content import LEVEL_INTRO, _sub_live
from .equations import equations_panel
from .mc import (ALGO_GRID, COMP_GRID_BOTH, COMP_GRID_L, VALUE_GRID, mc_accumulate,
                 mc_headline, mc_panel_fin, mc_panel_path, mc_prepare)
from .state import _reg, cal_open, close_cal, mc_active
from .theme import (C_FOLLOWER, C_GAP, C_LEADER, C_PROFIT, C_PSI, C_SERVED, NEUTRAL, PAL,
                    cal, fig_base, line, show)


# ======================================================================= diagnostics
def _warnings(sim, LEVEL):
    """Blow-up / consistency-cap diagnostics (top of the charts column)."""
    cap_ok = bool(sim["cap_xF_le_xR"].all() and sim["cap_W"].all())
    warns = []
    if LEVEL >= 2:   # γ (and thus the ψ blow-up) can only be non-trivial with the growth engine
        xL_max = float(np.nanmax(sim["x_L"]))
        if xL_max > 25.0:
            blow_t = float(sim["t"][np.argmax(sim["x_L"] > 25.0)])
            warns.append(f"BLOW-UP: the leader path passes +25 OOM at t = {blow_t:.1f} yr — the "
                         f"$\\psi$ feedback has gone super-exponential (finite-time singularity, "
                         f"spec N4). Curves beyond that point are meaningless; lower "
                         f"$\\gamma$/$\\beta_0$ or freeze AI assistance.")
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


# ======================================================================= the equations pane
def _equations_pane(d, p, LEVEL, sim):
    """Full-height left pane — EQUATIONS-ONLY (Pavel, round 2: the Introduction/Equations
    switch is gone; "each level can have short introduction just not to complicate it — it
    should be minimal"). A short per-level intro paragraph (content.LEVEL_INTRO, distilled
    from the retired level cards) sits above the equations. The notation expander that used to
    sit at the pane's bottom is RETIRED (D-096) — everything crucial in it was already said
    where the claim is made. The startup tour deck is a separate feature and untouched.
    `sim` rides through to the equations panel for the D-081 speed-race subsection."""
    with st.container(key="mainpane"):
        st.caption(_sub_live(LEVEL_INTRO[LEVEL], d))
        equations_panel(LEVEL, d, p, sim)
        # (the "Notation & conventions" expander is RETIRED — D-096; see the note where
        # NOTATION_SECTIONS used to live in ui/content.py for the audit and the two relocations)


def _pane_strip():
    """The thin vertical strip the equations pane folds into while the calibration panel is
    open (variant A2); clicking it closes the panel and reopens the pane."""
    with st.container(key="eqstrip"):
        st.button("Equations ▸", key="strip_btn", on_click=close_cal,
                  help="close the calibration panel and reopen the equations pane")


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
            mc_headline(mc_key, show_blowup=(LEVEL >= 2))
            need_rerun = mc_panel_fin(mc_key, visible=True, show_blowup=(LEVEL >= 2))
        else:
            # D-080 (Pavel): ONE financial graph — the coverage ratio ρ_t = E_t/B_t in
            # percent, "to bring attention to this most important output". The nominal
            # profit / revenue-vs-cost charts are gone (a nominal view returns later behind
            # an R₀-and-m toggle — future work). Break-even = the dashed 100% line; the
            # first crossing gets a year annotation.
            with np.errstate(divide="ignore", invalid="ignore"):
                cov = 100.0 * np.asarray(sim["revenue"], float) / np.asarray(sim["cost"], float)
            f = fig_base("Coverage — earnings ÷ model-building cost  (%)", "year", "%",
                         height=230)
            line(f, cal(sim["t"]), cov, "coverage  ρ = E/B", C_PROFIT)
            f.add_hline(y=100.0, line=dict(color=PAL["red"], dash="dash", width=1),
                        annotation_text="break-even (100%)", annotation_position="bottom left",
                        annotation_font_size=10)
            fin = np.isfinite(cov)
            if fin.any():
                lo_v, hi_v = float(np.min(cov[fin])), float(np.max(cov[fin]))
                f.update_yaxes(range=[min(lo_v, 90.0) - 0.04 * (hi_v - lo_v + 1),
                                      max(hi_v, 110.0) + 0.04 * (hi_v - lo_v + 1)])
            side = cov >= 100.0
            if side.any() and (~side).any():          # the path crosses break-even
                i = int(np.argmax(side != side[0]))   # first flip from the initial side
                t0, t1 = float(sim["t"][i - 1]), float(sim["t"][i])
                c0, c1 = float(cov[i - 1]), float(cov[i])
                tc = t0 + (100.0 - c0) * (t1 - t0) / (c1 - c0) if c1 != c0 else t1
                # D-105: the label stays, and rounds to a WHOLE year. A crossing quoted to a
                # tenth ("2031.4") reads as a forecast precise to the month, which nothing here
                # supports — the year is already an interpolation between two grid nodes, and
                # under FIN4 (D-104) the accounting basis alone moved this date by ~4 years.
                # It renders only when the path actually crosses inside the horizon, and it is
                # self-evidently a property of the dials the reader has set (Pavel: "if the
                # crossing happens before T, state it, it is clear that it is for the chosen
                # parameter values and model level"). D-077's "no crossing year until FIN4 is
                # settled" is discharged: FIN4 is settled.
                f.add_vline(x=float(cal(tc)), line=dict(color=NEUTRAL, dash="dot", width=1),
                            annotation_text=f"crosses 100% in {theme.YEAR0 + tc:.0f}",
                            annotation_position="top right", annotation_font_size=10)
            show(f, key="pt_coverage")
    return need_rerun


def _capability_tile(d, sim, LEVEL, mode_mc, mc_key):
    """Capability paths + the gap Δ (and, with the growth engine, the RSI-feedback share ψ that
    rides with the gap). Point paths, or the capability-gap fan in Monte-Carlo mode.
    D-068: the algo-progress and compute graphs moved out to their own level-gated tabs."""
    served = d["tau"] > 0.0                 # x^R differs from x^L only under a release delay
    show_growth = LEVEL >= 2
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
        # D-096: the unit convention is stated HERE, beside the axis that carries it, because
        # the retired notation expander was the only place that defined OOM or said where its
        # zero is — and it was cumulative, so this has to render at every level, not just L1.
        line(f, cal(sim["t"]), sim["x_L"], "leader" + (" developed" if served else "") + "  xᴸ",
             C_LEADER)
        if served:
            line(f, cal(sim["t"]), sim["x_R"], "leader served  xᴿ", C_SERVED, dash="dash")
        line(f, cal(sim["t"]), sim["x_F"], "follower  xᶠ", C_FOLLOWER)
        show(f, key="pt_cap")
        st.caption("**OOM** = orders of magnitude — factors of 10 — of *effective* compute, "
                   "physical compute times everything else (architecture, data, post-training "
                   "know-how). Measured **above the early-2026 frontier**, so 0 is today.")

        gap_ttl = ("Capability gap  Δ  &  RSI-feedback share  ψ" if show_growth
                   else "Capability gap — how far ahead the leader is  Δ = xᴸ − xᶠ")
        gap_ylab = "OOM (gap)   /   share (ψ)" if show_growth else "OOM"
        f = fig_base(gap_ttl, "year", gap_ylab, height=230)
        # anchor at 0 so a (near-)constant gap reads flat instead of auto-zooming into
        # integrator-precision noise on the y-axis
        f.update_yaxes(rangemode="tozero")
        line(f, cal(sim["t"]), sim["Delta"], "gap  Δ = xᴸ − xᶠ  (OOM)", C_GAP)
        if show_growth:
            line(f, cal(sim["t"]), sim["psi_share"], "ψ-share (fraction of algo progress from RSI)",
                 C_PSI, dash="dot")
            f.add_hline(y=0.25, line=dict(color=NEUTRAL, dash="dash", width=1),
                        annotation_text="ψ-share 25% (feedback no longer small)",
                        annotation_position="top left", annotation_font_size=10)
        show(f, key="pt_gap")


def _algo_tile(sim, mode_mc=False, mc_key=None):
    """Algorithmic-progress paths a(t) — its own tab from L2 (D-068, renumbered by D-081: moved out of Capability,
    where it crowded the gap graph). Both actors, unchanged data: catch-up flows through the
    algorithmic channel, so the follower's a can overtake the leader's while its total x trails.
    Monte-Carlo mode shows the leader/follower a(t) fans (same snapshot, own grid)."""
    with st.container(border=False):
        if mode_mc:
            mc_panel_path(mc_key, grid=ALGO_GRID)
        else:
            f = fig_base("Algorithmic progress — leader vs follower  a", "year",
                         "OOM above 2026 frontier", height=230)
            line(f, cal(sim["t"]), sim["a_L"], "leader  aᴸ", C_LEADER)
            line(f, cal(sim["t"]), sim["a_F"], "follower  aᶠ", C_FOLLOWER)
            show(f, key="pt_algo")
        st.caption("Catch-up flows through the *algorithmic* channel, so the follower's $a$ can "
                   "overtake the leader's $a$ while its total capability $x$ still trails — the "
                   "compute deficit is what keeps the gap open.")


def _compute_tile(sim, LEVEL, mode_mc=False, mc_key=None):
    """Compute paths c(t) — its own tab from L2 (D-068, renumbered by D-081). The follower's
    compute is NOT modeled before L3, so L2 plots ONLY the leader's frontier compute; L3
    (catch-up channels) adds
    the follower's compute line alongside the leader's (Pavel's explicit instruction).
    Monte-Carlo mode shows the compute fan(s) — same level gating for the follower's."""
    both = LEVEL >= 3
    with st.container(border=False):
        if mode_mc:
            mc_panel_path(mc_key, grid=COMP_GRID_BOTH if both else COMP_GRID_L)
        else:
            ttl = ("Compute — leader vs follower  c" if both
                   else "Compute — leader (frontier)  c")
            f = fig_base(ttl, "year", "OOM above 2026 frontier", height=230)
            line(f, cal(sim["t"]), sim["c_L"], "leader  cᴸ", C_LEADER)
            if both:
                line(f, cal(sim["t"]), sim["c_F"], "follower  cᶠ", C_FOLLOWER)
            show(f, key="pt_comp")
        st.caption(
            "Compute is the capital-intensive engine behind capability. "
            + ("The follower's own compute enters the model at this level, plotted alongside "
               "the leader's frontier compute." if both
               else "The follower's own compute is not modeled until the catch-up-channels "
                    "level (3); here only the leader's frontier compute is shown."))


def _value_tile(sim, mode_mc=False, mc_key=None):
    """Value flows W over the horizon — its own tab from L2 (D-068, renumbered by D-081). LOG y-axis: W(x) grows
    ~exponentially with capability, so on a log scale the value levels read off as slopes.
    x-axis is the horizon (year). D-077: W is an INDEX (W(0) = 1), so the y-axis is "× today's
    frontier value", not dollars — the dollar scale lives in the single coefficient κ. The two
    lines are the leader's value W(xᴸ) and the fringe's W(xᶠ), whose gap the leader earns on."""
    with st.container(border=False):
        if mode_mc:
            mc_panel_path(mc_key, grid=VALUE_GRID)
        else:
            f = fig_base("Value index over time — leader vs fringe  W  (× today, log)",
                         "year", "× today's frontier value  (log scale)", height=230)
            line(f, cal(sim["t"]), sim["W_R"], "leader  W(xᴸ)", C_LEADER)
            line(f, cal(sim["t"]), sim["W_F"], "follower  W(xᶠ)", C_FOLLOWER)
            f.update_yaxes(type="log")
            show(f, key="pt_value")
        st.caption("Each actor's capability commands a value $W(x)$, indexed so that today's "
                   "frontier is **1**; the leader earns on the **gap** $W(x^L_t) - W(x^F_t)$ "
                   "between the two lines, scaled so that the gap at $t = 0$ earns exactly "
                   "$\\rho$ — today's coverage. Log y-axis: near-exponential value growth reads "
                   "as straight-line slopes.")


# (the release-delay and under-the-hood sections are GONE with the retired levels —
# Pavel's D-081 ladder amendment; the release-delay machinery is x^R-parked in the
# spec (N9) and the notebook keeps delay_comparison for a future revival)


def _charts_column(d, items, sim, hl, p, LEVEL, mode_mc, mc_key, sample_keys):
    """The right chart panel: a fixed drag-collapsible column (D-050 — the divider, the » and
    the collapsed strip are CLIENT-side, injected by theme.inject_frontend_js; the server
    always renders this column, so collapsing only CSS-hides it and the MC finance component
    stays mounted with its background accumulation ticking). Content: a Finance | Model path
    switch (one tile group at a time — Pavel; charts inside a group still stack vertically),
    with warnings on top and the level-gated extras under Finance."""
    with st.container(key="chartscol"):
        # ---- the mode AND horizon switches live HERE now (Pavel, round 2: "it is more
        # natural there — it relates to the graphs"; the horizon addendum likewise — both
        # configure the graphs, not the model). Same "mode"/"w_hz" session keys as always,
        # read at the top of the run; a late-instantiating widget binding an existing key is
        # the standard pattern. On narrow/phone they ride the Graphs overlay with the rest of
        # this column — you see them exactly when you look at graphs.
        _reg("mode", "Point forecast")
        # ratios + the compact-pill CSS (theme: chartscol stButtonGroup) keep BOTH switches on
        # one row at the 330px default panel width
        _mc_c, _hz_c = st.columns([2.05, 1.45], vertical_alignment="center")
        _mc_c.segmented_control("Mode", ["Point forecast", "Monte Carlo"], key="mode",
                                label_visibility="collapsed",
                                help="**Point forecast** shows the single trajectory at the "
                                     "spot values. **Monte Carlo** shows the forecast fan "
                                     "across the sampling ranges — it keeps accumulating in "
                                     "the background either way, so switching is instant.")
        _hz_c.segmented_control("Horizon", ["5 yr", "10 yr"], key=_reg("w_hz", "10 yr"),
                                label_visibility="collapsed",
                                help="The time window every graph uses.")
        # ---- the tab switch (same segmented idiom); the shadow mem key is belt-and-braces
        # for widget GC. D-068 (renumbered by D-081): five tabs, introduced by level.
        # Financial + Capability are always present; the merged Dynamics level L2 unlocks Algo
        # progress, Compute AND Value together (its mechanisms touch all three); Compute gains
        # the follower line at L3. D-054 (round 2): the switch already names the visible tab,
        # so there are no per-tab section headers.
        tab_labels = ["Financial", "Capability"]
        if LEVEL >= 2:
            tab_labels += ["Algo progress", "Compute", "Value"]
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
        # (round 3) the "How to read the Monte-Carlo fans" expander is gone — its content
        # lives in the MC headline's help tooltip (mc_headline / _MC_HELP in ui/mc.py).
        need_rerun = False
        if fin_vis:
            need_rerun = _finance_tile(d, sim, hl, LEVEL, mode_mc, mc_key)
        elif tab == "Capability":
            _capability_tile(d, sim, LEVEL, mode_mc, mc_key)
        elif tab == "Algo progress":
            _algo_tile(sim, mode_mc, mc_key)      # point paths or a(t) fans
        elif tab == "Compute":
            _compute_tile(sim, LEVEL, mode_mc, mc_key)  # leader only < L3; +follower at L3
        elif tab == "Value":
            _value_tile(sim, mode_mc, mc_key)     # W over the horizon, log y-axis
        if not (mode_mc and fin_vis):
            # CRITICAL invariant: exactly ONE finance-component mount per run. It is visible
            # only on (MC mode ∧ Finance tab); in every other state it mounts HIDDEN so its
            # heartbeat keeps the background accumulation ticking — switching tab or mode
            # then shows a ready fan instantly.
            need_rerun = mc_panel_fin(mc_key, visible=False,
                                      show_blowup=(LEVEL >= 2)) or need_rerun
        if need_rerun:
            st.rerun()   # corner click: propagate the inspected draw to the sidebar ticks


# ======================================================================= main entry
def render_main(d, items, sim, hl, p, LEVEL):
    """Everything below the frozen top bar (D-047 layout): [calibration panel + fold strip
    (when open) | middle Equations pane (flex)] | right chart panel (fixed, collapsible).
    Column ratios are cosmetic — the theme CSS pins the real widths."""
    mode_mc = mc_active()
    open_key = cal_open()
    # ---- background Monte-Carlo accumulation runs on EVERY run (D-043) ---------------------
    mc_key, sample_keys, merge_delta, tro, pro = mc_prepare(d, LEVEL)
    mc_accumulate(items, mc_key, tuple(sample_keys), merge_delta, show_blowup=(LEVEL >= 2),
                  show_horizon=(LEVEL >= 2), target_ranges=tro, param_ranges=pro)
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
            _equations_pane(d, p, LEVEL, sim)
    with ccol:
        _charts_column(d, items, sim, hl, p, LEVEL, mode_mc, mc_key, sample_keys)
