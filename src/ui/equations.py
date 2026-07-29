"""The equations-&-calibration panel (one collapsible subsection per model block), the
base-model profitability race (Level 1) and the two-forces speed race (Level 2). D-050: no
"new/changed at this level" TEXT marks anywhere — the concise view already shows only the
level's own subsection(s), and the show-all view marks them with a subtle border/tint on the
expander header instead.

D-081 (levels 2–5 merged into one Level 2 "Dynamics"): the house rule "each level updates
exactly ONE subsection" cannot survive a 4-way merge, so `_CHANGED_AT` is now TUPLE-valued —
Level 2 changes four subsections at once AND adds one (the speed race). The concise view renders
them in STORY order (the two opposing forces — slowdown, then RSI acceleration — then their
riders ℓ and x_mid, then the speed race), opened by a one-line orientation caption naming the changed blocks
and stating that follower/revenue/profit carry over; the show-all view accent-marks all of them.
Every other level still changes exactly one subsection.

D-092 RETIRED THE `curve` SUBSECTION (Pavel: "I said that the sigmoid function should be hidden
inside the model so it should not be definition. It's just a technical detail which should not be
visible inside the widget"). D-084 had given the transition primitive its own subsection, first in
the order, carrying the closed form and the two identities; D-088 rewrote it around Γ's
observables. All of that is now GONE from the widget — no closed form, no σ, no plateau or slope
identity, no base. What replaced it is a small live GRAPH per transition variable, drawn inline
beside that variable's own equation (see `_transition_svg`), which is what Pavel asked for
instead. The mathematics survives in full in `Notes/model_draft_v2.md` I.1 and the paper's
Appendix A — verified present before deletion, with the spec strictly the more complete.

Γ itself is still NAMED in each of the three uses, with its four dialled observables. That is a
deliberate line: what Pavel called a technical detail is the SIGMOID, and Γ-with-four-observables
is the interface, not the machinery. It is also exactly the paper's own body-level treatment
(draft_v3 eq. 1 names Γ and shows a figure; the closed form is appendix-only), and the standing
rule is that the widget matches the paper.

The "race" subsection (Pavel's combination ruling, 2026-07-27): the merged level's headline —
do the two opposing forces speed the frontier up or slow it down? — spans compute AND
algorithms, so it lives in a subsection of its OWN, LAST in Level 2's story order, exactly as
the base model closes with the Profit subsection. It reads ċᴸ and ȧᴸ off the leader's already-
simulated (cached) path — the app holds zero model math (D-025) — and renders a live
which-force-wins verdict. Present from L2 up, with the standard expander / hover-highlight
machinery.
"""
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from .calibration import _cal_cards, _cal_delta_merged
from .model_access import m
from .theme import YEAR0
from .state import _gc_sym, _reg


# =========================================================== per-variable transition graphs (D-092)
# Pavel: "Keep the technicalities in the widget minimal, instead we rather use a small graph to
# illustrate the transition for each individual variable." So each USE of Γ draws its own curve,
# inline beside its own equation, in its own units — the picture answers "what is this dial doing
# to THIS variable" without any algebra.
#
# RENDERING: server-generated inline SVG, not a chart library. No JS, no new dependency, certainly
# safe in the pyodide build, and it is the idiom D-079's crop band and D-089's mini rail already
# use — the server knows the geometry, the markup draws it, no client state exists. Theme handling
# rides the app's own variables: `currentColor` for anything that should follow the text and
# `rgba(var(--accent-rgb), a)` for the curve, so light and dark are both correct with no branch.
#
# The curve is sampled from the MODEL'S OWN Γ (m.gamma_curve), never re-implemented here, so the
# illustration cannot drift from the mathematics it illustrates.
_SVG_W, _SVG_H = 300.0, 82.0          # viewBox units; the element itself scales to the column
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 6.0, 58.0, 12.0, 16.0   # right pad holds the asymptote label


def _transition_svg(y_today, y_inf, u_mid, p0, u_unit, fmt="{:.2f}", y_unit="", years=False):
    """One transition, drawn small. Returns an <svg> string to hand to st.markdown.

    Shows the four dialled observables as PICTURE rather than formula: today's value as a dot on
    the curve at u = 0, the asymptote as a dashed rule with its value, the midtime as a tick, and
    the position implicitly — how far down the curve today's dot already sits. The window runs a
    little before today to just past 2*u_mid, which is where the transition is essentially over.

    `years=True` labels the axis in CALENDAR YEARS off theme.YEAR0 (2026 at u = 0) — for the two
    transitions that run in time. The value slope runs over CAPABILITY and keeps its own units;
    calendar years there would be nonsense (D-092 follow-up, Pavel).

    EVERY marker is a point ON the sampled path, taken from the same gamma_curve call and pushed
    through the same Y() transform as the polyline. It has to be by construction, not by
    agreement: the first version derived the midtime marker independently as the average of
    today's value and the asymptote, which is NOT the curve there — the curve at u_mid is halfway
    between the two PLATEAUS, and since the -inf plateau sits above today for a slowdown, the
    marker rendered visibly BELOW the line (Pavel's screenshot).
    """
    if not (u_mid > 0) or not np.isfinite([y_today, y_inf, u_mid, p0]).all():
        return ""                                    # unreachable via the dials; fail silent
    u0, u1 = -0.18 * u_mid, 2.35 * u_mid
    us = np.linspace(u0, u1, 96)
    try:
        ys = np.asarray(m.gamma_curve(us, y_today, y_inf, u_mid, p0), dtype=float)
    except ValueError:                               # p0 outside the enforced domain
        return ""
    lo, hi = float(min(ys.min(), y_inf)), float(max(ys.max(), y_inf))
    if hi - lo < 1e-12:                              # a flat transition (amplitude zero)
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    lo, hi = lo - 0.12 * span, hi + 0.12 * span      # breathing room top and bottom

    def X(u):
        return _PAD_L + (u - u0) / (u1 - u0) * (_SVG_W - _PAD_L - _PAD_R)

    def Y(y):
        return _PAD_T + (hi - y) / (hi - lo) * (_SVG_H - _PAD_T - _PAD_B)

    pts = " ".join(f"{X(u):.1f},{Y(y):.1f}" for u, y in zip(us, ys))
    x0, xm = X(0.0), X(u_mid)
    # ON THE CURVE BY CONSTRUCTION: both markers are gamma_curve evaluated at their own u and
    # mapped through the SAME Y() the polyline uses. See the docstring for what the alternative
    # cost us. test_d092_markers_sit_on_the_curve pins this to under a pixel.
    y_now = Y(float(m.gamma_curve(0.0, y_today, y_inf, u_mid, p0)))
    y_half = Y(float(m.gamma_curve(u_mid, y_today, y_inf, u_mid, p0)))
    ink = "currentColor"
    acc = "rgba(var(--accent-rgb),0.95)"

    def _tick(u, label, op=0.55):
        return (f'<text x="{X(u):.1f}" y="{_SVG_H - 4:.1f}" font-size="9" fill="{ink}" '
                f'opacity="{op}" text-anchor="middle">{label}</text>')

    if years:
        # ROUND CALENDAR YEARS AT THEIR TRUE POSITIONS (D-092 follow-up, Pavel: "I would like
        # years like 2026 and 2028"). Every even year inside the window gets a tick where it
        # actually falls, so 2028 lands at u = 2.0 — slightly LEFT of a 2.3-yr midpoint line,
        # which is the reading he asked for and is correct rather than arranged.
        yrs = [yy for yy in range(YEAR0, YEAR0 + int(u1) + 1)
               if yy % 2 == 0 and u0 <= yy - YEAR0 <= u1]
        ticks = "".join(_tick(float(yy - YEAR0), str(yy)) for yy in yrs)
        # the midtime keeps a mark, but as the DIAL it is rather than a competing number
        ticks += (f'<text x="{xm:.1f}" y="{_PAD_T - 3:.1f}" font-size="8.5" fill="{ink}" '
                  f'opacity="0.5" text-anchor="middle">half way</text>')
    else:
        ticks = (_tick(0.0, "now") + _tick(u_mid, fmt.format(u_mid))
                 + _tick(2.0 * u_mid, u_unit, op=0.45))
    return (
        f'<svg viewBox="0 0 {_SVG_W:g} {_SVG_H:g}" width="100%" height="{_SVG_H:g}" '
        f'role="img" style="display:block;max-width:420px;overflow:visible" '
        f'aria-label="transition curve: {fmt.format(y_today)} today toward {fmt.format(y_inf)}, '
        f'half way at {fmt.format(u_mid)} {u_unit}">'
        # the asymptote, with its value on the right
        f'<line x1="{_PAD_L:.1f}" y1="{Y(y_inf):.1f}" x2="{_SVG_W - _PAD_R + 4:.1f}" '
        f'y2="{Y(y_inf):.1f}" stroke="{ink}" stroke-width="1" stroke-dasharray="3 3" '
        f'opacity="0.32"/>'
        f'<text x="{_SVG_W - _PAD_R + 8:.1f}" y="{Y(y_inf) + 3.2:.1f}" font-size="9.5" '
        f'fill="{ink}" opacity="0.62">{fmt.format(y_inf)}{y_unit}</text>'
        # today's guide and the midtime tick
        f'<line x1="{x0:.1f}" y1="{y_now:.1f}" x2="{x0:.1f}" y2="{_SVG_H - _PAD_B:.1f}" '
        f'stroke="{ink}" stroke-width="1" stroke-dasharray="2 3" opacity="0.28"/>'
        f'<line x1="{xm:.1f}" y1="{y_half:.1f}" x2="{xm:.1f}" y2="{_SVG_H - _PAD_B:.1f}" '
        f'stroke="{ink}" stroke-width="1" stroke-dasharray="2 3" opacity="0.28"/>'
        # the curve itself
        f'<polyline points="{pts}" fill="none" stroke="{acc}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        # today's value: the anchor Γ(0) = y(0)
        f'<circle cx="{x0:.1f}" cy="{y_now:.1f}" r="3" fill="{acc}"/>'
        f'<text x="{x0 + 5:.1f}" y="{y_now - 4:.1f}" font-size="9.5" fill="{ink}" '
        f'opacity="0.85">{fmt.format(y_today)}{y_unit}</text>'
        # the halfway dot
        f'<circle cx="{xm:.1f}" cy="{y_half:.1f}" r="2.2" fill="none" stroke="{acc}" '
        f'stroke-width="1.3" opacity="0.8"/>'
        # baseline ticks
        + ticks + f'</svg>')


def _draw_transition(y_today, y_inf, u_mid, p0, u_unit, caption, fmt="{:.2f}", y_unit="",
                     years=False):
    """Render one transition graph plus its one-line reading, if the transition is live."""
    svg = _transition_svg(y_today, y_inf, u_mid, p0, u_unit, fmt=fmt, y_unit=y_unit, years=years)
    if not svg:
        return
    st.markdown(svg, unsafe_allow_html=True)
    st.caption(f":gray[{caption}]")


def _eqcols():
    try:
        return st.columns([2, 1], vertical_alignment="top")
    except TypeError:
        return st.columns([2, 1])


def _profit_condition(level, dd):
    """The base-model profitability race (Level 1, where value is exponential and the gap is
    constant): revenue ×/yr vs cost ×/yr, both expressed through targets the user already
    controls, with a live verdict and the break-even value-multiplier pivot. Hidden from L2 —
    the merged Dynamics turn on saturation AND time-varying rates at once, invalidating the
    asymptotic reading (same hide-don't-annotate ruling as the horizon metrics)."""
    # every number below recomputes each rerun from the EFFECTIVE dict the simulation uses
    # (post-inversion, post-pin) — no default fallbacks.
    gc = dd["g_C0"]
    ga = dd["g_a"]
    nu_ = dd["nu"]
    gp = dd["g_p"]
    speed = gc + ga
    vm = float(10.0**nu_)                   # value multiplier per OOM (the Basics target)
    rev_ooms = nu_ * speed                  # revenue growth in value-OOMs per year
    cost_ooms = gc - gp                     # cost growth in OOMs per year
    rev_x = float(10.0**rev_ooms)           # revenue growth, ×/yr — vm^(g_c+g_a)
    cost_x = float(10.0**cost_ooms)         # cost growth, ×/yr — the bill-growth observable
    be_vm = float(cost_x ** (1.0 / speed))  # break-even value multiplier per OOM
    sym = _gc_sym()
    rel = ">" if rev_x > cost_x else "<"    # borderline equality reads as the red case
    with st.container(border=True):
        st.markdown("**Will the leader *ever* be profitable?** With the gap constant and value "
                    "exponential, one inequality decides it — a race between two growth rates "
                    "you already control:")
        # two stacked lines, not one long row: the single-line form clipped its right operand
        # at the card edge in the narrow tabbed pane (QA S4; .katex-display also scrolls now)
        st.latex(
            rf"\begin{{aligned}}"
            rf"&\small\underbrace{{\;10^{{\,\nu\,({sym}+g_a)}}\;}}"
            rf"_{{\text{{revenue}}\;\times{rev_x:.2f}\text{{/yr}}}}"
            rf"\;\;{rel}\;\;"
            rf"\underbrace{{\;10^{{\,{sym}-g_p}}\;}}"
            rf"_{{\text{{cost}}\;\times{cost_x:.2f}\text{{/yr}}}}"
            rf"\\[10pt]"
            rf"&\small\Longleftrightarrow\quad"
            rf"\nu\,({sym}+g_a)\;{rel}\;{sym} - g_p"
            rf"\end{{aligned}}"
        )
        st.markdown(
            f"Each year the leader gains **{speed:.2f} OOM** of capability, each worth "
            f"**×{vm:.2f}** in value, so earnings grow **×{rev_x:.2f}/yr** (the gap is constant, "
            f"so the earnings inherit $W$'s growth rate). The training bill grows "
            f"**×{cost_x:.2f}/yr**. In OOM terms: earnings climb "
            f"**+{rev_ooms:.2f} OOM/yr** vs the bill's **+{cost_ooms:.2f} OOM/yr**.")
        # where the race STARTS (D-080): the coverage dial IS today's intercept, so the race
        # has a handicap as well as a slope — in coverage terms, and since D-093 there are no
        # other terms it could be expressed in.
        rho = float(dd.get("rho", float("nan")))
        cov0 = 100.0 * rho
        ratio = (1.0 / rho) if rho > 0 else float("nan")   # how many times coverage must climb
        if rev_x > cost_x:
            import numpy as _np
            yrs = float(_np.log(ratio) / _np.log(rev_x / cost_x)) if ratio > 1 else 0.0
            st.markdown(f":green[**✓ ×{rev_x:.2f} > ×{cost_x:.2f} — earnings outrun cost ⇒ "
                        "coverage rises ⇒ break-even sooner or later, for good.**]")
            st.markdown(f"Coverage starts at **{cov0:.0f}%** (the $\\rho_0$ dial), so climbing "
                        f"to 100% takes about **{yrs:.1f} years**.")
        else:
            st.markdown(f":red[**✗ ×{rev_x:.2f} < ×{cost_x:.2f} — cost outruns earnings ⇒ "
                        "coverage falls ⇒ never reaches break-even at these settings.**]")
            st.markdown(f"And coverage *starts* at **{cov0:.0f}%**, so it only falls away. "
                        ":gray[This is the calibrated base model's honest verdict, not a "
                        "misconfiguration — the base holds compute growth constant for ever. "
                        "What can turn it around is the **compute slowdown** (Level 2): when "
                        "scaling stops, the bill stops growing while capability — and earnings — "
                        "keep compounding on algorithmic progress.]")
        st.caption(f"Break-even pivot: a value multiplier of **×{be_vm:.2f}/OOM** (now "
                   f"×{vm:.2f}) balances the race — slide *Value per OOM* past it "
                   "and the verdict flips.")
        # (the old "computed at today's rates" rider for levels 3–4 is gone: this card now
        # renders at Level 1 only, where every rate is constant by construction — D-081)
        st.caption("One verdict for all time, because every rate here is constant. At Level 2 "
                   "the rates start moving and the value curve bends, so the question becomes "
                   "*when*, not just *whether*.")


def _speed_race(dd, sim, level):
    """The Level-2 closing subsection body (Pavel's combination ruling — grafted from the
    rival D-081 variant): the two opposing forces the merged Dynamics level switches on, read
    off the leader's OWN simulated path. Compute growth decays toward its floor (force 1, the
    brake) while the ψ feedback compounds algorithmic progress (force 2, the accelerator);
    which one dominates decides whether the frontier decelerates or takes off, and that is a
    CALIBRATION question — hence a live card, mirroring the Level-1 profitability race inside
    Profit.

    ċᴸ and ȧᴸ are differentiated from the model's own integrated paths (no math is re-derived
    here — the app holds zero model math, D-025); np.gradient is one-sided at the ends, which
    is what "today" and "at the horizon" mean."""
    t = np.asarray(sim["t"], float)
    if t.size < 3:
        return
    dc = np.gradient(np.asarray(sim["c_L"], float), t)
    da = np.gradient(np.asarray(sim["a_L"], float), t)
    c0, cT, a0, aT = float(dc[0]), float(dc[-1]), float(da[0]), float(da[-1])
    if not np.isfinite([c0, cT, a0, aT]).all():
        return              # ψ blow-up: the charts column's BLOW-UP warning carries that case
    s0, sT = c0 + a0, cT + aT
    T = float(t[-1])
    psi_sh = float(np.asarray(sim["psi_share"], float)[-1])
    accel = sT > s0
    # same threshold as views._warnings: past +25 OOM the ψ loop has gone super-exponential and
    # "speed at T" is a meaningless number (it prints as 1e33 OOM/yr), so say that instead.
    blown = float(np.nanmax(np.asarray(sim["x_L"], float))) > 25.0
    with st.container(border=True):
        st.markdown("**Does the frontier speed up, or slow down?** The two forces pull in "
                    "opposite directions — the answer is a matter of calibration, not of "
                    "modelling:")
        if blown:
            st.markdown(":red[**↗ Run-away: the $\\psi$ feedback goes super-exponential inside "
                        "the horizon (spec N4), so the speed *at* $T$ has no meaning.**] Lower "
                        "$\\gamma$ or $\\beta_0$ — or freeze AI assistance — to read the race.")
            return
        st.latex(
            rf"\begin{{aligned}}"
            rf"&\small\underbrace{{\;\dot c^L\;}}"
            rf"_{{\text{{compute: }}{c0:.2f}\,\rightarrow\,{cT:.2f}}}"
            rf"\;\;+\;\;"
            rf"\underbrace{{\;\dot a^L\;}}"
            rf"_{{\text{{algorithms: }}{a0:.2f}\,\rightarrow\,{aT:.2f}}}"
            rf"\;\;=\;\;"
            rf"\underbrace{{\;\dot x^L\;}}"
            rf"_{{\text{{frontier speed: }}{s0:.2f}\,\rightarrow\,{sT:.2f}\ \text{{OOM/yr}}}}"
            rf"\end{{aligned}}")
        # the algo line NETS the ψ push against the experiment-compute term, which the slowdown
        # drags down — so its verb follows the realised sign, never a hard-coded "climbs"
        a_dir = "rises" if aT > a0 else ("falls" if aT < a0 else "holds")
        st.markdown(
            f"**Force 1, the brake —** compute growth decays from **{c0:.2f}** toward its floor "
            f"$g_{{c\\infty}} =$ {dd['g_C_inf']:.2f} OOM/yr, reaching **{cT:.2f}** at "
            f"$T = ${T:.0f} yr.  \n"
            f"**Force 2, the accelerator —** the $\\psi$ feedback (AI speeding up its own R&D) "
            f"supplies **{psi_sh * 100:.0f}%** of algorithmic progress by $T$ — pulling against "
            f"the experiment-compute term, which the slowdown drags down with it. Net: "
            f"$\\dot a^L$ {a_dir} from **{a0:.2f}** to **{aT:.2f}** OOM/yr.")
        if accel:
            st.markdown(f":green[**↗ The feedback wins — the frontier accelerates: "
                        f"{s0:.2f} → {sT:.2f} OOM/yr.**]")
        else:
            st.markdown(f":orange[**↘ The slowdown wins — the frontier decelerates: "
                        f"{s0:.2f} → {sT:.2f} OOM/yr.**]")
        st.caption("Turn the knobs and this flips: the compute floor $g_{c\\infty}$ and its "
                   "midpoint $t_{mid}$ push the brake, the RSI compounding $\\gamma$ and "
                   "today's speedup $\\beta_0$ push the accelerator."
                   + (" The follower's catch-up rate $\\delta$ is still fixed by the lag, so a "
                      "decelerating leader means a **closing** gap (Level 3 gives the follower "
                      "its own engine and makes the gap stationary again)." if level == 2
                      else ""))


# Per-subsection equation panel. Labels stay STABLE across level switches, so every subsection
# keeps the user's manual open/closed state (stable label -> stable widget identity).
# "race" (the D-081 speed race) exists only from L2; it sits after Profit — the model's two
# derived closers, in the order the ladder introduces them. (The old "extensions" subsection
# is gone with the retired extensions level — Pavel's ladder amendment; its equations are
# parked in the spec.)
# D-084 (Pavel): "you should define the universal s-curve in a separate subsection before the
# leader compute section." So the primitive gets its OWN block, FIRST in the order — every use
# below merely instantiates it. It is PARAMETER-FREE (pure notation): each use's midpoint and
# position stay registered to the use's own subsection, so the D-078 param→subsection reverse
# map, the hover highlight and click-to-reveal stay unambiguous.
# D-093 (Pavel's addendum: "the current Profit section you can focus on the coverage and just
# note that it is closely related to profit"). The id moved with the subject — `profit` →
# `coverage` — because it keys the D-078 param→subsection reverse map and the sidebar's
# click-to-reveal, and an id that names the wrong object is exactly how those drift apart.
_SUB_ORDER = ["leader_compute", "leader_algo", "follower", "value", "revenue", "cost",
              "coverage", "race"]
_SUB_LABEL = {"leader_compute": "Leader compute  ċᴸ",
              "leader_algo": "Leader algorithmic progress  ȧᴸ",
              "follower": "Follower", "value": "Value  W(x)",
              # "& release" retired with the mechanism (D-077 parked x^R in spec N9) — the
              # block's own equation has read E_t = κ[W(x^L) − W(x^F)] since then, with no
              # served-model line. (Renaming Revenue → Earnings outright is the equation
              # review's separate S-item, parked for the presentation pass.)
              "revenue": "Revenue", "cost": "Cost", "coverage": "Coverage",
              "race": "Frontier speed race  ẋᴸ"}
# level -> the subsections changed / new at that level, TUPLE-valued since D-081 (L1: nothing).
# The merged Level 2 changes FOUR at once and ADDS one, listed in STORY order — the concise
# view renders them in exactly this order (not _SUB_ORDER): force 1 the compute slowdown
# (leader_compute), force 2 the ψ/RSI acceleration (leader_algo), then the riders — ℓ, which
# the slowdown makes bite (cost), and the saturation bend x_mid (value) — and LAST the speed
# race that reads off which force wins (Pavel: "like the basic model has [a] subsection about
# profit"). Cost is tagged at BOTH 2 (ℓ unpins) and 5 (the R&D overhead φ_RD).
_CHANGED_AT = {2: ("leader_compute", "leader_algo", "cost", "value", "race"),
               3: ("follower",)}


def _existing_subsections(level):
    """The subsections that exist at all at this level: only the speed race joins from L2.
    (Until D-092 the Γ definition was gated here too; it is retired — the per-variable graphs
    carry what it used to say, and they live in the blocks that own the dials.)"""
    return [s for s in _SUB_ORDER if not (s == "race" and level < 2)]


# ---- D-048: the ONE authoritative subsection → parameters map ------------------------------
# (param_key, pinned) pairs whose calibration cards ride in each subsection at each level.
# The render code below consumes it AND the sidebar's equation-driven filter reads it — keep
# it as explicit data here, never as scattered conditionals.
def subsection_param_entries(sub_id, level):
    if sub_id == "leader_compute":
        cards = [("g_C0", False)]
        if level >= 2:
            # D-082: ξ retired. D-084: p0_c — the curve's position today — is the dial that
            # resolves its slope, so it rides here, right after the midpoint it is stated against
            cards += [("g_C_inf", False), ("t_mid", False), ("p0_c", False)]
        return cards
    if sub_id == "leader_algo":
        cards = [("g_a", False)]   # D-076: g_a's LEVEL is a base-model dial (the residual)
        if level >= 2:
            # equation order (Pavel's addendum): α is the first symbol inside the bracket,
            # (1−α)(ψ/ψ(0))^η, so it leads; η follows in the CES exponent; the ψ DEFINITION
            # introduces β₀ and γ after that. η is a real dial from L2 (Pavel: "I don't want
            # eta = 1 to be assumed") and since D-098 so is α.
            #
            # D-098 FOLLOW-UP — this registration is load-bearing twice over, and its absence
            # was the defect: this table drives `sidebar_filter_keys`, so an unregistered α made
            # `_vis("alpha")` False at Level 2 and the row was invisible in the default view
            # (reachable only via "show all parameters"); and it is the D-078 param→subsection
            # reverse map, so the row also had no click-to-reveal or hover highlight. The row was
            # built correctly and simply could not be seen — Pavel: "Regarding alpha, I only see
            # beta_0". Note the key here is the PARAMETER (α), not the observable the sidebar
            # dials (loss_half_gC): every target row is gated by its parameter key, exactly as
            # t_compute_x is gated by g_C0.
            cards += [("alpha", False), ("eta", False), ("beta0", False), ("gamma", False)]
        return cards
    if sub_id == "follower":
        if level <= 2:
            return [("delta_total", False)]
        return [("delta_dev", False), ("delta_rel", False), ("Delta0", False),
                ("split", False), ("g_a_F", False), ("g_CF0", False), ("g_CF_inf", False),
                ("t_mid_F", False), ("p0_F", False)]   # D-084: the fringe curve's own position
    if sub_id == "value":
        cards = [("nu", False)]
        if level >= 2:
            # D-083: the slope-transition pair; D-084: its own position dial after the midpoint
            cards += [("nu_inf", False), ("x_mid", False), ("p0_w", False)]
        return cards            # W is a dimensionless index here, and since D-093 there is
                                # no dollar coefficient anywhere for it to hand off to
    if sub_id == "revenue":
        # NO CARDS (D-093, Pavel: "the only parameter user should see is ρ, nothing else
        # matters in these two sections"). κ, R₀ and m used to ride here as pinned cards —
        # symbols "whose value does not matter for the result". Normalising the earnings leg
        # by its own t = 0 value removed them from the model rather than from the display, so
        # there is nothing left to pin. ρ itself is dialled one subsection down, where it is
        # the thing the reader is looking at.
        return []
    if sub_id == "cost":
        # k and B₀ left with the dollars (D-093): the bill's LEVEL is the normaliser now, so
        # this block has no constant of its own at any level.
        cards = []
        if level >= 2:
            cards.append(("ell", False))
        cards.append(("g_p", True))        # measured hardware leg; bill growth is a read-out
        return cards
    if sub_id == "coverage":
        return [("cov0", False)]           # D-080: the ONE money dial rides with the outcome
    return []   # race, extensions: no parameter cards


# D-078: the REVERSE of subsection_param_entries — which subsection carries a parameter at a
# level (a param appears in at most one subsection per level). Empty — every dial the map
# needs is reachable through subsection_param_entries; kept as the hook for future unmapped
# dials. (It was already empty when D-080 put R₀/m on pinned revenue cards, and D-093 deleted
# those cards along with the parameters.)
_EXTRA_PARAM_SUB = {}


def param_subsection(pkey, level):
    """The subsection id whose equations involve `pkey` at this level, or None (unmapped dial)."""
    for sub in _SUB_ORDER:
        if any(k == pkey for k, _ in subsection_param_entries(sub, level)):
            return sub
    return _EXTRA_PARAM_SUB.get(pkey)


def eq_show_all(level):
    """Effective show-all state: at L1 everything is new, so ALL subsections show and the
    toggle is hidden (D-048); above L1 the 'show all equations' checkbox decides."""
    return True if level == 1 else bool(st.session_state.get("w_eq_all", False))


def visible_subsections(level):
    """The subsections the Equations tab currently renders. Concise → the changed-at-level
    ones in the STORY order `_CHANGED_AT` lists them (this matters at the merged Level 2,
    where five subsections change/appear at once), PLUS the `_eq_focus` subsection while a
    clicked parameter row holds one (D-078: the reveal must work without permanently flipping
    the show-all toggle; the focused subsection appends after the story)."""
    changed = _CHANGED_AT.get(level)
    existing = _existing_subsections(level)
    if not eq_show_all(level) and changed is not None:
        out = [s for s in changed if s in existing]
        focus = st.session_state.get("_eq_focus")
        if focus in existing and focus not in out:
            out.append(focus)
        return out
    return existing


def sidebar_filter_keys(level):
    """The parameter keys the sidebar shows BY DEFAULT (D-065): only the parameters NEW at this
    level. D-081 sharpened the derivation for the merged Level 2: a changed subsection carries
    its whole card list (base context like g_C0 / the pinned money chips included), so we DIFF
    each changed subsection's (key, pinned) entries against the SAME subsection one level
    below — a key is "new" when it appears, or unpins, at this level. At the merged L2 that
    yields exactly the Dynamics dials {ℓ, γ, β₀, g_c∞, t_mid, x_mid}, not the whole
    five-subsection union. Independent of the middle-pane tab AND of 'show all equations'
    (that toggle governs the equations pane only); the 'show all parameters' sidebar toggle is
    the sole widener. None = show ALL params up to this level (level 1: everything is new)."""
    changed = _CHANGED_AT.get(level)
    if changed is None:            # level 1 — every parameter is new
        return None
    keys = set()
    for sub_id in changed:
        below = dict(subsection_param_entries(sub_id, level - 1))
        keys |= {k for k, pinned in subsection_param_entries(sub_id, level)
                 if below.get(k, True) != pinned or k not in below}
    return keys


def equations_panel(level, dd, p, sim=None):
    """Equations + merged calibration, one collapsible subsection at a time. Left [2/3] = glosses +
    st.latex (with new/changed tags); right [1/3] = calibration cards for that subsection's params
    (dial/range, or pinned-shown, at ≤ level). View modes control which subsections show / open.
    `sim` is the already-computed (cached) trajectory the charts use — the D-081 speed-race
    subsection reads the leader's realised path off it; None just drops that card."""
    st.markdown("**Equations & calibration at this level**")
    st.caption("Grades: **A** solid data anchor · **B** reasonable anchor · **C** judgment / weakly "
               "identified · **F** free choice or decision variable. Each right-hand card shows "
               "value, MC range and calibration target; **details** has the full note and any "
               "alternatives. Defaults are provisional → calibration session.")
    if level == 1:
        st.caption("The Level-1 controls are all **observables**: compute scaling (⇒ $g_c$), "
                   "effective-compute growth (⇒ $g_a$ as its residual), value per OOM (⇒ $\\nu$), "
                   "coverage today (⇒ $\\rho$, the finance side's one parameter) and the "
                   "fringe lag (⇒ $\\Delta_0$, $\\delta$). Shown but not dialled: $g_p$, the "
                   "measured hardware leg. Capability $x = a + c$ is in **OOM** — orders of "
                   "magnitude, factors of 10 — of *effective* compute, measured above the "
                   "early-2026 frontier, so $x = 0$ is today.")
    elif level == 2:
        # D-081 orientation line (Pavel's combination ruling: grafted from the rival variant,
        # extended for the race): the merge breaks the one-subsection-per-level rule, so the
        # pane says up front WHICH blocks moved — and that the rest carries over unchanged.
        st.caption("**Level 2 changes four blocks and adds one:** "
                   "leader compute (it *slows* — "
                   "force 1), leader algorithmic progress (AI now speeds up its own R&D — "
                   "force 2), cost (the lead time $\\ell$ starts to bite once the compute curve "
                   "bends) and value (it *bends* at $x_{mid}$) — plus a new **frontier speed "
                   "race** block at the end that reads off which force wins. Follower, revenue "
                   "and profit carry over from Level 1 unchanged. Each rate that now *moves* "
                   "carries a small graph of its own path, beside its own equation.")
    # CONCISE by default (Pavel's ruling): only the subsection(s) changed at this level show;
    # one checkbox expands to the full model. At L1 EVERYTHING is new, so all subsections
    # show and the checkbox is hidden (D-048). The old display-mode tabs' stale session key
    # is dropped so nothing rebinds to it.
    st.session_state.pop("eq_view", None)
    if level >= 2:
        st.checkbox("show all equations", key=_reg("w_eq_all", False),
                    help="Unticked: only the subsections that are **new or changed at this "
                         "level** are shown (and the left panel narrows to their parameters). "
                         "Ticked: every subsection of the model so far, expanded.")

    def eq(gloss, latex):
        st.caption(gloss)
        st.latex(latex)

    def render(sub_id):
        left, right = _eqcols()
        if sub_id == "leader_compute":
            with left:
                if level < 2:
                    # constant growth: plain g_c — the g_c/g_c∞ split is part of what L2 teaches
                    eq("Leader compute grows at the constant rate $g_c$.", r"\dot c^L_t = g_c")
                else:
                    # D-092: there is no Γ-definition block above any more (Pavel: the sigmoid
                    # "should not be visible inside the widget"), so THIS is where a reader first
                    # meets a transition. The gloss therefore has to carry the whole reading in
                    # words — from today's value, toward the floor, half way at the midtime, this
                    # far along already — and the graph below it does the rest. The other two
                    # uses are written to stand alone the same way.
                    # D-086 P1-2: the DIAL is today's growth g_c, and the curve's upper plateau
                    # is the DERIVED g_c^pre — the rate before the slowdown started. Stating it
                    # the other way round (dial = plateau) is what made three "today" anchors
                    # drift by up to 28% across the p₀ᶜ envelope.
                    eq("**Force 1 — compute growth slows down.** A rate that used to be "
                       "constant now *slides*: you dial **today's** rate $g_c$ — the same "
                       "$g_c$ as at Level 1 — the "
                       "long-run floor $g_{c\\infty}$ it heads for, the midtime $t_{mid}$, and "
                       "how much of the slowdown $p^c_0$ is already behind us. Where it "
                       "*started* ($g_c^{pre}$, above today's rate — we have come part of the "
                       "way down already) is derived, so saying the slowdown is further along "
                       "raises where it started, never what it is now.",
                       r"\dot c^L_t = g_{c,t} = \Gamma(t;\, g_c,\, g_{c\infty},\,"
                       r" t_{mid},\, p^c_0)")
                    _draw_transition(
                        dd["g_C0"], dd["g_C_inf"], dd["t_mid"], dd["p0_c"], "years",
                        f"Compute growth: ×{10.0 ** dd['g_C0']:.2f}/yr today, easing toward "
                        f"×{10.0 ** dd['g_C_inf']:.2f}/yr, half way at {dd['t_mid']:.1f} yr "
                        f"({YEAR0 + dd['t_mid']:.0f}).", years=True)
                    st.caption(":gray[That is what keeps $g_c$, effective growth "
                               "$\\dot x^L_0$ and the fringe lag equal to their dialled values "
                               "at every $p^c_0$ and every floor.]")
            _cal_cards(right, subsection_param_entries("leader_compute", level), dd, p)
        elif sub_id == "leader_algo":
            with left:
                if level < 2:
                    eq("Algorithmic progress at the constant rate $g_a$.", r"\dot a^L_t = g_a")
                else:
                    # (D-081: the old L3-only collapsed form — experiment-compute ratio ≡ 1
                    # under constant compute growth — is unreachable now that the engine and
                    # the slowdown switch on together at L2; the general ratio form IS the
                    # story: the slowdown drags the experiment term while ψ pushes. Pavel's η
                    # addendum: the CES exponent is DISPLAYED and dialled from this level —
                    # never a silent η = 1 substitution.)
                    eq("**Force 2 — algorithmic progress speeds up.** The constant $g_a$ of "
                       "Level 1 now responds to research inputs: a CES mix (curvature $\\eta$; "
                       "$\\eta = 1$, the default, is a plain weighted average) of the "
                       "AI-assistance feedback $\\psi$ and experiment compute.",
                       r"\dot a^L_t = g_a\left[(1-\alpha)\Big(\tfrac{\psi(x^L_t)}{\psi(0)}\Big)^{\eta} "
                       r"+ \alpha\Big(\tfrac{g_{c,t}}{g_c}\Big)^{\eta}\right]^{1/\eta},"
                       r"\quad \psi(x)=1+\beta_0\,10^{\gamma x}")
                    eq("ψ-share: the fraction of algo progress from the $\\psi$ feedback (past ~25% "
                       "it is no longer a small correction).",
                       r"\psi\text{-share} = 1 - \dot a^L_t\big|_{\psi\ \text{frozen}} \big/ \dot a^L_t")
            _cal_cards(right, subsection_param_entries("leader_algo", level), dd, p)
        elif sub_id == "follower":
            with left:
                if level <= 2:
                    eq("The follower has no engine of its own — pure catch-up at the single rate "
                       "$\\delta$, where $\\Delta_t = x^L_t - x^F_t$ is the **capability gap** (the "
                       "follower's own compute and algorithmic progress arrive at Level 3).",
                       r"\dot x^F_t = \delta\,(x^L_t - x^F_t) = \delta\,\Delta_t")
                else:
                    # (the served model is ALWAYS the developed frontier x^L — release delay
                    # is x^R-parked in the spec, N9/D-077)
                    # Pavel's display ruling (2026-07-28, D-084): "don't write down the max(0,..)
                    # … just assume it is positive in the equation" — the clipping is a code
                    # detail (imitation dies at parity, the follower never un-learns), carried as
                    # a note under the equation instead of inside it. Spec I.2 reads the same way.
                    eq("The follower gets its own algo rate $g_a^F$, and $\\delta$ unpacks into two "
                       "channels: ambient diffusion $\\delta_{dev}$ on the algo gap, plus "
                       "distillation $\\delta_{rel}$ from the served model; the **capability gap** "
                       "is $\\Delta_t = x^L_t - x^F_t$. The leader serves its newest model.",
                       r"\dot a^F_t = g_a^F + \delta_{dev}(a^L_t - a^F_t) + \delta_{rel}(x^L_t - x^F_t)")
                    st.caption(":gray[Both gaps are assumed positive here — the leader is ahead. "
                               "The code floors each at zero, so imitation dies at parity and the "
                               "follower never un-learns.]")
                    eq("Follower compute growth — the same universal curve as the leader's "
                       "(introduced here), with the follower's own dials throughout: its "
                       "growth **today** $g_c^F$, its own floor, its own midtime and its own "
                       "position $p^F_0$.",
                       r"\dot c^F_t = g^F_{c,t} = \Gamma(t;\, g_c^F,\, g_{c\infty}^F,\,"
                       r" t_{mid}^F,\, p^F_0)")
                    _draw_transition(
                        dd["g_CF0"], dd["g_CF_inf"], dd["t_mid_F"], dd["p0_F"], "years",
                        f"The fringe's own slowdown: ×{10.0 ** dd['g_CF0']:.2f}/yr today "
                        f"toward ×{10.0 ** dd['g_CF_inf']:.2f}/yr, half way at "
                        f"{dd['t_mid_F']:.1f} yr ({YEAR0 + dd['t_mid_F']:.0f}) — its own timing, "
                        f"not the leader's.", years=True)
                    eq("The initial gap $\\Delta_0$ splits between algo and compute.",
                       r"a^F_0 = -\text{split}\cdot\Delta_0, \qquad c^F_0 = -(1-\text{split})\,\Delta_0")
            if level <= 2:
                _cal_delta_merged(right, dd, p)
            else:
                _cal_cards(right, subsection_param_entries("follower", level), dd, p)
        elif sub_id == "value":
            with left:
                if level < 2:
                    eq("Value of capability **as an index** — normalised so $W(0) = 1$ at today's "
                       "frontier; pure exponential here: each OOM multiplies value by $10^{\\nu}$. "
                       "(The slope transition $\\nu \\to \\nu_\\infty$ arrives at Level 2.)",
                       r"W(x) = 10^{\,\nu x}")
                else:
                    # D-083 (Pavel: "log(W_t) grows exponentially with initial growth g_W and
                    # asymptotic growth g_W,infty — simply define w_t = log(W_t)"): value in
                    # LOGS, its SLOPE riding the universal curve (its third use). The hard
                    # ceiling W̄ is retired.
                    # D-084, Pavel's REVERSAL of the earlier "eliminate W" instruction: "I
                    # prefer to keep W in the basis model and define W(x)=10^w(x) where
                    # w'(x)=... in the dynamic model." So W stays the displayed dollar-side
                    # object at EVERY level, and THIS is where the reader meets w — as its
                    # logarithm, not as its replacement.
                    eq("Value in **logs** (new here): write $w = \\log_{10} W$, so "
                       "$W(x) = 10^{\\,w(x)}$ — the same $W$ as at Level 1, one level of "
                       "structure deeper. What the Dynamics add is that its SLOPE eases: "
                       "from $\\nu$ **today** down to the floor $\\nu_\\infty$, half-done at "
                       "$x_{mid}$, with $p^w_0$ of the easing already behind us. The anchor "
                       "$W(0) = 1$, i.e. $w(0) = 0$, is exact. No hard ceiling: value keeps "
                       "growing at the floor slope, so each further OOM is worth at least "
                       "$10^{\\nu_\\infty}$×.",
                       r"W(x) = 10^{\,w(x)}, \qquad"
                       r" w'(x) = \Gamma(x;\, \nu,\, \nu_\infty,\, x_{mid},\, p^w_0)")
                    _draw_transition(
                        # NO calendar years here (D-092 follow-up): this transition runs over
                        # CAPABILITY, so a year label would be meaningless. Its own units stay,
                        # and the caption says so out loud.
                        dd["nu"], dd["nu_inf"], dd["x_mid"], dd["p0_w"], "OOM of capability",
                        f"What one more OOM is worth: ×{10.0 ** dd['nu']:.2f} at today's "
                        f"frontier, easing toward ×{10.0 ** dd['nu_inf']:.2f}, half way "
                        f"{dd['x_mid']:.0f} OOM out. The axis here is CAPABILITY, not time — "
                        f"how far the frontier has moved, not when.")
                    st.caption(":gray[D-088: $\\nu$ is literally today's slope, $w'(0)$ — "
                               "dial 2.1× per OOM and you get 2.1× at today's frontier, at "
                               "every $p^w_0$. It used to be the slope *before* the easing, "
                               "which made the dial read about 1% low.]")
            _cal_cards(right, subsection_param_entries("value", level), dd, p)
        elif sub_id == "revenue":
            with left:
                # (the release-delay x^R equation is retired from the widget with its level —
                # x^R-parked in the spec, N9)
                served = "x^L"   # D-077: the model runs on the DEVELOPED frontier
                eq("Earnings ride the value gap between the leader's model and the follower, "
                   "**divided by that same gap today**. The normalisation is what makes this "
                   "block parameter-free: whatever the leader earns now is the unit, so the "
                   "line says how earnings grow, and $\\rho$ below says where they start.",
                   rf"E_t = \rho\;"
                   rf"\frac{{W({served}_t) - W(x^F_t)}}{{W({served}_0) - W(x^F_0)}}")
                st.caption(":gray[Measured in multiples of today's model-building bill, not "
                           "dollars (D-093). Only the ratio of earnings to cost was ever "
                           "identified, so the widget no longer displays a scale that the "
                           "result does not depend on.]")
            right.caption("(no parameter — the level is normalised away; the one finance "
                          "dial, $\\rho$, lives in **Coverage** below)")
        elif sub_id == "cost":
            with left:
                # (the φ_RD markup branch is retired with the cost-mechanism level — φ_RD is
                # provably inert under the observed-bill anchor; parked in the spec)
                if level >= 2:
                    # Under the ratified observed-bill anchor moving ℓ leaves the t = 0 bill
                    # where it is, so ℓ tilts the path rather than lifting the level. D-090
                    # referenced the path to c^L_ℓ so it stopped re-anchoring anything; D-093
                    # then normalised the constant away, so B₀ = 1 and the de-lagging happens
                    # entirely inside the cost function. (Extensions-sync round, audit X-24: the old EL1a/b/c TODO here is
                    # resolved — the counterfactual-ℓ view designed in brief 06b stays a PARKED
                    # design option, recorded in the decision log, not a pending code task.)
                    eq("Training in advance (new here): the firm pays now for the compute of the "
                       "model shipping $\\ell$ ahead, at prices falling at $g_p$. The bill is "
                       "measured against *today's*, so it starts at 1 whatever $\\ell$ is — what "
                       "$\\ell$ changes is the **tilt** of the path. Under Level 1's steady growth "
                       "even that cancels exactly; the slowdown arriving at this same level is "
                       "what makes it bite.",
                       r"B_t = 10^{\,c^L_{t+\ell} - c^L_\ell - g_p t}")
                    if level == 2:
                        # kept from the old Level-2 card (via the rival variant): the concrete
                        # anchor for why ℓ matters
                        st.caption("Funding the *next*, bigger model while the current one only "
                                   "breaks even is what can drag today's coverage down — the "
                                   "reported Anthropic-vs-OpenAI contrast (profit per model "
                                   "vs loss while scaling), depending on calibration.")
                else:
                    # D-093: no dollar figure survives here. The old line quoted k·R₀ = $75B,
                    # which was a calibration input, not a model quantity — and the widget's
                    # result never depended on it. The displayed form drops the reference point
                    # c^L_0 because it is EXACTLY 0 (the x^L_0 = 0 normalisation, bitwise at
                    # every configuration), which is what lets the base case read as one clean
                    # exponent — Pavel's own wording, and the form the paper ships (d4d9019).
                    # The ℓ > 0 branch above cannot do this: c^L_ℓ is not zero.
                    eq("The training bill: this period's model-building outlay — compute *and* "
                       "R&D overhead together — carried forward along the compute path at "
                       "prices falling at $g_p$, so it grows at "
                       f"×{10.0 ** (dd['g_C0'] - dd['g_p']):.2f}/yr. It is measured **in "
                       "multiples of today's bill**, so it starts at 1 by construction.",
                       r"B_t = 10^{\,c^L_t - g_p t}")
            # (no ℓ chip at L1 — Pavel, 2026-07-27: ℓ has not been introduced yet, so the base
            # level must not mention it at all; the cost line above already reads plain c^L_t)
            _cal_cards(right, subsection_param_entries("cost", level), dd, p)
        elif sub_id == "coverage":
            with left:
                # D-093 (Pavel): coverage IS the subject of this block now — it leads, and the
                # profit flow drops to the one-line note below it. The reported outcome is the
                # only identified object on the finance side, which is why it is also the only
                # financial graph (D-080).
                eq("The reported outcome: **coverage** — earnings over model-building cost. "
                   "It starts at $\\rho$, the one finance dial, and **break-even is 100%**. "
                   "Both legs are in multiples of today's bill, so the ratio is the same number "
                   "it would be in dollars — and unlike the dollars, it is identified.",
                   r"\rho_t = \frac{E_t}{B_t}, \qquad \rho_0 = \rho")
                st.caption("Closely related to **profit**: the undiscounted yearly flow is "
                           "$\\Pi_t = E_t - B_t$ (dimensionless, in the same units), so "
                           "$\\Pi_t > 0$ exactly when $\\rho_t > 1$ — the same verdict, and "
                           "the same crossing year, read off the graph above the 100% line. "
                           "Flows are **undiscounted** — this is not an NPV.")
                if level == 1:   # from L2 the merged Dynamics invalidate the asymptotic race
                    _profit_condition(level, dd)
            _cal_cards(right, subsection_param_entries("coverage", level), dd, p)
        elif sub_id == "race":
            # D-081 (Pavel's combination ruling): the Dynamics headline as the level's LAST
            # subsection — like the base model closes with Coverage. No free parameter of its
            # own; everything is read off the leader's realised path.
            with left:
                eq("The merged level's bottom line: the frontier's realised speed, today → "
                   "horizon, decomposed into the two forces.",
                   r"\dot x^L_t = \dot c^L_t + \dot a^L_t")
                if sim is not None:
                    _speed_race(dd, sim, level)
                else:
                    st.caption("(run the model to read the race — no trajectory available)")
            right.caption("(no free parameter — read off the simulated path; the dials live "
                          "in the four blocks above)")
    changed_here = _CHANGED_AT.get(level) or ()
    existing = _existing_subsections(level)
    shown = visible_subsections(level)
    if len(shown) < len(existing):
        st.caption(f"**{len(existing) - len(shown)} unchanged subsections hidden** — they "
                   "carry over from the levels below (tick **show all equations** for the "
                   "full model).")
    # D-048/D-050: the subsection(s) this level changes are marked VISUALLY (accent left border
    # + subtle header tint, both themes) — never with label text. D-081 (guard adopted from the
    # rival variant): a level may change several, so the mark applies per changed subsection and
    # only when some UNCHANGED subsection is also on screen (len(shown) > len(marked)) — the
    # concise view shows exactly the changed set and carries no marking at all; a one-line
    # neutral legend replaces the (now ambiguous) unmarked-by-elimination reading.
    marked = [s for s in changed_here if s in shown]
    if marked and len(shown) > len(marked):
        _sel = ", ".join(f".st-key-eqsub_{c} [data-testid='stExpander'] summary"
                         for c in marked)
        st.markdown(
            f"<style>{_sel} "
            "{ border-left: 3px solid #4c8dff; background: rgba(76,141,255,0.10); }"
            "</style>", unsafe_allow_html=True)
        if len(marked) > 1:
            st.caption("The accent-bordered blocks are the ones this level changes.")
    from .calpanel import param_row_key
    # D-078: a freshly clicked parameter row focuses its subsection. Force-open trick: expander
    # open/closed state rides the widget's identity (derived from its label among other params),
    # so suffixing the focused label with `bump % 2` zero-width spaces REMOUNTS it (default:
    # expanded) exactly once per click — a user's manual collapse afterwards sticks, and every
    # other subsection keeps its state (labels otherwise stay stable, see _SUB_LABEL).
    focus = st.session_state.get("_eq_focus")
    bump = int(st.session_state.get("_eq_focus_bump", 0))
    # D-078 follow-up (Pavel): while a clicked row HOLDS the focus, its subsection is visibly
    # highlighted — the same accent language as the changed-at-level mark above, so it reads
    # as one system. Persistent (re-injected per run, surviving the background-MC churn); it
    # clears when the row collapses or another row takes the focus (toggle_param_row resets
    # _eq_focus either way).
    if focus in shown:
        st.markdown(
            f"<style>.st-key-eqsub_{focus} [data-testid='stExpander'] summary "
            "{ border-left: 3px solid #4c8dff; background: rgba(76,141,255,0.10); }"
            f".st-key-eqsub_{focus} [data-testid='stExpander'] "
            "{ border-left: 3px solid rgba(76,141,255,0.55); }"
            "</style>", unsafe_allow_html=True)
    for sub_id in shown:
        label = _SUB_LABEL[sub_id] + ("\u200b" * (bump % 2) if sub_id == focus else "")
        with st.container(key=f"eqsub_{sub_id}"):
            with st.expander(label, expanded=True):
                # (D-055) hover-to-highlight: hovering this subsection lights up its parameters
                # in the sidebar (desktop only; wired client-side in theme.inject_frontend_js).
                # We emit the subsection → sidebar-row-key map as a hidden marker the shim reads,
                # so the TRIGGER is hover, not a click control. The old ⌖ click button is gone —
                # it was pointless when only one subsection shows, and hover reads cleaner.
                _entries = subsection_param_entries(sub_id, level)
                if _entries:
                    _rows, _seen = [], set()
                    for _k, _ in _entries:
                        _rk = param_row_key(_k)
                        if _rk not in _seen:
                            _seen.add(_rk)
                            _rows.append(_rk)
                    st.markdown(
                        f'<span class="eqhl-src" data-params="{" ".join(_rows)}"></span>',
                        unsafe_allow_html=True)
                render(sub_id)
    # D-078: scroll the pane to the focused subsection ONCE per row-title click (bump-guarded —
    # reruns while the focus holds must not re-scroll; same one-shot pattern as the calibration
    # panel's _autoscroll). block:'nearest' is a no-op when the subsection is already in view.
    if focus in shown and bump and st.session_state.get("_eq_scrolled_at") != bump:
        components.html(
            f"""<script>
            (function go(tries) {{
              const el = window.parent.document.querySelector('.st-key-eqsub_{focus}');
              if (el) el.scrollIntoView({{behavior: "smooth", block: "nearest"}});
              else if (tries > 0) setTimeout(function () {{ go(tries - 1); }}, 250);
            }})(8);</script>""",
            height=0)
        st.session_state["_eq_scrolled_at"] = bump
