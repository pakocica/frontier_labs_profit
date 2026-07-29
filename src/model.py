# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Frontier-AI-lab competition — the model (draft v2)
#
# **The single source of truth for all model code** (D-025). The Streamlit widget holds no model
# math: `ui/model_access.py` imports this module and hands its functions to the UI. Illustrations
# live in `model_demo.py`; the guided tour lives in `walkthrough.ipynb`.
#
# Stored as plain Python in **jupytext percent** format (D-085) — `# %%` opens a code cell,
# `# %% [markdown]` a prose cell — so Jupyter and VS Code still open this as a notebook while git
# sees readable text.
#
# **Where the equations live.** The authority on the model's math is what the widget *renders*
# (`ui/equations.py`, the "Equations & calibration" pane), with `Notes/model_draft_v2.md` as the
# written companion. Prose here is deliberately minimal: a stale equation in a comment is worse
# than no equation at all. Each function's own docstring carries its specification.
#
# **Units (spec N1).** States $a$, $c$, $x$ are in **OOMs** — base-10 logs of algorithmic level,
# training compute, and effective compute $x = a + c$. Growth rates ($g_C$, $g_a$) are OOM/yr;
# diffusion rates $\delta$ and the discount rate $r$ are continuous per year; the value slope
# $\nu$ (per OOM) and the compute-price decline $g_p$ (OOM/yr) are **base-10** (D-039). $t = 0$
# is early 2026, normalised $x^L_0 = 0$, so every capability is OOMs above today's frontier.
# Value $W$ is an **index**, $W(0) = 1$, and since D-093 **the money block carries no dollars at
# all**: earnings and cost are both normalised at $t = 0$, so $B_0 = 1$ and the whole finance
# side is measured in *multiples of today's model-building bill*. One parameter survives,
# $\rho$ — coverage at $t = 0$ — and absolute FLOP counts and absolute dollars alike never
# appear. The gap is $\Delta_t = x^L_t - x^F_t$; the follower starts $\Delta_0$ behind, divided
# by `split` into an algorithmic and a compute part.
#
# The acceptance tests and the harvest condition (N5) live in `tests/test_model.py`.

# %%
# ---- Cell E1: imports + Params dataclass ----
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field, replace

# =============================================================================================
# BASE-MODEL CALIBRATION (round 1, closed 2026-07-26; implemented 2026-07-27, D-076).
# Every value in the "BASE" blocks below was ratified by Pavel. Provenance: the per-parameter
# briefs in Notes/calibration/param_docs/ (PDF) with the decisions recorded in ANSWERS.md;
# the implementation contract is Notes/calibration/IMPLEMENTATION_base_sync.md.
# Convention: store FULL precision here, display rounded in the widget.
#
# The base model = constant compute growth, constant (residual) algorithmic growth, exponential
# value map, merged follower catch-up, NO build lag (with constant g_C the lag cancels from the
# whole cost path identically). Everything else is an extension whose defaults are
# UNCHANGED by this pass and are owned by the extensions round.
# =============================================================================================


# The ratified t = 0 compute observable (BASE, TC1b/TC2b): the capability frontier's compute
# growth TODAY, log10(3.24) OOM/yr. Since D-088 this IS Params.g_C0 -- the dial states today's
# growth and Gamma derives the pre-slowdown plateau from it, so the two no longer need to be
# kept in sync by hand (they did between D-086 and D-088).
G_C_TODAY = 0.5105450102066121

@dataclass
class Params:
    # ----- Scenario -----
    T: float = 10.0            # horizon (yr), D-007
    dt: float = 0.005          # integrator step (yr), design N1/N6
    tau: float = 0.0           # PARKED (D-077): release delay. The delayed-revelation extension
                               # is retired from the model statement -- every equation runs on the
                               # DEVELOPED frontier x^L. The machinery below (x^R, _xR_of, the tau
                               # sweep) is kept intact and pinned at tau = 0 so question (b) can be
                               # revived without rebuilding it; while tau = 0, x^R == x^L exactly.

    # ----- A: frontier progress -----
    A1: bool = False           # benchmark A1: adot_L == g_a (pure exogenous)
    g_C0: float = G_C_TODAY    # g_C(0) -- TODAY'S compute growth, log10(3.24) OOM/yr, and
                                       # literally what its name says (D-088). The OBSERVABLE is
                                       # the capability frontier's growth (the most capable
                                       # model's, NOT the largest training run's), where two
                                       # independent routes meet -- the dollar identity 2.4
                                       # (bill) x 1.35 (hardware price-performance) = 3.24, and
                                       # Epoch's capability-frontier reading 3-4x/yr (BASE,
                                       # TC1b/TC2b, ratified).
                                       # NOT a plateau. Between D-086 and D-088 this field held
                                       # the DERIVED pre-slowdown plateau 0.5143888991985982,
                                       # because the curve took a plateau; D-088 moved that solve
                                       # inside Gamma, so the dial is the observable again and
                                       # NOTHING here needs re-fitting when g_C_inf, t_mid or
                                       # p0_c move. (The plateau Gamma now derives from this
                                       # default is bitwise the old literal, which is what makes
                                       # the compute path byte-identical across D-088.)
    g_C_inf: float = 0.13      # EXTENSION (slowdown) -> extensions round (compute-growth floor)
    t_mid: float = 2.3         # EXTENSION (slowdown) -> extensions round. D-082: the
                                       # transition MIDPOINT (yr): half the slowdown has played
                                       # out by t_mid. Default ~ the old xi=0.3 half-decay time
                                       # ln2/0.3 = 2.31.
    p0_c: float = 1.0          # D-084 POSITION dial, in PERCENT: how far along the slowdown
                                       # already is at t = 0 (TODAY). It is what fixes the
                                       # curve's slope -- see slope_span -- so the path reads
                                       # p0_c% now, 50% at t_mid, (100-p0_c)% at 2*t_mid. The
                                       # 1.0 default IS the convention D-082 baked in, so every
                                       # pre-D-084 path is reproduced BITWISE. Envelope
                                       # [1, 25]% PROPOSED -> calibration round.
    g_a: float = 0.5440680443502757    # BASE (GE1c): = log10(3.5). RESIDUAL, not an independent
                                       # estimate: the observable is EFFECTIVE-compute growth
                                       # t_eff_x = 11.34x/yr (data, architecture and post-training
                                       # know-how all included) and g_a = g_eff - g_C0, so the
                                       # RL-compute double-count is impossible by construction.
                                       # 11.34 = 3.24 x 3.50 exactly. Reference-dependent (GE6):
                                       # the reference is 2026 frontier practice at frontier scale.
    alpha: float = 0.7         # BASE (brief 10, D-098): the CES weight on the experiment-compute
                               # channel. CALIBRATED -- it is no longer the 0.5 placeholder that
                               # shipped from the psi-engine round with a grade-F "no observable"
                               # row. Dialled through the OBSERVABLE loss_half_gC ("% of algo
                               # progress lost if experiment-compute growth halved"), which is
                               # alpha/2 at the base eta = 1, so 35% <-> 0.70. Range [0.45, 0.90],
                               # grade C+ (Pavel, ratified 2026-07-28). alpha is EXACTLY inert at
                               # Level 1 (A1 short-circuits the bracket) and under Leontief (the
                               # min() branch never reads it), so this change cannot touch the
                               # base calibration ratified 2026-07-26.
    eta: float = 1.0           # base CES exponent, D-018 (1 => weighted average)
    leontief: bool = False     # eta -> -inf (min) option
    beta0: float = 0.3         # EXTENSION (psi engine) -> extensions round. RENAMED from
                               # rho0 by D-084: rho was doing double duty -- the coverage
                               # ratio rho_t = E_t/B_t and its t=0 dial rho_0 = m/k (D-080)
                               # are the REPORTED outcome and keep the letter; the RSI
                               # feedback scale in psi moves to beta_0.
    gamma: float = 0.08685889638065036   # EXTENSION (psi engine) -> extensions round.
                               # BASE 10 since D-091 (= 0.2/ln10): decades of AI-R&D speed per
                               # OOM of capability, the same units as every other slope in the
                               # model. Super-exponential above ~0.182 (was ~0.42 in nats), N4.

    # ----- B: follower catch-up -----
    # The follower is the COMPETITIVE FRINGE; open-weight models are its measurement proxy, and
    # API-first competitively-priced models count as fringe from their API date (Pavel, 2026-07-26).
    Delta0: float = 0.6151909484915179  # BASE (TL1 final): (7.0/12) * xdot_L0 -- the 7.0-month
                                        # fringe lag on a strict, agentic/long-horizon rule.
                                        # MONTHS ARE THE MASTER (H1): Delta0 is derived from the
                                        # lag and the current speed, never pinned independently.
    split: float = 0.5         # EXTENSION (channels, L6) -> extensions round
    g_a_F: float = 0.3808476310451929  # L3 follower engine: 0.7 x g_a EXACT (Gundlach
                                # central share; extensions-sync 2026-07-28, audit X-10 --
                                # the old 0.31 encoded 0.7 x the pre-D-076 residual)
    delta_dev: float = 0.20    # EXTENSION (channels, L6) -> extensions round
    delta_rel: float = 0.26    # EXTENSION (channels, L6) -> extensions round
    g_CF0: float = 0.5         # L3 follower engine: grade-F scenario knob (Q-5 ruling).
                               # D-088: the fringe's compute growth TODAY, g_CF(0), not its
                               # pre-slowdown plateau.
    g_CF_inf: float = 0.10     # L3 follower engine: grade-F scenario knob (Q-5 ruling)
    t_mid_F: float = 2.3       # EXTENSION (channels, L6) -> extensions round. D-082: the
                                       # follower slowdown's transition midpoint (yr), the same
                                       # universal-curve family as t_mid.
    p0_F: float = 1.0          # D-084: the follower slowdown's OWN position dial (percent of
                                       # its transition already done at t = 0) -- see p0_c.

    # ----- D: value of capability -----
    # (D-093 retired kappa. The dollar coefficient is gone with the dollars: earnings are
    # normalised by their OWN t = 0 value, so the value block needs no scale constant at all.)
    nu: float = 0.3222192947339193   # BASE (TV1'): = log10(2.1). Value multiplier 2.1x per OOM of
                                     # capability TODAY -- pooled median of three independent
                                     # constructions (Davidson value datum 1.86, GATE ramp +
                                     # wage-bill ceiling 2.30, revenue decomposition 2.24).
                                     # D-088: nu = w'(0) literally, at every p0_w. Before D-088
                                     # it was the PRE-easing plateau, so this dial delivered
                                     # 2.089x today rather than the calibrated 2.1x.
    nu_inf: float = 0.09691001300805642  # EXTENSION (value slope transition, D-083) ->
                                       # calibration round. ASYMPTOTIC value slope, =
                                       # log10(1.25): each OOM is worth x1.25 more once
                                       # the transition is done (vs x2.1 today). The L1
                                       # pin nu_inf = nu switches the transition OFF.
    x_mid: float = 10.0        # EXTENSION (saturation) -> extensions round (base pins it huge)
    p0_w: float = 1.0          # D-084: the value-slope easing's OWN position dial (percent of
                                       # the nu -> nu_inf transition already done at today's
                                       # frontier, x = 0) -- see p0_c.

    # ----- C: cost -----
    phi_RD: float = 1.0        # EXTENSION -> extensions round. In the base it is pinned to 0: the
                               # ratified cost anchor is the OBSERVED BILL (compute AND R&D /
                               # researcher overhead together, SB1), so no separate markup exists.
    ell: float = 0.45          # EXTENSION (enters with the slowdown; EL2'/EL3' ratified 0.45 yr,
                               # lognormal 90% CI [0.25, 1.3]). ABSENT FROM THE BASE: with constant
                               # g_C, c(t+ell) - c(ell) = g_C*t identically, so under bill anchoring
                               # ell cancels from the entire base cost path.
    # (D-093 retired B0 too. D-090 had already made it the observable cost(0) = k*R0; normalising
    # by that observable turns it into the constant 1, which no dataclass needs to carry. The
    # cost path is still referenced to c^L(ell), which is what makes B_0 = 1 hold at every dial.)
    g_p: float = 0.14          # BASE (TC2b): effective compute-price decline, OOM/yr = 1.38x/yr.
                               # The HARDWARE price-performance leg is trusted directly (Hobbhahn
                               # et al.); the old "bill residual" reading is retired. Implied bill
                               # growth 10^(g_C0 - g_p) = 2.35x/yr vs Cottier's observed 2.4x/yr
                               # -- a 2% miss, documented rather than fitted away.
    r: float = 0.08            # discount rate (/yr) -- user-cost extension only

    # ----- C: the money side -- ONE parameter (D-093, Pavel: "the only parameter user should
    # see is rho, nothing else matters in these two sections") -----
    # Both money legs are normalised at t = 0, so the block is in MULTIPLES OF TODAY'S BILL:
    #
    #     E_t = rho * [W(x^L_t) - W(x^F_t)] / [W(x^L_0) - W(x^F_0)]      =>  E_0 = rho
    #     B_t = 10^{c^L_{t+ell} - c^L_ell} * 10^{-g_p t}                 =>  B_0 = 1
    #
    # HOW EXACTLY, measured -- the identities are algebraic, the arithmetic is not:
    #   * E_0 = rho is BITWISE at every base dial.
    #   * B_0 = 1 is BITWISE at ell = 0 (the base). With a build lag it holds to a few ulp
    #     (5 at the shipped ell = 0.45, 35 at ell = 1.3, 260 at ell = 2.5 under a hard
    #     slowdown), because cost_flow de-lags with the EXACT integral c_L_closed(ell) while
    #     simulate reads c^L(t+ell) off the RK4 grid by interpolation. That gap predates
    #     D-093 -- cost(0) was k*R0 to the same relative precision, which is why test_19
    #     was written with a tolerance -- and normalising neither caused nor cured it.
    #   * TWO PARKED EXTENSIONS BREAK THEM BY CONSTRUCTION, as they always did (D-077's own
    #     "still open" list): chi > 0 charges defense costs at t = 0 too, so E_0 = rho(1-0.4chi),
    #     and own_mode multiplies the whole cost leg by the user-cost factor, so B_0 ~ 1.75.
    #     Both are recorded here and asserted in test_18 rather than tolerated silently.
    #
    # and hence coverage rho_t = E_t/B_t starts at rho by construction. R0, m_margin, k_build,
    # kappa and B0 are GONE: only the ratio m/k was ever identified (test_18's scale invariance
    # is the machine proof), so the other four were pinned numbers doing no work in the model.
    # THEY SURVIVE AS EVIDENCE, NOT AS PARAMETERS -- the coverage source menu (_COVERAGE_SOURCES
    # below), Notes/calibration/cov0_source_menu_2026-07-27.md and the evidence register document
    # exactly how rho = m/k is derived from reported dollar figures. Inputs to a calibration are
    # not parameters of a model, and this dataclass now says so.
    rho: float = 0.5333333333333333   # BASE: coverage at t = 0 = m/k = 0.40/0.75. "Labs currently
                               # earn ~53 cents per dollar of model-building spend"; break-even
                               # is rho_t = 1. TODAY'S SNAPSHOT, never a structural constant --
                               # the dynamics move earnings and cost at different rates and
                               # question (a) is when they cross. The widget dials it in PERCENT
                               # (ui/state.py APP_RANGES['cov0'], envelope [33, 56]); the
                               # FIN4(b) basis ruling is the one open finance question and can
                               # only lower it.

    # ----- Extensions (dials; default OFF) -----
    phi_mix: float = 0.0       # II.2 public-knowledge mix (effective a~_L blend)
    conduct_gap: bool = False  # II.4 gap-dependent conduct (provisional form, D-027). Under D-077
                               # this is a DIMENSIONLESS multiplier on earnings, normalised to 1 at
                               # Delta0 -- it survives theta's retirement as a shape, not a level.
    conduct_scale: float = 0.7 # II.4 scale Delta_theta (OOM)
    chi: float = 0.0           # II.5 distillation-defense dial (0..0.35)
    own_mode: bool = False     # II.6 ownership / user-cost variant (provisional)
    delta_K: float = 0.3       # II.6 GPU economic depreciation (/yr)
    labor_line: bool = False   # II.7 decoupled labor line (run only if headline positive)
    L0: float = 10.0           # II.7 labor level today ($B/yr)
    g_w: float = 0.05          # II.7 wage growth (/yr)

# %% [markdown]
# ## Component A — frontier progress
#
# The CES exponent $\eta$ is the substitution reading: $\eta = 1$ (the D-018 base) makes the two
# research inputs near-substitutes, so the bracket is a weighted **average** and a compute
# slowdown barely dents algorithmic progress; $\eta \to 0$ is Cobb–Douglas; $\eta < 0$ makes them
# complements; `leontief=True` is the $\eta \to -\infty$ limit, where the scarcer input rules.
#
# **Caveat (N4).** With $\gamma > 0$ the $\psi$ feedback makes the law formally super-exponential.
# Above $\gamma \approx 0.4$ it drives a finite-time singularity inside the 10-yr horizon; the
# `psi_boost_share` diagnostic reports how much of $\dot a^L$ comes from $\psi$ having grown, and
# the widget flags it once it stops being small (>25%). $\gamma = 0$ freezes AI assistance at
# today's level.

# %%
# ---- Cell E2: Component A -- compute growth, psi, algo growth ----
def _logistic(u, y_minus_inf, y_inf, u_mid, s):
    """The RAW base-10 logistic, in plateau-and-slope form. PRIVATE: nothing outside this cell
    should dial a model transition this way -- use `gamma_curve`, which reaches it through the
    two identities. Kept exposed only because W()'s Level-1 compatibility branch evaluates a
    genuinely different object (the retired BOUNDED logistic, whose lower plateau is a hard 0
    and whose slope is nu itself) and must stay byte-identical.

        _logistic(u) = y_minus_inf + (y_inf - y_minus_inf) / (1 + 10^{-s(u - u_mid)})

    Evaluated as a SINGLE exp of the difference -- the same arithmetic the value map used before
    D-082, so routing anything through this primitive is bit-identical (asserted in test_20)."""
    u = np.asarray(u, dtype=float)
    z = np.clip(-s * np.log(10.0) * (u - u_mid), -700.0, 700.0)
    return y_minus_inf + (y_inf - y_minus_inf) / (1.0 + np.exp(z))

# The POSITION dial's enforced domain, in PERCENT (D-088). Open at both ends:
#   p0 -> 0   the transition has not started, the slope is infinite;
#   p0 -> 50  the midpoint is TODAY, and past it the midpoint is in the past (s <= 0, a curve
#             running backwards). The shipped envelope [1, 25] sits strictly inside.
# The plateau identity's own singularity is at p0 = 100 (where 1 - p0/100 vanishes and every
# plateau diverges); 50 is the tighter of the two bounds and the one that carries meaning, so
# it is the one enforced. Violations RAISE -- a silent NaN or a backwards curve would propagate
# into a whole simulated path before anything noticed.
P0_MIN_PCT, P0_MAX_PCT = 0.0, 50.0

def slope_span(p0):
    """IDENTITY 1 -- the slope, from the position. s * u_mid, a transition's slope measured in
    units of its own midpoint (D-084, Pavel: "whenever S is used to define the transition
    another parameter representing s should be introduced ... for example, how far on the
    s-curve we already are"):

        p0 = the PERCENTAGE of the transition already completed at u = 0 (today),
        s * u_mid = log10((100 - p0) / p0).

    Since (100-p0)/p0 is the odds of not-yet-done against done, this reads: slope x midpoint =
    the log-odds of the journey remaining. The whole curve then follows off p0: it is p0% along
    at u = 0, 50% at u_mid and -- by the logistic's symmetry -- exactly (100 - p0)% at 2*u_mid,
    for ANY p0.

    PERCENT units are load-bearing, not cosmetic. (100 - 1)/1 is exactly 99.0 in binary floating
    point, so the default p0 = 1 reproduces the retired constant SLOWDOWN_SLOPE = log10(99)
    BITWISE and every pre-D-084 path is byte-identical (test_20). Percent is also exact at
    strictly more dial values than a fraction: (1-0.05)/0.05 rounds to 18.999999999999996 while
    (100-5)/5 is exactly 19.0. And it is Pavel's own idiom -- "we are in the bottom 10% of the
    s-curve"."""
    if not (P0_MIN_PCT < p0 < P0_MAX_PCT):
        raise ValueError(
            f"position p0 = {p0} is outside the enforced domain ({P0_MIN_PCT}, {P0_MAX_PCT}) "
            "percent: p0 -> 0 is an unstarted transition with infinite slope, and p0 >= 50 puts "
            "the midpoint at or before today, which is not a transition. Shipped envelope [1, 25].")
    return float(np.log10((100.0 - p0) / p0))

GammaShape = namedtuple("GammaShape", "y_minus_inf s k")

def gamma_shape(y_today, y_inf, u_mid, p0):
    """THE complicated calculation, written ONCE (D-088, Pavel: "You define the complicated
    calculation once, and then just use the convenient Gamma function over and over"). Turns the
    four DIALLED OBSERVABLES into the two things the logistic actually needs.

    IDENTITY 1 (the slope, from the position) -- see slope_span:
        s = log10((100 - p0)/p0) / u_mid.
    IDENTITY 2 (the plateau, from TODAY'S VALUE) -- nobody dials y_{-inf}, so derive it. Today's
    value is the p0-weighted average of the two plateaus, y(0) = (1-f) y_{-inf} + f y_inf with
    f = p0/100; un-mixing that average gives
        y_{-inf} = (y(0) - f * y_inf) / (1 - f).
    As f -> 0 the plateau IS today (nothing has happened yet). For a slowdown (y_inf < y(0)) it
    returns y_{-inf} > y(0) -- correctly: we have already come part of the way down.

    Returns (y_minus_inf, s, k) with k = s * ln 10, the slope in NATURAL units, which the two
    closed-form integrals (c_L_closed, w_log) need. k is returned rather than recomputed from s
    because the two orderings differ in the last bit -- log10(99)*ln10/2.3 is 1.9978781957106917
    while (log10(99)/2.3)*ln10 is ...15 -- and every shipped path was integrated with the first.

    ZERO AMPLITUDE IS EXACT. When y_inf == y_today the transition has nothing to traverse, so
    every plateau is today's value and the curve is the constant y_today. The algebra says so
    ((y - f y)/(1 - f) = y) but IEEE rounding does not always agree, and a plateau off by one
    ulp leaves a non-zero amplitude that turns Level 1's constant-growth pin into a faint
    slowdown. Short-circuited, so "amplitude zero => exactly constant, for ANY u_mid and ANY p0"
    is a machine guarantee -- which is what apply_level_pins (g_C_inf := g_C0, nu_inf := nu,
    the follower's 0/0) and base_params() both rest on."""
    span = slope_span(p0)                       # raises outside (0, 50)%
    s = span / u_mid
    k = span * np.log(10.0) / u_mid
    y_today = float(y_today)
    y_inf = float(y_inf)
    if y_inf == y_today:
        return GammaShape(y_today, s, k)
    f = float(p0) / 100.0
    return GammaShape((y_today - y_inf * f) / (1.0 - f), s, k)

def gamma_curve(u, y_today, y_inf, u_mid, p0):
    """Gamma -- the ONE universal transition curve, dialled in OBSERVABLES ONLY (D-088; the
    lineage is D-082 "first define a universal function for the S-curve, then use it to define
    g_c ... the same way for W", D-084 which made the slope explicit, and D-086 which solved the
    plateau at one call site. Pavel: "I want you to revise the dynamics level so that the Gamma
    function has p_0 as input instead of s"). Renamed from Lambda by D-088, Pavel's own letter.

        Gamma(u; y(0), y_inf, u_mid, p0) = y_{-inf} + (y_inf - y_{-inf}) / (1 + 10^{-s(u-u_mid)})

        y(0)   TODAY'S VALUE -- the thing that is observed and dialled,
        y_inf  the u -> +infinity asymptote,
        u_mid  the midtime: the curve is exactly halfway there,
        p0     the POSITION: what percent of the transition is already behind us at u = 0,

    with y_{-inf} and s both DERIVED, once, in gamma_shape.

    The point of the signature: Gamma(0) = y(0) identically, for every use. The model's standing
    rule that an OBSERVATION NEVER DEPENDS ON A PARAMETER (D-076) becomes a property of the curve
    rather than something re-established at each call site -- moving u_mid or p0 says where the
    transition STARTED, never what it is now. (In exact arithmetic Gamma(0) = y(0) exactly; in
    IEEE double it holds to a few ulp, and test_20d pins the size of that.)

    Note what y_{-inf} is NOT: it is not "the initial value". The curve never equals its own
    starting plateau; y_{-inf} is the limit infinitely far in the past, which is exactly why it
    is a bad thing to ask a user for and is derived here instead."""
    sh = gamma_shape(y_today, y_inf, u_mid, p0)
    return _logistic(u, sh.y_minus_inf, y_inf, u_mid, sh.s)

def compute_growth(t, p):
    """g_C(t): frontier physical-compute growth (OOM/yr). Spec I.1. D-082 (Pavel): the universal
    curve, running down to the terminal floor g_C_inf and halfway there at t_mid.

    D-088: the dial g_C0 is TODAY'S growth -- g_C(0) == p.g_C0, at every t_mid and every p0_c --
    and the pre-slowdown plateau is derived inside Gamma. (D-086 established this for this one
    call site by solving the plateau in invert_targets; D-088 promoted the solve into the
    definition, so the same guarantee now holds for the fringe path and the value slope for
    free.) Exactly the constant g_C0 whenever g_C_inf = g_C0 -- the below-L2 pin -- for ANY t_mid
    and ANY p0_c, so the base model never sees the shape."""
    return gamma_curve(t, p.g_C0, p.g_C_inf, p.t_mid, p.p0_c)

_GC_TODAY_CACHE: dict = {}

def gc_today(p):
    """g_C(0) = Gamma(0) -- TODAY'S realised compute growth, the anchor D-086 P1-2 normalises
    the CES experiment input and the t = 0 target observables on.

    Since D-088 this equals the DIAL p.g_C0 -- that is the point of the new signature -- but it
    still EVALUATES the curve rather than returning the dial, deliberately. Gamma(0) = y(0) is a
    rounding identity, not a representable one (it agrees to a few ulp, test_20b measures it),
    and algo_growth_L needs the experiment ratio compute_growth(t)/gc_today(p) to be EXACTLY 1
    at t = 0. Reading the dial here instead would make it 1 +/- 1 ulp and move every RK4 path.

    Memoised purely for speed: algo_growth_L calls this on every RK4 stage, and evaluating the
    logistic there cost ~65% of a simulate. The cached value comes from the SAME
    compute_growth(0.0, p) call, so it is bit-identical to recomputing it -- the key is the
    complete set of parameters Gamma(0) depends on, so two Params that differ anywhere else
    (or agree here) are correctly treated as identical."""
    key = (p.g_C0, p.g_C_inf, p.t_mid, p.p0_c)
    v = _GC_TODAY_CACHE.get(key)
    if v is None:
        if len(_GC_TODAY_CACHE) > 4096:       # bound it: one entry per distinct MC draw
            _GC_TODAY_CACHE.clear()
        v = float(compute_growth(0.0, p))
        _GC_TODAY_CACHE[key] = v
    return v

def psi(x, p):
    """AI-assistance multiplier psi(x) = 1 + beta0 * 10^{gamma (x - x_L0)}, x_L0=0. Spec I.1.
    (D-084: beta0 was rho0 until the coverage ratio rho took the letter for good.)

    D-091 (Pavel: "Use always base 10 to keep it consistent"): BASE 10, like every other exponent
    in the model. gamma is now decades of AI-R&D speed per OOM of capability -- the same units as
    every other slope here -- rather than nats per OOM. The stored constant divides by ln 10:
    the 0.2 default becomes 0.08685889638065036.

    THE EVALUATION ORDER IS LOAD-BEARING; do not "tidy" it. 10^{gamma x} is computed as
    exp((gamma * ln10) * x), multiplying gamma by ln 10 FIRST: at the default that product is
    bitwise 0.2, so every pre-D-091 path is reproduced exactly. Writing gamma * (ln10 * x)
    instead re-associates the multiply and moves the last bits. The clip stays on the SAME
    quantity (the natural-log exponent) for the same reason.

    Not every gamma round-trips exactly -- (g/ln10)*ln10 == g holds at 0.2, 0.4, 0.42, 0.35 but
    fails at 0.25, 0.3, 0.5 -- so an MC draw at one of those shifts the exponent by 1 ulp. That
    is inherent to re-basing a CONTINUOUS dial, not a model change: the shipped defaults and both
    golden fixtures are exact (default 0.2; Level 1 pins gamma = 0).

    Exponent clipped for numerical safety in the explosive (super-exponential) regime; W-saturation
    then bounds revenue downstream. The clip only bites once psi is astronomically large (flagged)."""
    return 1.0 + p.beta0 * np.exp(np.clip((p.gamma * np.log(10.0)) * x, -700.0, 80.0))

def _ces_bracket(R_psi, R_gc, p):
    """CES aggregate of the two normalized inputs; handles eta=1, eta->0, eta<0, Leontief."""
    a = 1.0 - p.alpha
    b = p.alpha
    if p.leontief:                       # eta -> -inf : Leontief (scarcer input rules)
        return np.minimum(R_psi, R_gc)
    if abs(p.eta) < 1e-8:                 # eta -> 0 : Cobb-Douglas limit
        return R_psi**a * R_gc**b
    if abs(p.eta - 1.0) < 1e-12:          # eta = 1 : weighted average
        return a * R_psi + b * R_gc
    # general CES, guarded against overflow/NaN
    inner = a * R_psi**p.eta + b * R_gc**p.eta
    inner = np.maximum(inner, 1e-300)
    return inner**(1.0 / p.eta)

ALPHA_LOSS_CEILING = 0.5   # the observable saturates here: loss = 50% <=> alpha = 1, at EVERY eta

def alpha_from_loss(loss, eta, leontief=False):
    """THE alpha INVERSION (D-098). loss is a FRACTION, not percent.

    The observable is `loss_half_gC`: "what fraction of algorithmic progress is lost if
    experiment-compute growth is HALVED?" Both CES channels equal 1 at t = 0, so halving the
    compute channel sets R_gc = 1/2 and the bracket reads

        1 - loss = [ (1-alpha) + alpha * 2^{-eta} ]^{1/eta}    =>    (1-loss)^eta
                 = 1 - alpha (1 - 2^{-eta}),

    which inverts in closed form. At the base eta = 1 this is simply loss = alpha/2, so the
    ratified spot alpha = 0.70 IS the 35% dial. The eta -> 0 branch is the Cobb-Douglas limit
    1 - loss = 2^{-alpha}. The threshold below is `_ces_bracket`'s OWN 1e-8, deliberately: any
    other value would make the dial and the model disagree in a band around eta = 0.

    Two properties make one slider work across the whole eta menu: loss = 0 <=> alpha = 0 and
    loss = 50% <=> alpha = 1 at EVERY eta, and alpha is strictly increasing in loss in between.
    So the observable's natural domain is (0, 50%) and it can never produce an alpha outside
    (0, 1). Holding the OBSERVABLE fixed while eta moves is the point: the drag is what the
    evidence measures, so a more complementary eta implies a LOWER alpha, and the bottleneck
    evidence is spent once rather than twice (brief 10 sec. 6.2).

    DECLARED EXCEPTION (D-098). Under Leontief the bracket is min(R_psi, R_gc) and never reads
    alpha at all: the model always loses exactly 50%, whatever the dial says. The dial therefore
    CANNOT be honoured there, the sidebar disables the row and says so, and this function
    returns nan rather than a number the model will not deliver."""
    if leontief:
        return float('nan')
    loss = float(loss)
    if abs(eta) < 1e-8:                       # eta -> 0 : Cobb-Douglas limit
        return float(-np.log2(1.0 - loss))
    return float((1.0 - (1.0 - loss)**eta) / (1.0 - 2.0**(-eta)))

def loss_from_alpha(alpha, eta, leontief=False):
    """Forward map, the exact inverse of alpha_from_loss. Returns a FRACTION.

    Under Leontief the honest answer is the one the model delivers -- exactly 50% -- not the
    value the weight would imply, because the weight is not read."""
    if leontief:
        return ALPHA_LOSS_CEILING
    alpha = float(alpha)
    if abs(eta) < 1e-8:
        return float(1.0 - 2.0**(-alpha))
    return float(1.0 - (1.0 - alpha * (1.0 - 2.0**(-eta)))**(1.0 / eta))

def algo_growth_L(t, x_L, p):
    """adot_L(t): leader algorithmic progress (OOM/yr). Spec I.1.
    g_a * [ (1-a)(psi(x_L)/psi(0))^eta + a (g_C(t)/g_C(0))^eta ]^{1/eta}.
    Benchmark A1: adot_L == g_a.

    D-086 P1-2: the experiment input is normalised on TODAY'S realised compute growth
    g_C(0) = Gamma(0), not on the pre-transition plateau. Between D-084 and D-088 the two were
    different NUMBERS held in the same field -- the plateau was g_C0 and today's growth was
    g_C0 + (g_C_inf - g_C0) * p0_c/100 -- so the old plateau normalisation made
    BOTH ratios not-1 at t = 0 and adot_L(0) != g_a, silently breaking the residual
    identification the spec advertises (g_a = g_eff - g_c read off t = 0 observables). With
    this anchor adot_L(0) == g_a and xdot_L(0) == the dialled effective growth EXACTLY, at
    every p0_c. This is Pavel's own D-076 rule -- an observation never depends on a
    parameter -- applied to the D-084 position dial."""
    if p.A1:
        return p.g_a
    R_psi = psi(x_L, p) / psi(0.0, p)              # research input, normalized to 1 at t=0
    R_gc = compute_growth(t, p) / gc_today(p)     # experiment input, == 1 at t = 0 exactly
    return p.g_a * _ces_bracket(R_psi, R_gc, p)

def psi_boost_share(t, x_L, p):
    """Fractional boost to adot_L coming from psi having grown above psi(0) (RSI feedback).
    0 at t=0; N4 says flag when it stops being small (>~25%)."""
    if p.A1:
        return np.zeros_like(np.asarray(t, dtype=float))
    R_psi = psi(x_L, p) / psi(0.0, p)
    R_gc = compute_growth(t, p) / gc_today(p)              # same anchor as algo_growth_L (D-086)
    full = _ces_bracket(R_psi, R_gc, p)
    frozen = _ces_bracket(np.ones_like(np.asarray(R_gc, dtype=float)) if np.ndim(R_gc) else 1.0, R_gc, p)
    full = np.asarray(full, dtype=float)
    frozen = np.asarray(frozen, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        share = 1.0 - frozen / full
    return np.where(full > 0, share, 0.0)

def _softplus(z):
    """ln(1 + e^z), overflow-safe -- the antiderivative kernel of the logistic sigma."""
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))

def c_L_closed(t, p):
    """c^L(t): the leader's cumulative log-compute in CLOSED FORM (OOM above the 2026 frontier).
    The compute path is exogenous -- cdot^L = g_C(t) = Gamma(t; g_C0, g_C_inf, t_mid, p0_c)
    (D-082/D-088) -- and the logistic integrates exactly: with the PLATEAU g_c^pre and the
    natural-units slope k both taken from gamma_shape (D-088: this is an INTEGRAL of the curve,
    not an evaluation of it, so it cannot go through gamma_curve -- but it must not re-derive
    the shape either), and sigma's antiderivative softplus/k,

        c^L(t) = g_c^pre*t + (g_C_inf - g_c^pre)*[softplus(k(t-t_mid)) - softplus(-k*t_mid)]/k.

    Used by cost_flow (cell E5) to de-lag the normalised bill through the exact integrated
    path c^L(ell), never a g_C*ell linearization (Pavel, 2026-07-26: "use c(t+l), not
    g_C(t+l), as the growth might vary between t and t+l"). Agrees with simulate()'s RK4 path to
    integrator precision; in the base (g_C_inf = g_C0, constant growth) gamma_shape returns the
    plateau EXACTLY equal to g_C0, so the bracket multiplies zero and c^L(t) = g_C0*t exactly."""
    t = np.asarray(t, dtype=float)
    sh = gamma_shape(p.g_C0, p.g_C_inf, p.t_mid, p.p0_c)
    return sh.y_minus_inf * t + (p.g_C_inf - sh.y_minus_inf) * (
        _softplus(sh.k * (t - p.t_mid)) - _softplus(-sh.k * p.t_mid)) / sh.k

# %% [markdown]
# ## Component B — follower catch-up
#
# What catches up is the follower's *algorithmic* level — compute cannot be copied. The two
# channels differ in what they run on: the **developed** channel ($\delta_{dev}$, espionage,
# papers, mobility) diffuses methods and so runs on the algorithmic gap, while the **released**
# channel ($\delta_{rel}$, distillation) copies what the served model can do and so runs on the
# capability gap. Both die at parity (N2, D-015). The observed open-vs-closed lag pins the
# **total** $\delta_{dev} + \delta_{rel}$; the *split* between them is the free parameter, and it
# is precisely what the parked question (b) turns on.

# %%
# ---- Cell E3: Component B -- follower compute growth and algo catch-up ----
def follower_compute_growth(t, p):
    """g_CF(t): follower physical-compute growth (OOM/yr). The SAME universal curve as the leader
    (D-082), with the follower's OWN parameters throughout -- its own today-value, floor and
    midtime, and (D-084) its own position dial p0_F, so how far the fringe's slowdown has already
    gone is never silently tied to the leader's. D-088: g_CF0 is the fringe's growth TODAY, not
    its pre-slowdown plateau -- g_CF(0) == p.g_CF0 at every t_mid_F and every p0_F."""
    return gamma_curve(t, p.g_CF0, p.g_CF_inf, p.t_mid_F, p.p0_F)

def algo_growth_F(aL, aF, xR, xF, p):
    """adot_F: follower algorithmic progress (OOM/yr). Spec I.2.
    g_a_F + delta_dev * max(a~L - aF, 0) + (1-chi) delta_rel * max(xR - xF, 0).
    II.2: a~L = (1-phi_mix) aL + phi_mix aF blends in the public stock.
    II.5: chi throttles the released-model (distillation) channel.

    BOTH channels are floored at zero (D-086 P1-1; Pavel: "Floor it -- imitation only.").
    They measure IMITATION, so they vanish at parity and never reverse: a follower that has
    already absorbed a method does not un-learn it because the leader's own algorithmic level
    later falls behind. Before D-086 only the RELEASED channel carried its guard, while the
    displayed note and spec I.2 both claimed each gap was floored -- and the unguarded one is
    the one that binds. The algorithmic gap goes negative on EVERY level of the shipped ladder
    (the follower's distilled algorithms outrun the leader's own algorithmic progress while it
    stays behind on compute), so the developed channel ran backwards over 38-56% of the
    horizon and the leader DRAINED the follower. See the D-086 log entry for the measured
    paths; N2 now holds literally rather than approximately."""
    aL_eff = (1.0 - p.phi_mix) * aL + p.phi_mix * aF          # II.2 (phi_mix=0 -> aL)
    dev = p.delta_dev * np.maximum(aL_eff - aF, 0.0)           # developed-model channel (imitation only)
    rel = (1.0 - p.chi) * p.delta_rel * np.maximum(xR - xF, 0.0)  # released-model channel (cap x^F<=x^R)
    return p.g_a_F + dev + rel

# %%
# ---- Cell E4: Component D -- value of capability (an INDEX), and the II.4 conduct multiplier ----
def w_log(x, p):
    """w(x) = log10 W(x) (D-083, Pavel: "I would like log(W_t) to grow exponentially with
    initial growth g_W and asymptotic growth g_W,infty. You can simply define w_t = log(W_t)").
    Value's growth SLOPE in capability rides the universal S-curve -- its third use:

        w'(x) = S(x; nu, nu_inf, x_mid),        w(0) = 0,

    nu TODAY, the floor nu_inf asymptotically, halfway at x_mid (the TRANSITION midpoint --
    re-keyed by D-083; no longer a half-saturation point, and the hard value ceiling is RETIRED:
    value grows without bound at the floor slope). D-084: the slope comes from this use's OWN
    position dial p0_w -- the easing is p0_w% done at today's frontier, 50% at x_mid and
    (100-p0_w)% at 2*x_mid.

    D-088 CORRECTION: nu is w'(0), literally TODAY'S value slope. It used to be the PRE-easing
    plateau, so a user dialling "2.1x per OOM" got 2.089x today -- the dial lying by 0.7%, the
    same defect D-086 fixed on the compute path and did not reach here. nu is calibrated as a
    today-observable (TV1'), and invert_targets has always read it as one (nu = log10(t_value_x)),
    so the curve is what was wrong. gamma_shape derives the plateau; the closed form below is the
    softplus integral of that curve, exactly as c_L_closed; the w(0) = 0 anchor IS the D-077
    index anchor."""
    x = np.asarray(x, dtype=float)
    sh = gamma_shape(p.nu, p.nu_inf, p.x_mid, p.p0_w)
    return sh.y_minus_inf * x + (p.nu_inf - sh.y_minus_inf) * (
        _softplus(sh.k * (x - p.x_mid)) - _softplus(-sh.k * p.x_mid)) / sh.k

def W(x, p):
    """W(x): the value of capability x as a MULTIPLE OF TODAY'S FRONTIER -- W(0) = 1 by
    construction. D-083: W = 10^{w(x)} with the SLOPE-TRANSITION log form (see w_log); the
    bounded logistic and its ceiling Wbar are retired.

    TRANSITION-OFF route: under the Level-1 pins (nu_inf = nu AND x_mid at the X_MID_EXP
    sentinel, >= 100) this evaluates the PRE-D-083 bounded logistic byte-for-byte -- the merge
    lock's Level-1 digest is absolute (test_level_merge), and on any reachable x the two forms
    agree far beyond double precision (the x_mid = 200 logistic is 10^{nu x} to ~1e-60).
    Everywhere else (x_mid is a real transition midpoint) the general w-form runs; nu_inf = nu
    there gives the exact straight line w = nu*x, as D-083 intends.

    This branch is the one place that reaches _logistic directly rather than gamma_curve, and
    deliberately (D-088): the retired object is a BOUNDED logistic whose lower plateau is a hard
    0 and whose slope is nu itself -- it has no today-value and no position, so it is not a
    transition in Gamma's sense. Routing it through the identities would round it."""
    if p.nu_inf == p.nu and p.x_mid >= 100.0:
        nl = p.nu * np.log(10.0)                              # the pre-D-083 evaluation,
        lg = np.clip(nl * p.x_mid, -700.0, 700.0)             # kept byte-identical
        Wbar = 1.0 + np.exp(lg)
        return _logistic(x, 0.0, Wbar, p.x_mid, p.nu)
    return np.exp(np.clip(np.log(10.0) * w_log(x, p), -700.0, 700.0))

def W_exp_approx(x, p):
    """Exponential-regime form of the index: 10^{nu x} (the x << x_mid limit of W). Spec I.3."""
    x = np.asarray(x, dtype=float)
    return np.exp(np.clip(p.nu * np.log(10.0) * x, -700.0, 700.0))

def gap_index(base):
    """W(x^L_0) - W(x^F_0) = W(0) - W(-Delta0): the value gap the leader earns on AT t = 0, in
    INDEX units (multiples of today's frontier value). At the calibrated base it is
    1 - 10^{-nu*Delta0} = 0.3664606.

    This is the NORMALISER of the earnings leg (D-093): E_t = rho * gap(t) / gap(0), so it is
    the reason E_0 = rho holds identically instead of being solved for. Level-aware by
    construction, since it depends on nu, Delta0 AND x_mid -- the exponential-value levels pin
    x_mid huge, the logistic levels use the dial, and the normalisation tracks either without
    anything being re-fitted.

    (Lives here, beside W, rather than in the targets cell where it sat until D-093: simulate
    itself calls it now, so it is part of the value block, not a calibration helper.)"""
    return float(W(0.0, base) - W(-base.Delta0, base))

def conduct_mult(gap, p):
    """II.4 (extension, default OFF): a DIMENSIONLESS multiplier on earnings, normalised to 1 at
    the initial gap, so switching it on cannot move the t = 0 observable. Constant 1 when off.

    This is what survives of the retired theta (D-077): theta's LEVEL was never identified --
    it was absorbed into the earnings normalisation and is now gone entirely (D-093) -- but the
    SHAPE -- the idea that a wider lead is priced differently -- is a
    real extension and keeps its dial. mult(Delta) = (1-e^{-Delta/Ds}) / (1-e^{-Delta0/Ds})."""
    if not p.conduct_gap:
        return 1.0
    Ds = p.conduct_scale
    gap = np.asarray(gap, dtype=float)
    num = 1.0 - np.exp(-np.maximum(gap, 0.0) / Ds)
    den = 1.0 - np.exp(-p.Delta0 / Ds)
    return num / den

# %%
# ---- Cell E5: Component C -- cost flow ----
def cost_flow(t, c_L_at, c_L_ell, p):
    """Leader model-building cost flow, IN MULTIPLES OF TODAY'S BILL. Spec I.4 / N3. D-093
    normalises D-090's re-based form by its own t = 0 value:

        B_t = 10^{c_L(t+ell) - c_L(ell)} * 10^{-g_p t},        B_0 = 1 identically.

    c_L_at = c_L(t+ell). The firm still pays today for the compute of the model shipping at t+ell
    (ANCHOR, D-036 4th amendment). What D-093 removed is the stored constant in front: D-090 had
    already made it the OBSERVABLE k*R0 = $75B, and dividing the whole block through by that
    observable is what lets the finance side carry one parameter (rho) instead of four.

    THE UNITS ARE THE POINT. Cost is no longer $B/yr; it is "times what the leader spends on
    model-building this year". Coverage rho_t = E_t/B_t is unchanged in meaning and value --
    both legs were rescaled by the same k*R0 -- while profit Pi_t = E_t - B_t becomes
    dimensionless. Nothing here can be read as a dollar figure any more, deliberately.

    WHY THE TWO-STEP ARITHMETIC BELOW -- do not "simplify" it to 10.0**(c_L_at - cl - g_p*t).
    Two constraints it encodes that the collapsed form does not:
      * the de-lagging uses c_L_closed(p.ell, p), the EXACT integrated compute path (Pavel,
        2026-07-26), and deliberately NOT the c_L_ell argument, which simulate obtains by
        INTERPOLATION on the RK4 grid and which therefore differs in the last few digits. The
        argument is kept because it is what the caller has, and because the difference between
        the two is exactly the sort of thing a future edit would otherwise "tidy" into a
        discrepancy;
      * (1 + phi_RD) appears in both places so that its cancellation is STRUCTURAL rather than
        assumed -- test_22 pins that phi_RD is inert, and it stays inert by construction here.
    Ordering also still matters numerically: D-090 measured the collapsed form moving the path
    2-3 ulp at the shipped ell = 0.45, because (X/10^a)*10^b and X*10^(b-a) round differently.

    WHAT THE USER SETS (D-076, SB1/SB-R; D-093): nothing on this leg. The bill's LEVEL is the
    normaliser, so the only finance dial left is rho -- coverage at t = 0. Consequences worth
    knowing:
      * moving ell or the slowdown re-anchors nothing -- an observation cannot depend on a
        parameter, and B_0 is now the constant 1 rather than a number re-derived to stay
        consistent;
      * in the BASE model ell = 0 and phi_RD = 0, so the whole cost side is the compute path
        10^{c_L(t)} deflated by falling prices 10^{-g_p t}, growing at 10^{g_C0 - g_p} = 2.35x/yr.
    phi_RD > 0 (a retired extension) re-splits the same normalised bill into a compute leg and an
    overhead leg; it cancels identically here, which is what test_22 pins."""
    scale = 1.0 / ((1.0 + p.phi_RD) * 10.0**float(c_L_closed(p.ell, p)))
    base = (1.0 + p.phi_RD) * scale * 10.0**(c_L_at - p.g_p * t)   # D-039: g_p in OOM/yr
    if p.own_mode:
        # II.6 (provisional): Jorgenson user cost u = p_K (r + delta_K + g_p); competitive-rental
        # equivalent has price r + g_p, so owners additionally bear delta_K/(r+g_p). SIMPLIFIED:
        # captures the depreciation load, NOT the full CAPEX front-loading (flagged provisional).
        # r is a natural rate, so g_p (log10, D-039) is converted to natural units here.
        gp_nat = p.g_p * np.log(10.0)
        base = base * (p.r + p.delta_K + gp_nat) / (p.r + gp_nat)
    return base

# %% [markdown]
# ## Numerical integration (N1/N6)
#
# Fixed-step **RK4** on a $dt = 0.005$ yr grid — a fixed grid is what makes the delayed term
# exact. The **leader is autonomous** (its dynamics depend on neither the follower nor $x^R$), so
# it is integrated first on a half-step fine grid; the follower is then integrated on the coarse
# grid, reading the leader path off the fine grid *by index* rather than by interpolation, since
# every RK4 stage falls exactly on a fine-grid node. The state is integrated past the horizon —
# by at least 1.5 yr, more when the build lag $\ell$ is long — so `cost_flow` can read
# $c^L_{t+\ell}$ for every displayed $t$; outputs are then truncated to $[0, T]$.
#
# The release-delay machinery ($x^R$, `_xR_of`, the $\tau$ sweep) is **parked and pinned at
# $\tau = 0$** (D-077, spec N9), where $x^R \equiv x^L$; it is kept intact so question (b) can be
# revived without rebuilding it.

# %%
# ---- Cell E6: RK4 integrator + full simulation ----
# The leader (a_L, c_L) is autonomous -- its dynamics depend on neither the follower nor x^R -- so
# it is integrated FIRST, on a half-step "fine" grid. The follower is then integrated on the coarse
# grid; each RK4 stage falls exactly on a fine-grid node, so the delayed leader path x^R(t)=x_L(t-tau)
# is read off by direct indexing (no interpolation) -- exact and fast.
def _rk4_leader_fine(fine, p):
    """Integrate the leader (a_L, c_L) on the fine grid (step dt/2) with fixed-step RK4."""
    n = len(fine)
    aL = np.zeros(n); cL = np.zeros(n)
    aL[0] = 0.0; cL[0] = 0.0                     # normalization x_L0 = 0
    for i in range(n - 1):
        h = fine[i+1] - fine[i]
        t = fine[i]; a = aL[i]; c = cL[i]
        k1a = algo_growth_L(t, a + c, p);             k1c = compute_growth(t, p)
        k2a = algo_growth_L(t+h/2, a+h/2*k1a + c+h/2*k1c, p); k2c = compute_growth(t+h/2, p)
        k3a = algo_growth_L(t+h/2, a+h/2*k2a + c+h/2*k2c, p); k3c = compute_growth(t+h/2, p)
        k4a = algo_growth_L(t+h,   a+h*k3a   + c+h*k3c,   p); k4c = compute_growth(t+h, p)
        aL[i+1] = a + h/6*(k1a + 2*k2a + 2*k3a + k4a)
        cL[i+1] = c + h/6*(k1c + 2*k2c + 2*k3c + k4c)
    return aL, cL

def _xR_of(s, grid, xL, p):
    """Released capability x^R at time argument s = t - tau. History lookup on the grid, with the
    design's pre-period backward extrapolation x_L(0) + (t-tau)(g_C0 + g_a) for s < 0."""
    s = np.asarray(s, dtype=float)
    pre = 0.0 + s * (p.g_C0 + p.g_a)             # x_L0 = 0; pre-2026 trend ~ today's rates
    interp = np.interp(np.clip(s, grid[0], grid[-1]), grid, xL)
    return np.where(s < 0.0, pre, interp)

def _rk4_follower_fine(coarse, aL_fine, xR_fine, gCF_fine, p):
    """Integrate the follower (a_F, c_F) on the coarse grid. Stage inputs (leader algo aL, released
    capability xR, follower compute growth gCF) are read by index from the fine grid: coarse node i
    is fine node 2i; the half-step stages use fine node 2i+1; the full step uses 2i+2."""
    n = len(coarse)
    aF = np.zeros(n); cF = np.zeros(n)
    aF[0] = 0.0 - p.split * p.Delta0             # a_F0 : algo part of the initial gap
    cF[0] = 0.0 - (1.0 - p.split) * p.Delta0     # c_F0 : compute part of the initial gap
    for i in range(n - 1):
        h = coarse[i+1] - coarse[i]
        j = 2*i
        a = aF[i]; c = cF[i]
        k1a = algo_growth_F(aL_fine[j],   a,          xR_fine[j],   a + c,          p); k1c = gCF_fine[j]
        k2a = algo_growth_F(aL_fine[j+1], a+h/2*k1a, xR_fine[j+1], a+h/2*k1a + c+h/2*k1c, p); k2c = gCF_fine[j+1]
        k3a = algo_growth_F(aL_fine[j+1], a+h/2*k2a, xR_fine[j+1], a+h/2*k2a + c+h/2*k2c, p); k3c = gCF_fine[j+1]
        k4a = algo_growth_F(aL_fine[j+2], a+h*k3a,   xR_fine[j+2], a+h*k3a   + c+h*k3c,   p); k4c = gCF_fine[j+2]
        aF[i+1] = a + h/6*(k1a + 2*k2a + 2*k3a + k4a)
        cF[i+1] = c + h/6*(k1c + 2*k2c + 2*k3c + k4c)
    return aF, cF

def simulate(p, tau_fn=None):
    """Integrate the full system and return time series on [0, T]. Spec I.0-I.5.
    Internally integrates PAST the horizon so cost_flow can read c_L(t+ell) for every displayed
    t <= T; outputs truncated to T."""
    # The pad must COVER the training lead ell (slider allows up to 3 yr): with a fixed pad the
    # c_L(t+ell) interpolation silently clamped at the grid end, freezing the compute factor while
    # 10^{-g_p t} kept falling -> a spurious cost peak/decline kink at t = T - (ell - pad)
    # (Pavel's bug report 2026-07-23, horizon 5 / ell 3 -> kink at 3.5). 1.5 stays as the floor.
    pad = max(1.5, float(p.ell) + 2.0 * float(p.dt))
    # Grids are built from an INTEGER step count, not from arange endpoints, so the invariant the
    # follower integrator relies on -- len(fine) == 2*len(grid)-1, with fine node 2i EXACTLY on
    # coarse node i -- holds for every (T, dt, ell). (Bug found 2026-07-27: with arange endpoints,
    # an ell whose T+pad+dt/2 landed just above a multiple of dt gave the coarse grid one extra
    # node, the `fine[:2n-1]` slice became a no-op, and _rk4_follower_fine ran off the end. It was
    # masked only because every ell ever drawn or dialled was a round number.)
    n_steps = int(np.ceil((float(p.T) + pad) / float(p.dt)))
    grid = float(p.dt) * np.arange(n_steps + 1)              # coarse grid, step dt
    fine = 0.5 * float(p.dt) * np.arange(2 * n_steps + 1)    # fine grid, step dt/2
    aL_f, cL_f = _rk4_leader_fine(fine, p)
    xL_f = aL_f + cL_f
    # tau may be the constant p.tau, or a time-varying schedule tau_fn(t) passed as a callable
    # (e.g. a delay that switches on for a window): released path is x^R(t) = x_L(t - tau(t)).
    tau_arr = np.asarray(tau_fn(fine), dtype=float) if callable(tau_fn) else p.tau
    xR_f = _xR_of(fine - tau_arr, fine, xL_f, p)             # delayed leader path on the fine grid
    gCF_f = follower_compute_growth(fine, p)
    aF, cF = _rk4_follower_fine(grid, aL_f, xR_f, gCF_f, p)
    # sample leader arrays back onto the coarse grid
    aL = aL_f[::2]; cL = cL_f[::2]; xL = xL_f[::2]; xR = xR_f[::2]
    xF = aF + cF
    Delta = xL - xF

    # value INDEX on the leader's and the follower's capability (D-077: the model runs on the
    # DEVELOPED frontier x^L; x^R is retained, pinned equal to x^L, only for the parked
    # release-delay extension, so W_R below is W(x^L) whenever tau = 0)
    W_R = W(xR, p)
    W_F = W(xF, p)
    served_gap = xR - xF
    _cm = conduct_mult(served_gap, p)            # II.4: dimensionless, == 1 when off
    cond = np.broadcast_to(np.asarray(_cm, dtype=float), grid.shape).astype(float)
    gapW = np.maximum(W_R - W_F, 0.0)            # N2: earnings floor at zero if W_R < W_F
    # EARNINGS before model-building costs, IN MULTIPLES OF TODAY'S BILL (D-093). The whole
    # earnings leg is normalised by its own t = 0 value, so E_0 = rho identically -- no solve,
    # no stored coefficient. (The dict key stays `revenue`: renaming it to `earnings` is the
    # equation review's separate S-item, parked since D-077, and the MC's stored draw records
    # key on this name -- renaming it would silently break reloading a saved snapshot.)
    coef = p.rho / gap_index(p)                  # = rho / [W(x^L_0) - W(x^F_0)]
    revenue = coef * cond * gapW
    revenue = revenue * (1.0 - 0.4 * p.chi)      # II.5: defense costs 0.4*chi of earnings

    # cost: c_L(t+ell) via interpolation; c_L(ell) constant
    cL_ell = float(np.interp(p.ell, grid, cL))
    cL_at = np.interp(grid + p.ell, grid, cL)    # t+ell within padded grid for t<=T
    cost = cost_flow(grid, cL_at, cL_ell, p)

    profit = revenue - cost
    if p.labor_line:                             # II.7 decoupled labor line
        profit = profit - p.L0 * np.exp(p.g_w * grid)

    # diagnostics
    psi_share = np.asarray(psi_boost_share(grid, xL, p), dtype=float)
    cap_xF_le_xR = xF <= xR + 1e-9               # slack good
    cap_W = W_R >= W_F - 1e-9

    # truncate to [0, T]
    m = grid <= p.T + 1e-9
    t = grid[m]
    disc = np.exp(-p.r * t)
    npv_integrand = disc * profit[m]
    npv_cum = np.concatenate([[0.0], np.cumsum((npv_integrand[1:] + npv_integrand[:-1]) / 2.0 * np.diff(t))])
    # undiscounted running total of profit (integral of Pi); the widget shows this, not NPV
    prof_m = profit[m]
    cum_profit = np.concatenate([[0.0], np.cumsum((prof_m[1:] + prof_m[:-1]) / 2.0 * np.diff(t))])

    out = dict(
        t=t, a_L=aL[m], c_L=cL[m], x_L=xL[m], a_F=aF[m], c_F=cF[m], x_F=xF[m], x_R=xR[m],
        Delta=Delta[m], W_R=W_R[m], W_F=W_F[m], conduct=cond[m],
        revenue=revenue[m], cost=cost[m], profit=profit[m], npv_cum=npv_cum, cum_profit=cum_profit,
        psi_share=psi_share[m], cap_xF_le_xR=cap_xF_le_xR[m], cap_W=cap_W[m],
    )
    return out

# %% [markdown]
# ## Outputs — question (a)
#
# Does the leader's profit flow turn positive within ~5 yr, and does it *stay* positive? The
# **reported** outcome in the widget is the coverage ratio $\rho_t = E_t / B_t$ (earnings over
# model-building cost), which is break-even at 1 and, given $\rho_0 = m/k$, invariant to $R_0$
# and $m$ separately — the money triple's only identified combination (D-080). NPV is a secondary
# statistic. `delay_comparison` sweeps the parked release delay $\tau$ (D-077).

# %%
# ---- Cell E7: headline statistics (question a) and delay comparison (question b) ----
def headline(sim, p):
    """Question (a) outputs. Spec I.5a.
    first sign crossing (yr), positive-within-5yr, stays-positive, NPV, terminal gap."""
    t = sim['t']; profit = sim['profit']
    pos = profit >= 0.0
    # first sign crossing: earliest t at which profit is non-negative
    idx = np.argmax(pos) if pos.any() else -1
    sign_crossing_year = float(t[idx]) if idx >= 0 else float('nan')
    within5 = bool(pos[t <= 5.0 + 1e-9].any())
    # stays positive from the crossing to T
    stays_positive = bool(idx >= 0 and pos[idx:].all())
    npv = float(sim['npv_cum'][-1])
    terminal_gap = float(sim['Delta'][-1])
    # cross-consistency: t = 0 earnings, which D-093 makes an IDENTITY -- E_0 = rho exactly,
    # so this reads back the coverage dial rather than a solved dollar figure (cost_t0 = 1.0
    # likewise). Both are in multiples of today's bill; neither is a dollar amount.
    op_profit_t0 = float(sim['revenue'][0])
    return dict(
        sign_crossing_year=sign_crossing_year,
        positive_within_5yr=within5,
        stays_positive=stays_positive,
        npv=npv,
        terminal_gap=terminal_gap,
        op_profit_t0=op_profit_t0,
        cost_t0=float(sim['cost'][0]),
        psi_share_max=float(np.nanmax(sim['psi_share'])),
        cap_ok=bool(sim['cap_xF_le_xR'].all() and sim['cap_W'].all()),
    )

def delay_comparison(p, taus=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0)):
    """Question (b). Spec I.5b. Pi_t paths and NPV as a function of release delay tau (yr)."""
    res = {'taus': list(taus), 'paths': [], 'npvs': [], 't': None}
    for tau in taus:
        s = simulate(replace(p, tau=tau))
        if res['t'] is None:
            res['t'] = s['t']
        res['paths'].append(s['profit'])
        res['npvs'].append(float(s['npv_cum'][-1]))
    res['npvs'] = np.array(res['npvs'])
    best_i = int(np.argmax(res['npvs']))
    res['best_tau'] = taus[best_i]
    res['best_npv'] = float(res['npvs'][best_i])
    return res

# %% [markdown]
# ## Targets (D-037) — observables in natural units
#
# Targets-first parameterization: wherever a parameter has a clean observable, the *observable* is
# the primitive. Slider bounds, Monte-Carlo distributions and the calibration documentation all
# live in target space (`TARGET_RANGES`, one source of truth), and `invert_targets` maps them back
# into model parameters at $t = 0$. The defaults are the exact forward images of `Params()`, so
# the default targets invert back to the parameter defaults precisely.

# %%
# ---- Cell E9: targets (D-037) -- observable targets: ranges, forward map, inversions ----
# One source of truth per target: TARGET_RANGES drives the widget slider bounds, the Monte-Carlo
# distribution, and the calibration cards. Defaults are the FORWARD images of Params() defaults
# (stored exact, displayed rounded), so the default targets invert exactly back to Params().
#
# D-076 (base-model calibration sync, 2026-07-27): the target set changed.
#   * t_algo_x  -> t_eff_x    : the observable is EFFECTIVE-compute growth (data, architecture and
#                               post-training know-how included); g_a = log10(t_eff_x) - g_C0 is a
#                               RESIDUAL, so the RL-compute double-count is impossible (Pavel).
#   * t_bill_x  -> t_price_x  : the trusted leg is hardware price-performance (TC2b). Bill growth
#                               10^(g_C0-g_p) becomes a READ-OUT, not a dial.
#   * t_profit_B removed      : the money side was anchored on (R0, m_margin, k_build) directly
#                               (SB-R Option 2), with kappa and B0 derived by money_anchors.
#                               D-093 went further: normalising both legs at t = 0 left ONE
#                               finance parameter, rho, which needs no target and no inversion --
#                               it IS the observable (coverage today), dialled directly.

TARGET_PARAM = {          # target key -> the parameter it pins (t_lag_mo also sets the deltas)
    't_compute_x': 'g_C0',    # compute scaling today, x/yr           g_C0 = log10(.)
    't_eff_x':     'g_a',     # EFFECTIVE-compute growth today, x/yr  g_a  = log10(.) - g_C0
    't_lag_mo':    'Delta0',  # follower (fringe) lag, months         Delta0 = lag_yr * leader speed
    't_price_x':   'g_p',     # compute price-performance, x/yr       g_p  = log10(.)
    't_value_x':   'nu',      # value multiplier per OOM, x           nu   = log10(.)
    't_value_inf_x': 'nu_inf',  # ASYMPTOTIC value per OOM, x (D-083)   nu_inf = log10(.)
    't_floor_x':   'g_C_inf', # long-run compute-growth floor, x/yr   g_C_inf = log10(.)
    # D-098. The one target NOT named t_*: it is a PERCENT, not a "times" multiplier, and the
    # t_..._x / t_..._mo convention says so in the name. The prefix is convention, never a code
    # path -- every consumer keys off TARGET_RANGES membership, not on the string.
    'loss_half_gC': 'alpha',  # % of algo progress lost if g_c growth halves  alpha = f(loss, eta)
}

# D-093: `gap_index` MOVED UP to the value cell (E4) -- simulate calls it now, so it belongs
# beside W rather than among the calibration helpers.
#
# `money_anchors` and `with_money` are GONE. They existed to solve the two dollar constants
# (kappa, B0) so that the forward path reproduced the reported $40B earned and $75B spent at
# t = 0. Normalising both legs by their own t = 0 values makes that solve unnecessary rather
# than cheaper: E_0 = rho and B_0 = 1 hold as identities, at every level and every dial, so
# there is no anchor left to re-derive and no way for a Params to be "money-incoherent".
# Every `with_money(p)` call site simply dropped the wrapper.

# D-088: `plateau_from_today` is GONE. It solved identity 2 at this one call site (D-086); the
# identity now lives inside gamma_shape, where every use gets it. Nothing outside model.py ever
# called it -- ui/sidebar.py already seeds d["g_C0"] with today's dialled growth and lets
# invert_targets finish the job -- so there is no shim. Its `tied` flag went with it: the dial
# is today's growth whether or not the floor is tied to it, so the transition-off context needs
# no special case at all.

def xdot_L0(p):
    """The leader's EXACT capability speed at t = 0, from the model's own rate functions:
    compute_growth(0) = Gamma(0) is today's realised compute growth (p0_c% of the way into
    the slowdown), and algo_growth_L(0, 0) is the engine's t = 0 state -- both input ratios
    are exactly 1 there, so it returns g_a (D-086 P1-2). Hence xdot_L0 = Gamma(0) + g_a, and
    the effective-growth dial means TODAY at every p0_c.

    (Defined ahead of target_defaults, which calls it at import time through _TD0.)"""
    return float(gc_today(p) + algo_growth_L(0.0, 0.0, p))

def target_defaults(p=None):
    """Forward map Params -> target values (exact; round-trips through invert_targets).

    D-086 P1-2: the three t = 0 observables are read off the model's OWN t = 0 rates --
    compute_growth(0, p) = Gamma(0) and xdot_L0(p) -- not off the pre-transition plateau
    g_C0 + g_a. Before D-086 a p0_c of 25% made the dial reading "3.24x/yr today" deliver
    2.60x/yr and "11.34x/yr today" deliver 8.11x/yr; now every dial means what it says at
    every p0_c."""
    p = p if p is not None else Params()
    gc0_today = gc_today(p)
    speed0 = xdot_L0(p)
    return {
        't_compute_x': float(10.0**gc0_today),
        't_eff_x':     float(10.0**speed0),
        't_lag_mo':    float(p.Delta0 / speed0 * 12.0),
        't_price_x':   float(10.0**p.g_p),
        't_value_x':   float(10.0**p.nu),
        't_value_inf_x': float(10.0**p.nu_inf),
        't_floor_x':   float(10.0**p.g_C_inf),
        # D-098, in PERCENT (the p0_c convention: a percent dial states percent). Under
        # Leontief this reports the 50% the model actually delivers, not the weight's image.
        'loss_half_gC': float(100.0 * loss_from_alpha(p.alpha, p.eta, p.leontief)),
    }

_TD0 = target_defaults()
# D-042 two-tier ranges: TARGET_RANGES is the ENVELOPE -- the outer bounds of what a user may
# sample; the DEFAULT simulation range is the tight documented span (SIM_DEFAULT, cell E8b).
# ENVELOPE RULE (Pavel, 2026-07-26): the envelope must contain the UNION of the confidence ranges
# of the sources the menu includes, with the left bound rounded down and the right rounded up.
# Rows labelled "different object" / "display only" do not enter the union.
TARGET_RANGES = {
    # union 2.5 (Pilz supercomputers) .. 5.3 (full-window) -> rounded out. BASE MC stays [3, 4].
    't_compute_x': ('uniform', 2.5, 6.0),                      # x/yr
    # union: Ho-2024's own 95% CI on g_a, in t_eff units [5.8, 20.6], plus the Rosetta row (19).
    # The "test-time included" (29) and whole-stack (32) rows are bound/wrong-object labels, not
    # estimates, so they do not set the envelope (Pavel, 2026-07-27: narrower than the audit's 30).
    't_eff_x':     ('lognormal', float(np.log(np.sqrt(5.0 * 21.0))),
                    float(np.log(21.0 / 5.0) / 3.29)),         # ~90% CI [5, 21] x/yr
    # union of the three constructions' 90% bands [1.62, 2.84] -> [1.5, 3.0] (TV5' ratified).
    # Triangular (not uniform) so a user narrowing the range keeps the pooled mode.
    't_value_x':   ('triangular', 1.5, 2.1, 3.0),              # x/OOM
    # D-083: the ASYMPTOTIC value slope's envelope — a PROPOSAL flagged for the
    # calibration round (no documented sources yet): from x1 (full commoditization,
    # the retired ceiling's spirit) to x2 (still below today's x2.1). Default x1.25.
    't_value_inf_x': ('uniform', 1.0, 2.0),                    # x/OOM, asymptotic
    # rows 1-7 span 4 .. 10; the ratified MC CI is [4, 12] -> the envelope IS [4, 12].
    't_lag_mo':    ('lognormal', float(np.log(np.sqrt(4.0 * 12.0))),
                    float(np.log(12.0 / 4.0) / 3.29)),         # ~90% CI [4, 12] months
    # Hobbhahn's own CI [1.27, 1.54] rounded out; the model's spot 1.38 sits inside it.
    't_price_x':   ('uniform', 1.25, 1.55),                    # x/yr
    't_floor_x':   ('uniform', 10.0**0.05, 10.0**0.30),        # EXTENSION (slowdown) -- untouched
    # D-098, PERCENT. NOT a chosen box: this IS the union of the adoptable alpha rows' intervals
    # ([23.5, 44.5] -- Cottier's low end to Gundlach's scale-dependence) with the left bound
    # rounded DOWN and the right rounded UP, exactly per Pavel's envelope rule (2026-07-26).
    # display_only rows (the Anthropic residual bound, AI-2027's different-construct elasticity)
    # do not enter the union. test_09 asserts both that it contains the union and that it is
    # TIGHT, so the derivation cannot silently rot into an arbitrary box.
    'loss_half_gC': ('uniform', 22.0, 45.0),                   # %
}

def channels_from_lag(lag_yr, speed, own_speed):
    """Wedge-split at the channels levels (L6+): rescale the jointly calibrated channel defaults
    (delta_dev, delta_rel) = (0.20, 0.26) so the stated lag is stationary in the same sense at the
    CURRENT speeds: rescale = (wedge/Delta0) / (wedge0/Delta0_0), with wedge = leader speed minus
    the follower's own speed (both at t=0) and Delta0 = lag_yr*speed. rescale = 1 at the default
    lag and speeds, so the notebook defaults are recovered exactly (D-037). (Named `kappa` until
    D-093 -- an unrelated object that shared the retired dollar coefficient's letter.)

    Extensions-sync round (2026-07-28, audit X-04): under Pavel's refined re-anchor rule this
    function supplies the channel DIRECTION only -- stationary_catchup keeps the calibrated
    delta_dev:delta_rel ratio and re-solves the LENGTH against the exact t = 0 transfer
    identity (which is where `split` enters), so the old TODO about ignoring `split` is
    resolved there, not here. Remaining direct consumers use it for doc-level numbers only
    (ui/content._live_vals). The merged levels (1-2) never call it."""
    p0 = Params()
    Delta0 = lag_yr * speed
    wedge = max(speed - own_speed, 0.0)
    wedge0 = (p0.g_C0 + p0.g_a) - (p0.g_a_F + p0.g_CF0)
    rescale = (wedge / max(Delta0, 1e-9)) / (wedge0 / p0.Delta0)
    return float(p0.delta_dev * rescale), float(p0.delta_rel * rescale)

def stationary_catchup(p, merged=True):
    """Pavel's refined re-anchor rule (D-081 addendum, 2026-07-27): EVERY dial configuration
    must reproduce, at t = 0, (a) the observed gap Delta0 AND (b) gap stationarity,
    Delta_dot(0) = 0 exactly. The degree of freedom that absorbs this is the follower's
    catch-up intensity -- the generalisation of the base construction
    delta = (g_c + g_a)/Delta0:

        merged:   delta = xdot_L0(p) / Delta0            (routes through delta_rel, D-034)
        channels: delta_dev*split*Delta0 + delta_rel*Delta0
                      = xdot_L0(p) - g_a_F - g_cF(0)     (the follower's t = 0 transfer gap)

    with the exact t = 0 rates on BOTH sides. The channel DIRECTION comes from
    channels_from_lag's rescaled calibrated defaults; only its LENGTH is re-solved, so the
    split redistributes within the derived total. Consequence: dials shape the trajectory
    only FORWARD of t = 0. BOTH documented residuals here are now gone: D-090 killed S0's
    movement with the cost-SHAPE dials (ell, t_mid, g_c_inf), which was pure parameterisation
    artefact, and D-093 killed kappa's movement with x_mid -- not by pinning it but by deleting
    it. There is no derived finance constant left to drift, because E_0 = rho and B_0 = 1 are
    identities rather than solutions.
    Negative-wedge corner (the follower's own t = 0 speed exceeds the leader's): stationarity
    is unsatisfiable with non-negative coefficients; returns (0, 0), the documented clamp --
    flagged, not hidden (D-081 addendum)."""
    x0 = xdot_L0(p)
    if merged:
        return split_delta(x0 / p.Delta0)
    own0 = float(p.g_a_F + follower_compute_growth(0.0, p))
    wedge = x0 - own0
    if wedge <= 0.0:
        return 0.0, 0.0                       # unsatisfiable corner -- documented clamp
    # DIRECTION only: channels_from_lag scales delta_dev and delta_rel by a COMMON factor, and
    # the `s` below renormalises the pair, so this call's arguments cancel out of the result to
    # floating-point rounding (measured worst deviation 2.7e-16; asserted in test_24). They are
    # nevertheless stated at the exact t = 0 rates for the same reason as everything else in
    # D-086 P1-2 -- the plateau readings g_C0 + g_a and g_a_F + g_CF0 are not the model's t = 0
    # speeds once a position dial is off zero.
    d0, r0 = channels_from_lag(p.Delta0 / x0, x0, own0)
    denom = d0 * p.split * p.Delta0 + r0 * p.Delta0
    if denom <= 0.0:
        return 0.0, float(wedge / p.Delta0)   # degenerate direction: all through distillation
    s = wedge / denom
    return float(d0 * s), float(r0 * s)


def invert_targets(targets, base, merged=True):
    """Invert a dict of target values into a dict of model parameters (t=0 inversion, D-037 Q2).

    `base` supplies every parameter no target pins (split, follower engine, x_mid, ...);
    inversions see already-inverted values (cross-target coupling by design: the effective-compute
    target needs the inverted g_C0 to take its residual; the lag inversion uses the resulting
    speed). NO MONEY STEP (D-093): there are no derived dollar constants left to re-anchor, so
    every Params is money-coherent by construction rather than by a closing fix-up.
    merged=True (pure-catch-up levels 1-2, D-081): the lag pins Delta0; the merged delta
    re-derives per configuration as xdot_L0/Delta0 (= 12/lag at the base pins), so the gap
    is STATIC at t = 0 for any dial setting. merged=False (channels, level 3): Delta0 from
    the lag; (delta_dev, delta_rel) from stationary_catchup (same exactness)."""
    out = {}
    if 't_floor_x' in targets:
        out['g_C_inf'] = float(np.log10(targets['t_floor_x']))
    if 't_compute_x' in targets:
        # D-088: the dial IS the parameter. g_C0 means g_C(0) -- today's growth -- and Gamma
        # derives the pre-slowdown plateau from it, so this inversion is the identity map.
        # (Between D-086 and D-088 this line solved the plateau here, which made it depend on
        # g_C_inf and p0_c and forced a `tied` special case and a load-bearing ORDERING against
        # the floor target above. All three are gone: no ordering, no floor dependence, no
        # special case, and today's growth cannot drift with any other dial by construction.)
        out['g_C0'] = float(np.log10(targets['t_compute_x']))
    if 't_eff_x' in targets:
        # RESIDUAL definition (Pavel, 2026-07-26): g_a = g_eff - g_c, both read at t = 0.
        # D-086 P1-2: the subtrahend is TODAY's realised compute growth Gamma(0), not the
        # plateau -- which is exactly what makes g_a a residual of two OBSERVABLES again.
        # Floored at 0: the model has no "algorithms get worse" regime, and the corner of
        # the joint draw where a high compute draw meets a low effective-compute draw would
        # otherwise produce one. The widget also clamps its slider (GE5).
        gc = gc_today(replace(base, **out))
        out['g_a'] = float(max(np.log10(targets['t_eff_x']) - gc, 0.0))
    if 't_value_x' in targets:
        out['nu'] = float(np.log10(targets['t_value_x']))
    if 't_value_inf_x' in targets:
        out['nu_inf'] = float(np.log10(targets['t_value_inf_x']))
    if 't_price_x' in targets:
        out['g_p'] = float(np.log10(targets['t_price_x']))
    if 'loss_half_gC' in targets:
        # D-098. ORDER-FREE, unlike every other inversion here: alpha enters no t = 0 rate
        # (both CES channels are 1 at t = 0, so adot_L(0) = g_a at every alpha), so this branch
        # neither reads nor feeds `ref`. It may sit anywhere in this function; it is placed
        # last among the scalar inversions only for reading order. It DOES read the active eta,
        # which is a free dial and never a target -- so `base` carries it.
        out['alpha'] = alpha_from_loss(float(targets['loss_half_gC']) / 100.0,
                                       base.eta, base.leontief)
    ref = replace(base, **out)
    if 't_lag_mo' in targets:
        lag_mo = float(targets['t_lag_mo'])
        lag_yr = lag_mo / 12.0
        # D-086 P1-2: the lag converts at the leader's EXACT t = 0 speed xdot_L0 = Gamma(0)
        # + g_a, so "7 months behind" means 7 months at every p0_c (it drifted to 8.1 months
        # at the top of the envelope while the plateau speed was used).
        speed = xdot_L0(ref)
        # The catch-up intensity then re-derives from the same exact t = 0 rates, so the gap
        # is also STATIC at 0 (Pavel's refined re-anchor rule, D-081 addendum -- at the base
        # pins this reduces to delta = 12/lag exactly).
        out['Delta0'] = lag_yr * speed
        ref = replace(ref, Delta0=out['Delta0'])
        out['delta_dev'], out['delta_rel'] = stationary_catchup(ref, merged=merged)
        ref = replace(ref, delta_dev=out['delta_dev'], delta_rel=out['delta_rel'])
    return out

# ---- base-model self-checks (D-076). These run at import, so an incoherent default can never
# reach the widget. The BASE = the full model with the later mechanisms pinned exactly as
# ui/levels.apply_level_pins does at Level 1.
def base_params(**kw):
    """The calibrated BASE model: constant compute growth, constant residual g_a, exponential
    value, merged catch-up, no build lag, no R&D markup on top of the observed bill. This is the
    Level-1 model the calibration round closed on 2026-07-26."""
    # (D-088: g_C0 is no longer pinned here. It IS G_C_TODAY -- the dataclass default states
    # today's growth directly, so the old explicit pin, which existed only to undo D-086's
    # plateau-valued default, has nothing left to undo.)
    pins = dict(A1=True, gamma=0.0, ell=0.0, phi_RD=0.0, x_mid=200.0, tau=0.0,
                g_a_F=0.0, g_CF0=0.0, g_CF_inf=0.0, split=0.0)
    pins.update(kw)          # explicit overrides win, so a caller may probe e.g. x_mid
    p = Params(**pins)
    p = replace(p, g_C_inf=p.g_C0)                      # constant compute growth
    if 'nu_inf' not in kw:
        p = replace(p, nu_inf=p.nu)                     # D-083: value-slope transition OFF
    p = replace(p, delta_dev=0.0,
                delta_rel=(p.g_C0 + p.g_a) / p.Delta0)  # merged delta = 12/lag_mo
    return p                 # D-093: no with_money() tail -- there is no derived money constant

_PB = base_params()
assert np.isclose(_PB.g_C0 + _PB.g_a, 1.0546130545568877, rtol=1e-12)   # speed 11.34 x/yr
assert np.isclose(_PB.Delta0, 0.6151909484915179, rtol=1e-12)           # 7.0-month lag
assert np.isclose(gap_index(_PB), 0.3664606328362475, rtol=1e-12)       # the value gap, in index units
# D-093: the two stale-literal guards on kappa and B0 are GONE WITH THE FIELDS. Nothing here
# needs re-fitting when a default moves -- E_0 = rho and B_0 = 1 are identities, and the two
# asserts below check them on the model's OWN forward path rather than on a stored constant.
_SB = simulate(replace(_PB, T=1.0))
assert np.isclose(_SB['revenue'][0], _PB.rho, rtol=1e-12)               # E_0 = rho, 53.3% coverage
assert np.isclose(_SB['cost'][0], 1.0, rtol=1e-12)                      # B_0 = 1 (today's bill)

# %% [markdown]
# ## Monte Carlo — joint draws from the calibration ranges
#
# `sample_params` draws from `PARAM_RANGES` **jointly**, independently per parameter, and
# `monte_carlo` runs `simulate` on each draw. Draws whose implied bill growth falls outside
# `BILL_COHERENCE` are rejected and redrawn: the compute-growth and price-decline draws jointly
# imply a bill-growth rate, and an incoherent pair would be a scenario no observer has seen.

# %%
# ---- Cell E8: parameter ranges, joint sampling, Monte Carlo ----
# Distributions per design section 5. Each entry: (kind, *args).
#   uniform:    ('uniform', lo, hi)
#   triangular: ('triangular', lo, mode, hi)
#   lognormal:  ('lognormal', mu, sigma)     [drawn as exp(Normal(mu, sigma))]
#   scale_of:   ('scale_of', other_name, lo, hi)   [param = draw(other) * U(lo,hi)]
#   choice:     ('choice', [values...])
#
# D-076: PARAMETER-space ranges are the ENVELOPE for free dials and the legacy whole-parameter
# Monte Carlo. Everything with an observable is drawn in TARGET space instead (TARGET_RANGES) --
# "draw observables, not parameters". Four dimensions left this dict entirely:
#   phi_RD         -- inert under bill anchoring (it cancels exactly inside cost_flow),
#   kappa, B0      -- retired outright by D-093 (see the money block in Params).
# The money side draws ONE dimension and always did: the LEVEL is provably irrelevant (scaling
# earnings and cost by any lambda leaves the crossing year and the verdict exactly unchanged),
# so only the cost-to-earnings ratio carries uncertainty (SB6). Since D-093 that dimension is
# rho ITSELF rather than k_build standing in for it. Its entry below is in FRACTION units,
# because this dict is in parameter units; the WIDGET dials and crops the same dimension in
# percent through an app-side overlay (ui/state.py APP_RANGES['cov0']) and ui/mc.mc_prepare
# converts the crop. Two representations of one envelope, never two dimensions.
PARAM_RANGES = {
    'g_C0':     ('uniform', 0.39794, 0.778151),  # image of t_compute_x envelope [2.5, 6.0] x/yr
    'g_C_inf':  ('uniform', 0.05, 0.30),
    't_mid':    ('uniform', 0.7, 7.0),  # D-082: image of the old xi in [0.1, 1.0] under
                                        # the half-decay map t_mid = ln2/xi, rounded out
    # D-084 POSITION dials (PERCENT of the transition already done at u = 0), one per use of the
    # universal curve. The envelopes below are PROPOSALS, FLAGGED for the calibration round: from
    # 1% (the convention D-082 baked in -- "the transition has barely started") to 25% (visibly
    # under way). 50% is excluded BY CONSTRUCTION: it would put the midpoint at u = 0, and more
    # would put it in the past, contradicting what the midpoint dial means. All three are POINT
    # defaults in the MC (deliberately absent from SIM_DEFAULT), so the default fans are
    # untouched until a calibration pass widens them -- exactly how t_value_inf_x entered.
    'p0_c':     ('uniform', 1.0, 25.0),
    'p0_w':     ('uniform', 1.0, 25.0),
    'p0_F':     ('uniform', 1.0, 25.0),
    'g_a':      ('uniform', 0.0, 0.9),           # residual g_eff - g_C0; floored at 0 (see
                                                 # invert_targets). Widget draws t_eff_x instead.
    # D-098: RE-WIRED, not deleted. This is the image of the ratified alpha range [0.45, 0.90]
    # at the base eta = 1, and it is the envelope for the LEGACY whole-parameter path only
    # (`sample_params`, which iterates all of PARAM_RANGES). The widget never reads it: it draws
    # the OBSERVABLE loss_half_gC from TARGET_RANGES and derives alpha per draw at that draw's
    # eta. Deleting the row would have silently pinned alpha at p_base in the legacy path, which
    # is worse than a slightly redundant envelope -- the same reason g_C0 keeps a parameter-space
    # range beside its target.
    'alpha':    ('uniform', 0.45, 0.90),
    'eta':      ('choice', [1.0, 0.61, 0.0, -2.0]),
    'beta0':    ('uniform', 0.1, 0.5),   # D-084: renamed from rho0 (rho = coverage)
    'gamma':    ('uniform', 0.0, 0.17371779276130073),  # = 0.4/ln10 (D-091 base-10 rescale);
                                        # capped: above ~0.182 it blows up inside horizon (N4)
    'g_a_F':    ('scale_of', 'g_a', 0.5, 0.9),  # ENVELOPE around the Gundlach 0.6-0.8x
                                # follower/leader band; the DIAL is this share (audit X-10),
                                # so dial, trim crop and prior are one object in share units
    'g_CF0':    ('uniform', 0.3, 0.7),
    'g_CF_inf': ('uniform', 0.05, 0.20),
    't_mid_F':  ('uniform', 0.7, 7.0),  # D-082: same half-decay map as t_mid
    'Delta0':   ('lognormal', np.log(0.615), 0.334),  # image of the [4, 12]-month lag envelope
    'split':    ('uniform', 0.3, 0.8),
    'delta_dev':('uniform', 0.08, 0.40),
    'delta_rel':('uniform', 0.12, 0.75),
    'nu':       ('triangular', 0.176091, 0.322219, 0.477121),  # image of t_value_x [1.5, 2.1, 3.0]
    'rho':      ('uniform', 0.33, 0.56),  # coverage today, the ONE money dimension (D-093). The
                                          # vetted [33, 56]% envelope in fraction units: it spans
                                          # BOTH documented FIN4 bases (run-rate restatement
                                          # ~0.42, calendar 0.533). Deliberately absent from
                                          # SIM_DEFAULT -- the widget's tight band is the app-side
                                          # APP_SIM_DEFAULT, and the two are already the same
                                          # union, so a second copy here could only drift.
    'x_mid':    ('uniform', 2.0, 20.0),
    'ell':      ('lognormal', float(np.log(np.sqrt(0.25 * 1.3))),
                 float(np.log(1.3 / 0.25) / 3.29)),   # EL3': ~90% CI [0.25, 1.3] yr
    'g_p':      ('uniform', 0.09691, 0.190332),  # image of t_price_x envelope [1.25, 1.55] x/yr
    'r':        ('uniform', 0.03, 0.15),
}

# Coherence constraint (TC6 rider, ratified in principle). g_C0 and g_p are drawn independently,
# but their DIFFERENCE is itself observed: the training bill grows 10^(g_C0 - g_p) x/yr, which
# Cottier et al. 2024 measure at 2.4x/yr with a 90% CI [2.0, 2.9]. Corner draws outside that band
# imply hardware getting more expensive, or improving twice as fast as any observed series, so
# they are REJECTED and redrawn -- and the rejection count is reported, never silently truncated.
BILL_COHERENCE = (2.0, 2.9)
_COHERENCE_TRIES = 50

def bill_growth(p):
    """Implied training-bill growth today, x/yr = 10^(g_C0 - g_p). A read-out, not a dial: the
    base calibration trusts the compute leg (3.24) and the hardware-price leg (1.38) and lets the
    bill fall where it falls (2.35 vs Cottier's observed 2.4 -- a 2% miss, documented not fitted)."""
    return float(10.0**(p.g_C0 - p.g_p))

def bill_coherent(p, band=BILL_COHERENCE):
    """Does this draw's implied bill growth sit inside the observed band?"""
    return bool(band[0] <= bill_growth(p) <= band[1])

def _draw_one(kind_args, rng, drawn):
    kind = kind_args[0]
    if kind == 'uniform':
        return float(rng.uniform(kind_args[1], kind_args[2]))
    if kind == 'triangular':
        return float(rng.triangular(kind_args[1], kind_args[2], kind_args[3]))
    if kind == 'lognormal':
        return float(np.exp(rng.normal(kind_args[1], kind_args[2])))
    if kind == 'choice':
        return kind_args[1][int(rng.integers(len(kind_args[1])))]
    if kind == 'scale_of':
        base = drawn[kind_args[1]]
        return float(base * rng.uniform(kind_args[2], kind_args[3]))
    raise ValueError(kind)

def sample_params(rng, p_base):
    """Draw one Params jointly from PARAM_RANGES (independent per parameter, with the g_a_F scale
    coupling). Extension dials and scenario fields are inherited from p_base. No money fix-up
    since D-093: E_0 = rho and B_0 = 1 hold for every draw by construction, and rho itself is
    drawn app-side (ui/mc.py maps the coverage crop straight onto this field)."""
    drawn = {}
    # resolve non-coupled first so scale_of can reference them
    order = [k for k in PARAM_RANGES if PARAM_RANGES[k][0] != 'scale_of'] + \
            [k for k in PARAM_RANGES if PARAM_RANGES[k][0] == 'scale_of']
    for k in order:
        drawn[k] = _draw_one(PARAM_RANGES[k], rng, drawn)
    return replace(p_base, **drawn)

def monte_carlo(n, p_base, seed=0, taus=(0.0, 0.5, 1.0), n_delay=60):
    """Joint-draw forecast (design section 3/5). Runs simulate per draw; returns headline-stat
    distributions plus a subsample (~50) of Pi_t paths for the fan chart. The NPV-vs-tau
    distribution for question (b) is computed on the first `n_delay` draws only (each extra tau is
    an extra simulate), which keeps the headline stats over all n draws affordable."""
    rng = np.random.default_rng(seed)
    npvs = np.empty(n); crossings = np.empty(n); within5 = np.empty(n, dtype=bool)
    stays = np.empty(n, dtype=bool); term_gap = np.empty(n); cum_profits = np.empty(n)
    blowup = np.empty(n, dtype=bool)   # leader path passes +25 OOM inside horizon (N4 singularity)
    tgrid = None
    paths = []
    n_paths = min(50, n)
    n_delay = min(n_delay, n)
    delay_npvs = np.full((n_delay, len(taus)), np.nan)
    for i in range(n):
        p = sample_params(rng, p_base)
        s = simulate(p)                                # base release rule (p_base.tau)
        h = headline(s, p)
        if tgrid is None:
            tgrid = s['t']
        npvs[i] = h['npv']; crossings[i] = h['sign_crossing_year']
        within5[i] = h['positive_within_5yr']; stays[i] = h['stays_positive']
        term_gap[i] = h['terminal_gap']; cum_profits[i] = float(s['cum_profit'][-1])
        blowup[i] = bool(np.nanmax(s['x_L']) > 25.0)
        if i < n_paths:
            paths.append(s['profit'])
        if i < n_delay:
            for j, tau in enumerate(taus):
                sj = s if tau == p.tau else simulate(replace(p, tau=tau))
                delay_npvs[i, j] = float(sj['npv_cum'][-1])
    return dict(
        t=tgrid, npvs=npvs, crossings=crossings, within5=within5, stays=stays,
        terminal_gap=term_gap, paths=np.array(paths), n=n, cum_profits=cum_profits,
        blowup=blowup, blowup_frac=float(blowup.mean()),
        p_npv_positive_sane=float((npvs[~blowup] > 0).mean()) if (~blowup).any() else float('nan'),
        p_profitable_within_5yr=float(within5.mean()),
        p_npv_positive=float((npvs > 0).mean()),
        p_profitable=float(np.isfinite(crossings).mean()),
        median_crossing=float(np.nanmedian(crossings)) if np.isfinite(crossings).any() else float('nan'),
        p_cumprofit_positive=float((cum_profits > 0).mean()),
        p_cumprofit_positive_sane=float((cum_profits[~blowup] > 0).mean()) if (~blowup).any() else float('nan'),
        taus=list(taus), delay_npvs=delay_npvs,
        delay_helps_frac=float((delay_npvs.argmax(axis=1) > 0).mean()),
    )


def _draw_dict(rng):
    """One joint draw from PARAM_RANGES as a plain dict (same logic as sample_params)."""
    drawn = {}
    order = [k for k in PARAM_RANGES if PARAM_RANGES[k][0] != 'scale_of'] + \
            [k for k in PARAM_RANGES if PARAM_RANGES[k][0] == 'scale_of']
    for k in order:
        drawn[k] = _draw_one(PARAM_RANGES[k], rng, drawn)
    return drawn


# merged catch-up delta (base-model levels 1-5). At these levels the follower has NO engine of its
# own (g_a_F, follower compute pinned to 0), so ALL of its motion is catch-up: the single effective
# rate delta must supply the leader's FULL speed. split_delta routes it entirely through delta_rel,
# acting on the full gap, so xdot^F = delta*(x^L - x^F) exactly.
_DELTA_DEV_DEFAULT = Params().delta_dev
_DELTA_ALGO_SHARE = 0.3
# merged-delta prior for Monte Carlo: the image of the ratified lag prior (lognormal, 90% CI
# [4, 12] months) under delta = 12/lag -> median 1.71/yr, 90% CI [1.0, 3.0]/yr. Only used if a
# caller samples delta directly; the widget samples the LAG and inverts.
MERGED_DELTA_RANGE = ('lognormal', float(np.log(12.0 / np.sqrt(4.0 * 12.0))),
                      float(np.log(12.0 / 4.0) / 3.29))

def split_delta(delta_total):
    """Base-model mapping: the merged catch-up rate delta acts entirely on the capability gap,
    i.e. (delta_dev, delta_rel) = (0, delta), so catch-up = delta*(x^L - x^F) EXACTLY (the base
    model the widget's Level 1 presents). Level 6 unpacks delta into the two channels; the
    effective single rate then is delta_rel + algo_share*delta_dev (see D-034)."""
    return 0.0, float(delta_total)

def mc_draw_batch(p_base, n, seed=0, n_points=200, sample_keys=None, merge_delta=False,
                  target_ranges=None, param_ranges=None, coherence=BILL_COHERENCE):
    """Per-draw records for the live Monte-Carlo view. Each record carries the sampled values,
    the (downsampled) trajectories the widget plots, and a few headline scalars.

    D-037: sampling is TARGET-SPACE wherever a target exists. `sample_keys` may mix
    TARGET_RANGES keys (drawn in natural units and inverted per draw via invert_targets;
    merge_delta selects the merged vs channels lag inversion) with PARAM_RANGES keys (free dials,
    drawn in parameter space). sample_keys=None -> every PARAM_RANGES key (legacy full param
    space, no targets). Everything not sampled stays pinned at p_base. The `params` record shows
    the drawn targets in their natural units (for the MC inspector strip). Trajectories are
    downsampled to <= n_points points; every draw shares the same time grid (T, dt from p_base).

    D-076: each accepted record carries `rejects` -- how many draws were discarded before it by
    the bill-growth coherence constraint (see BILL_COHERENCE). Pass coherence=None to disable."""
    # target_ranges / param_ranges: optional overrides (e.g. user-narrowed sampling ranges,
    # layered over the module defaults by the widget); None -> the module-level dicts.
    TR = TARGET_RANGES if target_ranges is None else target_ranges
    PR = PARAM_RANGES if param_ranges is None else param_ranges
    rng = np.random.default_rng(seed)
    keys = list(PR) if sample_keys is None else list(sample_keys)
    tkeys = [k for k in keys if k in TR]
    raw = [k for k in keys if k in PR]
    raw_plain = [k for k in raw if PR[k][0] != 'scale_of']
    raw_scaled = [k for k in raw if PR[k][0] == 'scale_of']
    out = []
    idx = None

    def _one_draw():
        """One joint draw -> (Params, drawn dict, targets dict). No acceptance test here."""
        drawn = {}
        for k in raw_plain:
            drawn[k] = _draw_one(PR[k], rng, drawn)
        targets = {tk: float(_draw_one(TR[tk], rng, {})) for tk in tkeys}
        # invert the non-lag targets first, so scale_of raws (g_a_F ~ g_a) couple to the DRAWN
        # effective-compute target; then draw those raws; then the lag inversion (which needs them).
        t1 = {k: v for k, v in targets.items() if k != 't_lag_mo'}
        inv = invert_targets(t1, replace(p_base, **drawn), merged=merge_delta)
        for k in raw_scaled:
            ctx = dict(drawn); ctx.update(inv); ctx.setdefault('g_a', p_base.g_a)
            drawn[k] = _draw_one(PR[k], rng, ctx)
        t2 = {k: v for k, v in targets.items() if k == 't_lag_mo'}
        if t2:
            inv.update(invert_targets(t2, replace(p_base, **drawn, **inv), merged=merge_delta))
        return replace(p_base, **drawn, **inv), drawn, targets

    for _ in range(n):
        rejects = 0
        p, drawn, targets = _one_draw()
        while coherence is not None and not bill_coherent(p, coherence) \
                and rejects < _COHERENCE_TRIES:
            rejects += 1
            p, drawn, targets = _one_draw()
        s = simulate(p)
        if idx is None:
            nt = len(s['t']); k = max(1, int(np.ceil(nt / n_points)))
            idx = np.arange(0, nt, k)
            if idx[-1] != nt - 1:
                idx = np.append(idx, nt - 1)
        prof = s['profit']
        pos = prof >= 0.0
        crossing = float(s['t'][int(np.argmax(pos))]) if pos.any() else float('nan')
        out.append(dict(
            params={**drawn, **targets},
            t=s['t'][idx], profit=prof[idx],
            x_L=s['x_L'][idx], x_F=s['x_F'][idx], Delta=s['Delta'][idx],
            a_L=s['a_L'][idx], a_F=s['a_F'][idx], c_L=s['c_L'][idx], c_F=s['c_F'][idx],
            W_R=s['W_R'][idx], W_F=s['W_F'][idx],
            revenue=s['revenue'][idx], cost=s['cost'][idx], cum_profit=s['cum_profit'][idx],
            crossing=crossing, cum_profit_T=float(s['cum_profit'][-1]),
            blowup=bool(np.nanmax(s['x_L']) > 25.0), rejects=int(rejects),
        ))
    return out

# %%
# ---- Cell E8b: calibration sources + tight default simulation ranges (D-042, resynced D-076) ----
# CAL_SOURCES is the documented per-parameter source table. It drives the widget's calibration
# panel (source rows with [choose] / [choose range]). Keyed by PARAMETER name; `value` is in the
# TARGET's natural units where a target drives the parameter, in parameter units for free dials.
# Faithful to Notes/calibration/param_docs/ (the briefs Pavel ratified) and to the evidence
# register Notes/calibration/evidence/ -- no invented sources.
#
# Row fields: source · value · unit · note · grade (A solid measurement · B reasonable anchor ·
# C judgment / weakly identified · F free choice) · ci (the source's own documented interval) ·
# ci_default=False (a judgment band, offered for [choose range] but excluded from the default
# span) · group (menu subheading -- rows measuring DIFFERENT OBJECTS are grouped apart) ·
# display_only=True (shown for context, never clickable: a different object, a bound, or a
# retired reading) · triple (RETIRED with the money rows, D-093 -- name reserved, never reuse) ·
# disp (display string replacing "value unit", for a row whose machine value is an exact fraction
# that must not be rounded on screen) · basis (money and coverage rows: the ACCOUNTING BASIS --
# calendar / run-rate / mixed -- mandatory disclosure, see the coverage block) · why
# (display_only rows: the short reason THIS row cannot be adopted, replacing the generic caption).

# ---- the fringe lag (brief 05, TL4 as amended: rows 1-7 plus the fringe-consistent reading;
# the stale-Cottier, compute-basis and drift rows are documentation, not menu rows).
# LANGUAGE: the follower is the COMPETITIVE FRINGE. Open-weight models are its measurement proxy;
# API-first competitively-priced models count as fringe from their API date (Pavel, 2026-07-26).
_LAG_SOURCES = [
    dict(source="Epoch ECI, published headline (lenient rule)", value=4.0, unit="mo", grade="A",
         note="public composite; a FLOOR, and the wrong catch-up rule for this model"),
    dict(source="UK AISI cyber, narrow-task suite", value=5.5, unit="mo", grade="A",
         ci=(4.0, 7.0), note="private agentic, narrow (gameable sub-score)"),
    dict(source="Epoch ECI, strict rule, same window", value=6.0, unit="mo", grade="A",
         note="the model's own definition: capability the fringe can actually match"),
    dict(source="Fringe-consistent 2026 reading", value=6.5, unit="mo", grade="B",
         note="counts API-first Chinese models (Kimi K3 etc.) as fringe from their API date"),
    dict(source="UK AISI cyber, autonomous ranges", value=7.0, unit="mo", grade="A",
         note="private agentic, long-horizon — construct-matched to the model's object"),
    dict(source="FrontierMath private tiers (reconstruction)", value=7.35, unit="mo", grade="C",
         ci=(6.6, 8.1), note="private; only four open models evaluated"),
    dict(source="Local ECI daily-grid refit, 2026 H1 / Jul", value=7.65, unit="mo", grade="B",
         ci=(7.5, 7.8), note="strict rule, time-averaged envelopes; own refit of Epoch's data"),
    dict(source="METR / private composite", value=9.0, unit="mo", grade="B", ci=(8.0, 10.0),
         note="agentic time-horizon; blog-composited"),
]

# ---- the money side. `_MONEY_SOURCES` -- the five (R0, m, k) lab triples -- was DELETED by
# D-093 along with the three parameters it documented. The evidence itself is not lost and was
# never in danger: every one of those rows is restated on the model's own margin definition in
# _COVERAGE_SOURCES below (the four per-lab rows plus the four A-D constructions), derived in
# Notes/calibration/cov0_source_menu_2026-07-27.md, and traced to
# Notes/calibration/fin4_restatement_example_2026-07-27.md and the evidence register. What went
# is the MENU, because the dials it fed no longer exist -- inputs to a calibration are not
# parameters of a model, and only the coverage ratio was ever identified.
#
# (The `triple` row field went with it. It is still documented in the field list above as a
# retired key, so a future money row cannot silently reuse the name for something else.)

# ---- the COVERAGE dial ρ₀ = m/k (D-080 follow-up, Pavel: "I want the calibration of coverage to
# let one choose based on different sources"). ONE ROW = ONE (before-model-building profit,
# model-building outlay) pair reduced to the only identified number, IN PERCENT -- R₀ cancels out
# of ρ₀ entirely. Every value is DERIVED, with the arithmetic on the row and in full in
# Notes/calibration/cov0_source_menu_2026-07-27.md, from the FIN4(b) restatement
# (Notes/calibration/fin4_restatement_example_2026-07-27.md) and through it from the register.
#
# BASIS is mandatory disclosure, not decoration: FIN4 §3's operative finding is that a CALENDAR
# cost may not be divided by a RUN-RATE profit -- that mismatch alone is worth ~1.4× on k/m, more
# than the gross-margin question that dominates the per-lab rows. Rows on different bases are not
# commensurable and the chip says which is which.
#
# CHOOSABLE ⟺ the implied ρ₀ lands inside the vetted [33, 56]% envelope (state.APP_RANGES). Rows
# outside it -- including every restatement whose m came out ≤ 0 -- are display_only and carry
# `why`. The envelope was deliberately NOT widened to admit any of them: the one row that would
# argue for a higher ceiling (Anthropic, 100%) is grade D with an 8× internal contradiction in its
# cost leg. Flags for Pavel are in §4 of the derivation file.
# ---- alpha group headings (D-098). The headings carry the CONSTRUCT, because the two readings
# answer different questions and a reader who cannot tell them apart cannot use the menu. The
# eta note is on the headings rather than buried in the log: a row labelled 0.67 that delivers
# 0.44 at another eta is exactly the two-literals trap this session hit repeatedly.
_ALPHA_ETA_NOTE = "values shown at η = 1; the delivered α moves with the substitution setting"
_ALPHA_DEFAULT = f"The shipped default ({_ALPHA_ETA_NOTE})"
_ALPHA_LEVEL = ("Level reading — compute's share of R&D SPEND "
                f"({_ALPHA_ETA_NOTE})")
_ALPHA_GROWTH = ("Growth reading — compute's share of research-effort GROWTH "
                 f"({_ALPHA_ETA_NOTE})")
_ALPHA_BOUND = "Bounds and counter-readings"
_ALPHA_PRACTICE = "State of practice — not evidence"

_COV_CAL = "Industry total — the calibrated readings"
_COV_CON = "Industry total — the four basis × population constructions"
_COV_LAB = "Per-lab restatements — context only"
_COVERAGE_SOURCES = [
    dict(source="Ratified base calibration (D-076) — the widget default", value=160.0 / 3.0,
         disp="ρ₀ = 53.3%", unit="%", grade="C", basis="mixed", group=_COV_CAL,
         note="m 40% ÷ k 75% = 53.3% (k/m 1.875). Profit ≈\\$40B before model-building "
              "(costs-023 Reading 2, re-founded on the model's own line in param_W0.md §D.1) "
              "over the ≈\\$75B bottom-up model-building sum (costs-020). MIXED: the profit's "
              "Anthropic leg is a one-month May-2026 annualization while the cost is "
              "calendar-2026 — FIN4 §1 Row 1 (iv) calls that the row's largest defect. This row "
              "carries the exact 160/3, so choosing it restores the calibrated default"),
    dict(source="FIN4(b) restated central — the recommended reading", value=100.0 / 2.4,
         disp="ρ₀ = 41.7%", unit="%", grade="C", basis="mixed", group=_COV_CAL,
         # D-094: the interval IS the source's own, inverted — k/m ∈ [1.8, 3.0] (FIN4 §3, the
         # span of constructions A–D below) maps to ρ₀ ∈ [1/3.0, 1/1.8]. Inverting swaps the
         # ends, which is why the LOW k/m gives the HIGH coverage. The only adoptable bracket
         # in this menu, and exactly where the [33, 56] envelope came from.
         ci=(100.0 / 3.0, 100.0 / 1.8),
         note="1 ÷ 2.40 = 41.7%. FIN4 §3's verdict on the industry row is k/m ≈ 2.4, interval "
              "[1.8, 3.0] ⇒ ρ₀ ∈ [33.3, 55.6]% — which is where this dial's [33, 56] envelope "
              "comes from. It spans BOTH bases (it is the span of the four constructions "
              "below); two independent routes — the cost anchor, and the loss identity "
              "k/m = 1 + Loss/(m·R₀) which bypasses the cost anchor entirely — agree to 6%"),
    dict(source="A — run-rate profit, broad population", value=100.0 * 35.9 / 75.0,
         disp="ρ₀ = 47.9%", unit="%", grade="C", basis="mixed", group=_COV_CON,
         note="\\$35.9B ÷ \\$75B = 47.9% (k/m 2.09). Profit = OpenAI 4.7 + Anthropic 21.2 "
              "(run-rate) + Google 10 (judgment, unsourced) + xAI/Meta 0; cost = costs-020's "
              "calendar-2026 bottom-up sum. This is param_W0.md §D.1's own construction"),
    dict(source="B — run-rate profit, Meta struck from both sides", value=100.0 * 35.9 / 65.0,
         disp="ρ₀ = 55.2%", unit="%", grade="C", basis="mixed", group=_COV_CON,
         note="\\$35.9B ÷ \\$65B = 55.2% (k/m 1.81). The model casts Meta as the open-weight "
              "fringe, which earns no rent (costs-019), so charging its \\$5–15B of spend to the "
              "leader while crediting it nothing is an asymmetry — param_W0.md §D.2's "
              "recommended fix, still open for Pavel. The most labs-favourable corner"),
    dict(source="C — calendar/de-spiked profit, broad population", value=100.0 * 25.1 / 75.0,
         disp="ρ₀ = 33.5%", unit="%", grade="C", basis="calendar", group=_COV_CON,
         note="\\$25.1B ÷ \\$75B = 33.5% (k/m 2.99). Anthropic's de-spiked calendar-2026 revenue "
              "(costs-026) replaces the \\$47B run-rate, so BOTH legs are calendar. FIN4 §3 "
              "argues this is the correct basis: a calendar-year total is, to first order, the "
              "mid-year instantaneous rate, which is param_S0.md's t = 0 ≈ mid-2026 convention"),
    dict(source="D — calendar/de-spiked profit, Meta struck from both sides",
         value=100.0 * 25.1 / 65.0, disp="ρ₀ = 38.6%", unit="%", grade="C", basis="calendar",
         group=_COV_CON,
         note="\\$25.1B ÷ \\$65B = 38.6% (k/m 2.59). C and D together are the conservative end "
              "of the envelope; the population question (Meta in or out) is worth ~5 pp, the "
              "BASIS question ~14 pp — which is the whole lesson of this menu"),
    dict(source="Anthropic, mid-2026 (restated)", value=100.0 * 0.45 / 0.45, disp="ρ₀ = 100.0%",
         unit="%", grade="D", basis="mixed", group=_COV_LAB, display_only=True,
         # D-094: m ∈ [39, 50]% (EBTIT 36% plus a \$2–5B non-training-R&D add-back, FIN4 §1
         # Row 4) over k = 45% ⇒ ρ₀ ∈ [86.7, 111.1]%. DISPLAY-ONLY still: an interval does not
         # make an out-of-envelope row adoptable, and this one does not even reach [33, 56].
         ci=(100.0 * 0.39 / 0.45, 100.0 * 0.50 / 0.45),
         why="restates ABOVE the [33, 56]% envelope, and its cost leg is contradicted 8× inside "
             "the register — adopting it would clip to 56% and hide both facts.",
         note="m 45% ÷ k 45% = 100% (k/m 1.00). Shipped as EBTIT 36% ÷ 45% = 80%; EBTIT is "
              "STRICTER than the model's m (it also nets non-training R&D, which k already "
              "charges), so adding \\$2–5B back raises m to 45% [39, 50] ⇒ ρ₀ ∈ [86.7, 111.1]%. "
              "The k leg is a backloaded 2026–29 average (costs-012) over a run-rate "
              "denominator; costs-011/020 instead give k = 5–11%, i.e. ρ₀ ≈ 410–900%"),
    dict(source="OpenAI 2026 (leaked financials, restated)", value=100.0 * 0.184 / 1.404,
         # D-094: the SG&A share s is an ASSUMPTION, s ∈ [3, 28]% of revenue, over which FIN4
         # §1 Row 2 reports k/m swinging [4.4, 12.5] ⇒ ρ₀ ∈ [1/12.5, 1/4.4] = [8.0, 22.7]%.
         # The WIDTH is the finding: k/m is hyperbolic near m = 0, so this row cannot pin the
         # ratio however it is read. Display-only, below the envelope.
         ci=(100.0 / 12.5, 100.0 / 4.4),
         disp="ρ₀ = 13.1%", unit="%", grade="D", basis="mixed", group=_COV_LAB,
         display_only=True,
         why="restates BELOW the envelope, and k/m is hyperbolic near m = 0 — the row is "
             "uninformative about the ratio however it is read.",
         note="m 18.4% ÷ k 140.4% = 13.1% (k/m 7.63). Shipped m 39% was a GROSS MARGIN "
              "(costs-007, Q1-2026) ⇒ ρ₀ 30.5%; deducting the S&M+G&A residual (20.2% of "
              "revenue, recovered from the −122% operating margin against costs-005's \\$32B "
              "R&D line) drops m to 18.4%. That residual is an ASSUMPTION, range [3, 28]%, over "
              "which ρ₀ swings [8.0, 22.7]%"),
    dict(source="OpenAI 2025 (realized, leaked full-year, restated)", value=100.0 * -0.134 / 1.467,
         disp="ρ₀ = −9.1%", unit="%", grade="B", basis="calendar", group=_COV_LAB,
         display_only=True,
         why="m is NEGATIVE on the source's own reported numbers, so coverage has no meaning — "
             "a negative ratio cannot be a calibration spot.",
         note="m −13.4% ÷ k 146.7% = −9.1%. The only fully-decomposed source in the corpus and "
              "the only one that reproduces its own operating margin EXACTLY (−160.1%): revenue "
              "13.07, cost of revenue 7.50, R&D 19.18, S&M+G&A 7.32 (costs-008). Shipped as "
              "gross margin 43% ÷ 147% = 29.3%, which silently omits the \\$7.32B of S&M+G&A. "
              "The sharpest evidence that a gross margin is not a conservative proxy for the "
              "model's m — it is the wrong SIGN"),
    dict(source="Epoch, OpenAI 2024 decomposition (restated)", value=100.0 * -0.621 / 1.757,
         disp="ρ₀ = −35.3%", unit="%", grade="C", basis="calendar", group=_COV_LAB,
         display_only=True,
         why="m is NEGATIVE and the level is stale (2024) — but note that its SHIPPED triple "
             "implied an adoptable-looking 37.8%. This row is the menu's cautionary tale.",
         note="m −62.1% ÷ k 175.7% = −35.3%. Shipped m 51% was a COMPUTE-ONLY margin (nets "
              "inference compute alone, costs-006); the register's own 2024 cost-of-revenue "
              "figure (costs-008) gives a true gross margin of 28.4%, and S&M+G&A ran ≈90% of a "
              "\\$3.7B top line (costs-039). k rises to 175.7% because the model's k includes "
              "R&D staff (\\$1–2B assumed, costs-038), which the decomposition excludes"),
]

CAL_SOURCES = {
    # ---------------------------------------------------------------- brief 01: g_C0 (TC1b/TC5)
    "g_C0": [
        dict(source="Epoch 2025-09, “GPT-5 used less compute”", value=3.5, unit="×/yr", grade="A",
             ci=(3.0, 4.0), group="Capability frontier — the model's object",
             note="frontier-BY-CAPABILITY, 2023–25: the growth rate of the compute behind the "
                  "most capable model"),
        dict(source="Implied by the bill ÷ hardware price-performance", value=3.24, unit="×/yr",
             grade="B", group="Capability frontier — the model's object",
             note="THE CALIBRATED SPOT. Dollar identity 2.4 (bill growth) × 1.35 (price-perf.) — "
                  "an independent route that lands inside the capability-frontier band"),
        dict(source="Sevilla & Roldán 2024 (Epoch), frontier post-2018", value=4.2, unit="×/yr",
             grade="A", ci=(3.6, 4.9), group="Compute frontier — largest run (different object)",
             note="the headline series; tracks the largest TRAINING RUN, not the most capable model"),
        dict(source="Same report, full 2010–24 window", value=5.3, unit="×/yr", grade="A",
             group="Compute frontier — largest run (different object)",
             note="includes the pre-2018 catch-up transient"),
        dict(source="Epoch 2026-01, global AI chip capacity", value=3.3, unit="×/yr", grade="A",
             ci=(2.7, 4.1), group="Compute frontier — largest run (different object)",
             note="installed base, supply side"),
        dict(source="Pilz et al. 2025, AI supercomputers", value=2.5, unit="×/yr", grade="A",
             group="Compute frontier — largest run (different object)",
             note="deployed FLOP/s of clusters, not single-run compute"),
        dict(source="Epoch 2025-09, lead-time analysis", value=5.0, unit="×/yr", grade="A",
             display_only=True, group="Compute frontier — largest run (different object)",
             note="~5×/yr for 1–2 years, then converging to stock growth ~2.2×/yr — a MECHANISM "
                  "row (it belongs to the compute slowdown), not a competing estimate of today"),
    ],
    # ------------------------------------------------- brief 07: effective-compute growth (GE1c)
    # The dial is t_eff_x; g_a = log10(t_eff_x) − g_C0 is the residual. Values below are t_eff in
    # ×/yr = 3.24 × the source's algorithmic rate, so the menu is directly comparable.
    "g_a": [
        dict(source="Gundlach et al. 2025 — frontier ablation (2.23×/yr algo)", value=7.2,
             unit="×/yr", grade="A", group="Lower bound — pretraining efficiency only",
             note="excludes post-training know-how, which this model puts in a (D-011)"),
        dict(source="Ho et al. 2024 — pretraining efficiency (2.69×/yr algo)", value=8.7,
             unit="×/yr", grade="A", ci=(5.8, 20.6),
             group="Lower bound — pretraining efficiency only",
             note="the flagship source; its own 95% CI on the algorithmic rate is [1.79, 6.35]×/yr"),
        dict(source="Mertens et al. — developer-effects design (3.2×/yr algo)", value=10.4,
             unit="×/yr", grade="B", group="Lower bound — pretraining efficiency only",
             note="cross-developer panel; pretraining basis"),
        dict(source="Epoch “Rosetta Stone” §3.2.2, test-time-deflated", value=11.34, unit="×/yr",
             grade="B", ci=(10.7, 14.3), group="Bias-corrected reading — the calibrated default",
             note="THE CALIBRATED SPOT (11.34 = 3.24 × 3.50). Epoch's delivered-capability "
                  "estimate 5.86×/yr algo, deflated by the DIRECTLY MEASURED test-time share "
                  "(Qwen3 Instruct→Thinking, same weights same date, +5.81 ECI pts): capability "
                  "bought at inference time is in neither a nor c"),
        dict(source="Epoch “Rosetta Stone” §3.2.2, as published (5.86×/yr algo)", value=19.0,
             unit="×/yr", grade="A", group="Upper bound — test-time compute included",
             note="delivered-capability basis: contains capability bought at inference time"),
        dict(source="Same paper, “frontier of algorithmic quality” (≈9×/yr algo)", value=29.2,
             unit="×/yr", grade="A", display_only=True,
             group="Upper bound — test-time compute included",
             note="measured at the quality frontier rather than the capability frontier; "
                  "range 3–40×/yr"),
        dict(source="Ho 2026 — whole-stack (≈10×/yr algo)", value=32.4, unit="×/yr", grade="B",
             display_only=True, group="Wrong object",
             note="also counts INFERENCE-cost efficiency, which this model prices separately "
                  "through g_p — including it here would double-count"),
    ],
    # ---------------------------------------------------------------- brief 05: the fringe lag
    "Delta0": _LAG_SOURCES, "delta_dev": _LAG_SOURCES, "delta_rel": _LAG_SOURCES,
    "delta_total": _LAG_SOURCES,
    # ---------------------------------------------------------------- brief 02R: nu (TV4'/TV5')
    "nu": [
        dict(source="Pooled — the three constructions together", value=2.1, unit="×/OOM",
             grade="B", ci=(1.7, 2.65), group="Aggregate constructions — the model's object",
             note="THE CALIBRATED SPOT. Median of three independent routes on the current ruler"),
        dict(source="A — Davidson value datum", value=1.86, unit="×/OOM", grade="B",
             ci=(1.62, 2.21), group="Aggregate constructions — the model's object",
             note="FLOP-gap arithmetic; OOM-native"),
        dict(source="B — GATE ramp + wage-bill ceiling", value=2.30, unit="×/OOM", grade="B",
             ci=(1.98, 2.84), group="Aggregate constructions — the model's object",
             note="automation accounting; OOM-native"),
        dict(source="C — revenue decomposition", value=2.24, unit="×/OOM", grade="B",
             ci=(1.84, 2.72), group="Aggregate constructions — the model's object",
             note="observed revenue growth ÷ the frontier speed this calibration fixes"),
        dict(source="RLI (paid outcomes)", value=16.1, unit="×/OOM", grade="C", display_only=True,
             group="Single-channel benchmark slopes — NOT the economy's slope",
             note="one fixed basket of tasks. A dollar-weighted mixture of channels whose "
                  "midpoints are spread over ±2–3 OOM aggregates to ~2×/OOM even when individual "
                  "channels ramp at 16× — these rows CORROBORATE the aggregate, they don't "
                  "compete with it"),
        dict(source="SWE-bench", value=10.7, unit="×/OOM", grade="C", display_only=True,
             group="Single-channel benchmark slopes — NOT the economy's slope", note=""),
        dict(source="OSWorld", value=4.66, unit="×/OOM", grade="C", display_only=True,
             group="Single-channel benchmark slopes — NOT the economy's slope", note=""),
        dict(source="GDPval", value=3.11, unit="×/OOM", grade="C", display_only=True,
             group="Single-channel benchmark slopes — NOT the economy's slope",
             note="its near-linearity is a graded-metric artefact"),
        dict(source="Legacy widget default", value=1.73, unit="×/OOM", grade="F",
             display_only=True, group="Retired",
             note="e^0.55 — an artefact of a retired natural-log unit convention, never a "
                  "measurement"),
    ],
    # ------------------------------------------------------------- brief 01 rider: g_p (TC2b)
    "g_p": [
        dict(source="Trusted hardware leg (the calibration)", value=1.38, unit="×/yr", grade="B",
             ci=(1.30, 1.45),
             note="THE CALIBRATED SPOT, g_p = 0.14 OOM/yr. Prices fall to 72% of the year before; "
                  "implied bill growth 10^(g_C0−g_p) = 2.35×/yr vs Cottier's observed 2.4×/yr"),
        dict(source="Hobbhahn, Heim & Aydos 2023 (Epoch) — ML hardware price-performance",
             value=1.39, unit="×/yr", grade="A", ci=(1.27, 1.54),
             note="FP32 regression; the measured leg the calibration trusts"),
        dict(source="Dollar-identity leg (bill 2.4 ÷ compute 3.24)", value=1.35, unit="×/yr",
             grade="B", note="the same identity read backwards — how the 3.24 spot was built"),
        dict(source="Bill-residual reading (retired)", value=1.75, unit="×/yr", grade="F",
             display_only=True,
             note="the pre-audit convention: g_p as whatever reconciles a 4.2×/yr compute trend "
                  "with a 2.4×/yr bill. Retired — the hardware leg is measured, not residual"),
    ],
    # ------------------------------------------------------------ brief 04/04b: the money side
    # D-093: the "k_build" / "R0" / "m_margin" entries are gone with their parameters, and
    # so are the two DERIVED read-out rows ("kappa", "B0") -- a menu documenting a number the
    # user can neither set nor see is dead weight. THE MONEY SIDE NOW HAS EXACTLY ONE MENU,
    # which is the point: rho is the one identified object, so it is the one thing with sources.
    "cov0": _COVERAGE_SOURCES,
    # ------------------------------------------------------------ EXTENSIONS — extensions round
    "g_C_inf": [
        dict(source="Hobbhahn 2023 — hardware-only trend", value=1.35, unit="×/yr", grade="C",
             note="floor LEVEL is our extrapolation; power binds ~2030 (Sevilla et al. 2024)"),
    ],
    "ell": [dict(source="Payment-weighted composite, 2026 vintage", value=0.45, unit="yr",
                 grade="B", ci=(0.25, 1.3),
                 note="run duration + finish→release gap, weighted by WHEN the dollars are paid; "
                      "RL dollars paid late shorten it. Enters with the compute slowdown — in "
                      "the base, constant compute growth makes it cancel identically")],
    "phi_RD": [dict(source="Folded into the bill anchor", value=0.0, unit="×", grade="B",
                    display_only=True,
                    note="the ratified cost anchor is the OBSERVED bill — compute AND R&D / "
                         "researcher overhead together (they move roughly proportionally), so no "
                         "separate markup is calibrated in the base")],
    # D-084 position dials -- grade F scenario knobs, no evidence pass yet (flagged with the
    # envelopes for the calibration round). The two rows per dial are the honest bracket: the
    # inherited convention, and Pavel's own worked example of the alternative.
    "p0_c": [
        dict(source="Convention inherited from D-082", value=1.0, unit="%", grade="F",
             note="the slowdown has barely started at t = 0 -- what the widget assumed before "
                  "the dial existed; keeping it as the default is what makes every pre-D-084 "
                  "path bit-identical"),
        dict(source="Already visibly under way", value=10.0, unit="%", grade="F",
             note="Pavel's worked example: \"we are in the bottom 10% of the s-curve and it "
                  "will flatten (middle) in 3 years\" -- set t_mid = 3 alongside it"),
    ],
    "p0_w": [
        dict(source="Convention inherited from D-082", value=1.0, unit="%", grade="F",
             note="the value slope has barely started easing at today's frontier, so nu IS "
                  "essentially today's slope"),
        dict(source="Commoditization already biting", value=10.0, unit="%", grade="F",
             note="today's slope already a tenth of the way from nu down to nu_inf -- the "
                  "reading under which benchmark saturation is visible now"),
    ],
    "p0_F": [
        dict(source="Convention inherited from D-082", value=1.0, unit="%", grade="F",
             note="the fringe's own slowdown has barely started; independent of the leader's "
                  "p0_c by construction"),
        dict(source="Already visibly under way", value=10.0, unit="%", grade="F", note=""),
    ],
    "x_mid": [
        dict(source="Early-commoditization reference", value=2.0, unit="OOM", grade="C",
             note="the value-slope transition ν → ν_∞ is half-done 2 OOM out (D-083 re-keying)"),
        dict(source="Mid reference", value=5.0, unit="OOM", grade="C", note=""),
        dict(source="Harvest-continues reference", value=10.0, unit="OOM", grade="C",
             note="today's slope ν carries essentially across the horizon (midpoint 10 OOM out)"),
    ],
    "g_a_F": [
        dict(source="Scale-bias low, 0.6 × leader (Gundlach)", value=0.6, unit="× leader",
             grade="B", note=""),
        dict(source="Scale-bias central, 0.7 × leader (Gundlach)", value=0.7, unit="× leader",
             grade="B", note=""),
        dict(source="Scale-bias high, 0.8 × leader (Gundlach)", value=0.8, unit="× leader",
             grade="B", note="one source, three readings; the dial IS this share (g_a^F = share × g_a, audit X-10)"),
    ],
    "eta": [
        dict(source="Whitfill & Wu 2025 — substitutes (σ = 2.58)", value="0.61", unit="",
             grade="B", note="N = 27, 4 labs"),
        dict(source="Whitfill & Wu 2025 — complements (scale control)", value="-2 (complements)",
             unit="", grade="B", note="sign flips on the K_train control"),
    ],
    "gamma": [dict(source="Tentative default (no observable yet)", value=0.2, unit="/OOM",
                   grade="C", note="")],
    "beta0": [dict(source="Tentative default (no observable yet)", value=0.3, unit="", grade="C",
                  note="")],
    "t_mid": [dict(source="Scenario default — sweep it", value=2.3, unit="yr", grade="F",
               note="≈ the old ξ = 0.3 half-decay time ln 2/0.3 = 2.31 (D-082)")],
    "t_mid_F": [dict(source="Scenario default", value=2.3, unit="yr", grade="F",
                 note="same S-curve family and convention as t_mid (D-082)")],
    "split": [dict(source="Placeholder (open question)", value=0.5, unit="", grade="F", note="")],
    "g_CF0": [dict(source="Scenario knob — no calibration yet (Q-5 ruling)", value=0.5,
                   unit="OOM/yr", grade="F",
                   note="declared a grade-F scenario dial like t_mid_F; follower-compute calibration pass to come")],
    "g_CF_inf": [dict(source="Scenario knob — no calibration yet (Q-5 ruling)", value=0.10,
                      unit="OOM/yr", grade="F",
                      note="declared a grade-F scenario dial like t_mid_F; follower-compute calibration pass to come")],
    # ---- alpha (brief 10, D-098). Values are the OBSERVABLE loss_half_gC in PERCENT, converted
    # from each source's alpha at the BASE eta = 1 (where loss = alpha/2). The delivered alpha
    # then moves with the active eta -- adopting Epoch's 0.67 gives 0.67 at eta = 1 and 0.44 at
    # eta = -2. That is the ratified anti-double-counting property, not a defect, which is why
    # _ALPHA_ETA_NOTE says so in the user's own terms on the menu itself rather than only here.
    #
    # THE FORK IS NOT RESOLVED BY FIAT (Pavel: "This as default and let user choose in the
    # calibration"). The two readings are separate GROUPS, each labelled with its construct: the
    # cost-share evidence identifies a LEVEL elasticity, while the model's alpha-channel argument
    # g_c(t)/g_c(0) is a GROWTH rate. The default sits between them by design.
    "alpha": [
        dict(source="Ratified default — midpoint of the two readings", value=35.0, unit="%",
             grade="C", group=_ALPHA_DEFAULT,
             note="brief 10, Pavel 2026-07-28: α = 0.70 at η = 1. Deliberately between the "
                  "level cluster (32–34%) and the growth cluster (39–44%) — the fork is a user "
                  "choice, not ours"),
        # -- level reading: alpha = the compute share of R&D spend, S_E --
        dict(source="Epoch AI (Ho & Whitfill 2025) — ε_K, the only published estimate",
             value=33.5, unit="%", grade="B", ci=(29.5, 37.5), group=_ALPHA_LEVEL,
             note="ε_K ≈ 0.67 (range 0.59–0.75): 'the elasticity of output with respect to "
                  "capital should equal the compute share'. Verified verbatim 2026-07-28"),
        dict(source="Z.ai / Zhipu — audited HKEX prospectus", value=41.0, unit="%", grade="A",
             ci=(40.6, 41.4), group=_ALPHA_LEVEL,
             note="compute 82.7% of (compute+labour) R&D spend 2024, 81.2% H1 2025; cash+equity"),
        dict(source="MiniMax — audited HKEX prospectus", value=39.0, unit="%", grade="A",
             ci=(37.9, 40.2), group=_ALPHA_LEVEL,
             note="75.7% (2024) → 80.3% (9M 2025); near-frontier Chinese lab, steep trend"),
        dict(source="Epoch — OpenAI 2025 cost stack", value=32.2, unit="%", grade="B",
             group=_ALPHA_LEVEL,
             note="$8.3bn R&D compute vs $4.6bn labour, EQUITY-INCLUSIVE — equity treatment is "
                  "the single largest lever here, worth ~20 points"),
        dict(source="Cottier et al. 2024 — frontier training costs", value=27.5, unit="%",
             grade="A", ci=(23.5, 32.0), group=_ALPHA_LEVEL,
             note="hardware 47–64% incl. equity (61–76% excl.); ≤2023 vintage, and a per-model "
                  "amortised cost rather than a flow input share"),
        dict(source="Epoch — Anthropic 2025 cost stack", value=29.3, unit="%", grade="B",
             group=_ALPHA_LEVEL, display_only=True,
             why="a LOWER BOUND: the 'staff' line is a residual absorbing all other opex"),
        # -- growth reading: alpha = S_E g_E / (S_E g_E + S_L g_L) --
        dict(source="Epoch ε_K + measured input growth (the model's own symbols)", value=41.8,
             unit="%", grade="C", ci=(39.2, 44.2), group=_ALPHA_GROWTH,
             note="α = S_E·g_E/(S_E·g_E+S_L·g_L) with R&D compute ~3×/yr vs headcount "
                  "~1.25–1.6×/yr (costs-039) ⇒ α ≈ 0.84. Collapses to the level reading only "
                  "if g_E = g_L, which is where the 2026-07-22 note stopped"),
        dict(source="Cottier-internal + measured input growth", value=26.5, unit="%", grade="C",
             group=_ALPHA_GROWTH,
             note="the most labour-favourable defensible combination (S_E ≈ 0.31) ⇒ α ≈ 0.53"),
        # -- bounds and counter-readings --
        dict(source="Gundlach et al. 2025 — scale-dependence share", value=44.5, unit="%",
             grade="B", group=_ALPHA_BOUND,
             note="≈89% of measured algorithmic progress is scale-dependent ⇒ α ≈ 0.9; read as "
                  "a MECHANISM instead it is η evidence, so it is not spent twice"),
        dict(source="Barnett 2025 — compute-cap innovation catalogue", value=25.0, unit="%",
             grade="C", group=_ALPHA_BOUND,
             note="a GPT-2-compute or 8×H100 cap still allows about half of 36 catalogued "
                  "innovations ⇒ α ≲ 0.5. Discovery-weighted, so the low end of the fork"),
        dict(source="AI 2027 — elicited compute-cut elasticity", value=20.0, unit="%", grade="C",
             group=_ALPHA_BOUND, display_only=True,
             why="a LEVEL elasticity at a 10× cut, not this model's growth-rate object (and "
                 "n = 6, shaded up from 0.22 by the authors)"),
        # -- what the field assumes, kept because it is exactly what we were doing --
        dict(source="Davidson / Forethought — assumed, not measured", value=25.0, unit="%",
             grade="D", group=_ALPHA_PRACTICE,
             note="uses this model's exact CES and exact letter: 'I'll assume α = 0.5 "
                  "throughout', with no justification. STATE OF PRACTICE, NOT EVIDENCE — and "
                  "the same unjustified 0.5 this widget shipped until D-098"),
    ],
    "r": [dict(source="Standard discount rate", value=0.08, unit="/yr", grade="C",
               note="user-cost extension only")],
}


def lognormal_from_ci(lo, hi):
    """Lognormal from its 90%-CI endpoints: geometric-mid mu, sigma = ln(hi/lo)/(2·1.645)."""
    return ('lognormal', float(np.log(np.sqrt(lo * hi))), float(np.log(hi / lo) / 3.29))


def dist_bounds(rng):
    """Natural-unit endpoints of a distribution: uniform/triangular bounds, lognormal ~90% CI."""
    if rng[0] == 'lognormal':
        med = float(np.exp(rng[1]))
        return med * float(np.exp(-1.645 * rng[2])), med * float(np.exp(1.645 * rng[2]))
    if rng[0] == 'triangular':
        return float(rng[1]), float(rng[3])
    return float(rng[1]), float(rng[2])


def source_span(pkey):
    """[min, max] across a parameter's documented source values and measured CIs (D-042).
    Display-only rows (different objects, bounds, retired readings) are excluded."""
    vals = []
    for rw in CAL_SOURCES.get(pkey, []):
        if rw.get('display_only'):
            continue
        if isinstance(rw.get('value'), (int, float)):
            vals.append(float(rw['value']))
        ci = rw.get('ci')
        if ci is not None and rw.get('ci_default', True):
            vals += [float(ci[0]), float(ci[1])]
    return (min(vals), max(vals)) if vals else None


def _tight(key, pkey, kind='uniform'):
    """Tight default simulation range for one dimension: the source span, clipped to the
    envelope (TARGET_RANGES / PARAM_RANGES bounds)."""
    elo, ehi = dist_bounds(TARGET_RANGES.get(key) or PARAM_RANGES[key])
    slo, shi = source_span(pkey)
    lo, hi = max(float(slo), elo), min(float(shi), ehi)
    return lognormal_from_ci(lo, hi) if kind == 'lognormal' else ('uniform', lo, hi)


# D-042/D-076: the DEFAULT simulation range per MC dimension. These are the RATIFIED bands from
# the calibration round, stated explicitly rather than derived from the menu span — the menu now
# deliberately contains rows measuring different objects and bounds, which must not widen the
# default draw. Every dimension NOT listed here defaults to a POINT: it starts in spot mode and
# is not sampled until the user widens it (the envelope caps how far).
SIM_DEFAULT = {
    # TC6: "the range is so wide because there are different definitions and outliers; I want to
    # focus on the main trend" — the capability-frontier band, spot deliberately off-centre.
    't_compute_x': ('uniform', 3.0, 4.0),
    # GE4: Ho et al. 2024's own 95% CI on the algorithmic rate, shifted by the fixed g_C0.
    't_eff_x':     lognormal_from_ci(6.0, 20.0),
    # TV5': the pooled band of the three constructions; triangular with the mode at the spot
    # (the pooled distribution's central value), which the shape of the evidence supports.
    't_value_x':   ('triangular', 1.7, 2.1, 2.65),
    # TL5(b): lognormal, 90% CI [4, 12] months, median ~7.
    't_lag_mo':    lognormal_from_ci(4.0, 12.0),
    # TC2(b) rider: the external four-model bracket on hardware price-performance.
    't_price_x':   ('uniform', 1.30, 1.45),
    # SB6: the LEVEL is fixed (it provably cannot move the verdict); only the cost-to-earnings
    # ratio is drawn, and since D-093 that ratio IS the model's one finance parameter. The draw
    # lives app-side, in percent (ui/state.py APP_SIM_DEFAULT['cov0'] = [33, 56]); no money key
    # appears here or in PARAM_RANGES, so a future level cannot re-introduce a second money
    # dimension by listing one.
    # EXTENSION (channels, L6): Gundlach's own documented 0.6–0.8× band.
    'g_a_F':       ('scale_of', 'g_a', 0.6, 0.8),
    # D-098 (brief 10, ratified): the OBSERVABLE is drawn, never alpha itself, and alpha is
    # derived inside invert_targets at each draw's OWN eta. Triangular with the mode at the
    # ratified spot. Drawing alpha and eta as independent raw parameters would put mass in the
    # (alpha >= 0.8, eta <= -2) double-bottleneck corner that no source supports -- and sec. 3.4
    # of the note shows that is precisely where the verdict is most violently determined.
    'loss_half_gC': ('triangular', 22.0, 35.0, 45.0),          # %
}
