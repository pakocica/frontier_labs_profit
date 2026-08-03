"""Model parameters, their calibrated defaults, and the sampling envelopes (D-110).

Pure DATA plus two distribution helpers: the `Params` dataclass with every ratified default and
its provenance, the target/parameter dictionaries the widget and the Monte Carlo draw from, and
the tight default simulation ranges. It imports nothing but numpy — no model math reaches back
into it — which is what makes it the first module a JS/TS port can serialize rather than rewrite.

`model.py` re-exports every name here, so `import model as m` still reaches them as `m.Params`,
`m.TARGET_RANGES`, `m.SIM_DEFAULT` and the rest.

**Units (spec N1).** States a, c, x are in **OOMs** — base-10 logs of algorithmic level, training
compute, and effective compute x = a + c. Growth rates (g_C, g_a) are OOM/yr; diffusion rates
delta and the discount rate r are continuous per year; the value slope nu (per OOM) and the
compute-price decline g_p (OOM/yr) are **base-10** (D-039). t = 0 is early 2026, normalised
x^L_0 = 0, so every capability is OOMs above today's frontier. Value W is an **index**, W(0) = 1,
and since D-093 **the money block carries no dollars at all**: earnings and cost are both
normalised at t = 0, so B_0 = 1 and the whole finance side is measured in *multiples of today's
model-building bill*. One parameter survives, rho — coverage at t = 0 — and absolute FLOP counts
and absolute dollars alike never appear. The gap is Delta_t = x^L_t - x^F_t; the follower starts
Delta_0 behind, divided by `split` into an algorithmic and a compute part.
"""
import numpy as np
from dataclasses import dataclass


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
    rho: float = 25.1 / 75.0   # BASE: coverage at t = 0 = E_0/B_0 = $25.1B/$75B = 33.5%. "Labs
                               # currently earn ~33 cents per dollar of model-building spend";
                               # break-even is rho_t = 1. TODAY'S SNAPSHOT, never a structural
                               # constant -- the dynamics move earnings and cost at different
                               # rates and question (a) is when they cross. The widget dials it
                               # in PERCENT (ui/state.py APP_RANGES['cov0'], envelope [26, 46]).
                               #
                               # D-104 (FIN4 SETTLED, Pavel 2026-07-29) re-dated this from
                               # 0.40/0.75 = 53.3%. It was not overruled, it was RESTATED: every
                               # leg of rho_0 must estimate the instantaneous flow at ONE anchor
                               # date, t = 0 ~ mid-2026, and the shipped 53.3% divided a mid-2026
                               # cost by an earnings flow whose Anthropic leg was a spiked
                               # May-2026 run-rate. On one date the same evidence IS 33.5%.
                               # Consistency was worth x1.4-1.5; WHICH instant only +-3.5% per
                               # half-year, rho being a ratio. Stored as the fraction 25.1/75.0
                               # rather than a decimal so that 100*rho is bitwise construction C's
                               # own value in _COVERAGE_SOURCES and [choose] on it restores the
                               # calibrated spot exactly. Grade C; the contestable step is
                               # rejecting the May run-rate as ~2x off-trend (costs-026), so an
                               # audited Anthropic calendar-2026 figure above $26B moves it back
                               # up. Full working: Notes/calibration/param_docs/12_FIN4_resolution.md

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


# =============================================================================================
# TARGETS (D-037) -- observables in natural units
#
# Targets-first parameterization: wherever a parameter has a clean observable, the *observable* is
# the primitive. Slider bounds, Monte-Carlo distributions and the calibration documentation all
# live in target space (`TARGET_RANGES`, one source of truth), and `invert_targets` maps them back
# into model parameters at t = 0. The defaults are the exact forward images of `Params()`, so
# the default targets invert back to the parameter defaults precisely.
# =============================================================================================

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


# D-042 two-tier ranges: TARGET_RANGES is the ENVELOPE -- the outer bounds of what a user may
# sample; the DEFAULT simulation range is the tight documented span (SIM_DEFAULT, below).
# ENVELOPE RULE (Pavel, 2026-07-26): the envelope must contain the UNION of the confidence ranges
# of the sources the menu includes, with the left bound rounded down and the right rounded up.
# Display-only rows -- context, bounds, retired readings -- do not enter the union.
#
# D-109 (Pavel, 2026-08-02) sharpened this in two ways. The direction of fit is menu -> range: the
# defendable values are chosen first and the envelope adapts to contain them, never the reverse.
# And over-containment is out — a bound no row witnesses is not evidence of anything. The rule as
# written said "different object" rows are excluded from the union too, which is what let g_C0's
# right bound drift to a 6.0 that NO row has ever supported; his TC1 ruling keeps those rows on
# the menu as grade-A readings, so they are part of what the envelope must contain. An envelope
# may still sit WIDER than its union where something other than the menu pins it (t_lag_mo's [4,
# 12] is also the ratified sampling distribution, so it cannot be hugged to the menu's max of 10).
TARGET_RANGES = {
    # D-109: the menu's own witnessed span, 2.5 (Pilz) .. 5.3 (full 2010-24 window), and nothing
    # beyond it. Was [2.5, 6.0]; the 6.0 was witnessed by no row on the menu, present or retired.
    # BASE MC stays [3, 4], well inside. R11's 10% dial padding does NOT apply here -- this is a
    # target, not one of the thirteen free dials in _DIAL_SPEC, and targets sit at the envelope
    # exactly (ui/state.py, _tbounds).
    't_compute_x': ('uniform', 2.5, 5.3),                      # x/yr
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


# =============================================================================================
# MONTE-CARLO ENVELOPES -- the outer bounds of what a user may sample
# =============================================================================================
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
    'rho':      ('uniform', 0.26, 0.46),  # coverage today, the ONE money dimension (D-093). The
                                          # vetted [26, 46]% envelope in fraction units. D-104
                                          # RECENTRED it from [33, 56] (width 23 -> 20 pp): the
                                          # old width was a UNION ACROSS BASES, and once every leg
                                          # is dated to one instant there is no union left to
                                          # take. Ends are the one-at-a-time span around the two
                                          # admissible constructions C and D -- floor 26.2%
                                          # (Google struck both sides), ceiling 46.3% (Meta
                                          # struck, labs-favourable corner). Deliberately absent
                                          # from SIM_DEFAULT -- the widget's tight band is the
                                          # app-side APP_SIM_DEFAULT, and the two are the same
                                          # envelope, so a second copy here could only drift.
    'x_mid':    ('uniform', 2.0, 20.0),
    'ell':      ('lognormal', float(np.log(np.sqrt(0.25 * 1.3))),
                 float(np.log(1.3 / 0.25) / 3.29)),   # EL3': ~90% CI [0.25, 1.3] yr
    'g_p':      ('uniform', 0.09691, 0.190332),  # image of t_price_x envelope [1.25, 1.55] x/yr
    'r':        ('uniform', 0.03, 0.15),
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


# D-042/D-076: the DEFAULT simulation range per MC dimension. These are the RATIFIED bands from
# the calibration round, stated explicitly rather than derived from the menu span — the menu now
# deliberately contains rows measuring different objects and bounds, which must not widen the
# default draw. Every dimension NOT listed here defaults to a POINT: it starts in spot mode and
# is not sampled until the user widens it (the envelope caps how far).
#
# The helper that DID derive a band from the menu span (`_tight`, source span clipped to the
# envelope) is gone as of D-110's extraction: it had been dead code since the ratified literals
# below replaced it, and it was the only other reader of `source_span`. Its absence is what makes
# a menu trim provably unable to move a sampling default — the numbers here are the only ones.
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
    # lives app-side, in percent (ui/state.py APP_SIM_DEFAULT['cov0'] = [26, 46]); no money key
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
