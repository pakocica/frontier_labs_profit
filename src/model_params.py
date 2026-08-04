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
compute-price decline g_p (OOM/yr) are **base-10** (D-039). t = 0 is mid-2026 (D-104), normalised
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
    g_C_inf: float = 0.15      # EXTENSION (slowdown): the compute-growth FLOOR, OOM/yr = 1.4125
                               # x/yr. RECALIBRATED by D-129 from 0.13 (x1.3490) on Pavel's
                               # ruling that the floor is "hardware price-performance + the
                               # asymptotic economic growth" -- TWO LEGS, added in OOM/yr:
                               #
                               #   g_C_inf = g_pp,inf + s_inf = 0.14 + 0.01 = 0.15
                               #             hardware   budget growth (+2.33 %/yr real)
                               #
                               # The decomposition is an IDENTITY in this model, not an analogy:
                               # compute bought = dollars spent x FLOP per dollar, and the model
                               # contains exactly ONE price series, g_p. So the implied long-run
                               # real budget growth is the exact read-out 10^(g_C_inf - g_p),
                               # with no residual -- which is why the hardware leg is g_p = 0.14
                               # itself rather than the evidence cluster's midpoint 1.35 (0.14 is
                               # inside the measured cluster [1.30, 1.39] anyway). The near-term
                               # dial was ALREADY built this way: g_C0 = 3.24x/yr is bill 2.4 x
                               # hardware 1.35, so this is a consistency fix, not a new construct.
                               #
                               # WHAT THE OLD 0.13 ASSERTED, and why it could not stand: it was
                               # the hardware leg ALONE, i.e. 10^(0.13-0.14) = x0.977/yr -- the
                               # leader's real training budget SHRINKING 2.33 %/yr forever. The
                               # card's own story was the flat-budget world, and flat budget is
                               # g_C_inf = g_p exactly. The 0.01 gap was the residue of taking
                               # the hardware leg from the cluster midpoint while the model's
                               # price leg was ratified at 1.38. It now reads +2.33 %/yr: the
                               # bill GROWS asymptotically.
                               #
                               # The budget leg is long-run real GDP/GWP growth, between Jones's
                               # 2 %/yr US trend and the AI-boosted band's 2.5 % floor. It is a
                               # balanced-growth assumption and the only one under which the floor
                               # is a genuine asymptote: above trend, the training budget's share
                               # of output rises without bound. Any mainstream choice in 2-3 %/yr
                               # moves the floor by less than 0.02 x/yr, which is the strongest
                               # argument for the construct -- the contested leg barely matters.
                               # Grade C, deliberately NOT upgraded: both legs are grade A, but
                               # the PAIRING is a judgment extrapolated past 2030 and no source
                               # measures post-2030 frontier compute growth.
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
                               # progress lost if experiment-compute growth halved"), which at the
                               # base eta = 0 is loss = 1 - 2^(-alpha), so 38.44% <-> 0.70.
                               # Range [0.45, 0.90], grade C+ (Pavel, ratified 2026-07-28).
                               # D-125 did NOT move this number: the flip re-states the OBSERVABLE
                               # (35% -> 38.44%), not the weight, because the cost-share evidence
                               # the weight is read from is natively a Cobb-Douglas object --
                               # under CD the output elasticity EQUALS the cost share, which is
                               # the identification eta = 1 never had. alpha is EXACTLY inert at
                               # Level 1 (A1 short-circuits the bracket) and under Leontief (the
                               # min() branch never reads it), so this change cannot touch the
                               # base calibration ratified 2026-07-26.
    eta: float = 0.0           # base CES exponent. D-125 (Pavel, 2026-08-03) flips the default
                               # 1 -> 0, superseding D-018's perfect-substitution simplification:
                               # both are conventions, and only Cobb-Douglas has a literature
                               # behind it (the RSI field's own baseline, register algo-015).
                               # Continuous dial since the same ruling, envelope [-1.20, +1.00]
                               # (see PARAM_RANGES) -- eta = 1 is the CES family's mathematical
                               # ceiling, since sigma = 1/(1-eta) is negative above it.
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
    nu: float = 0.3228142429766918   # BASE (TV1'), DERIVED. The ANCHOR IS THE RATE, not the
                                     # per-OOM multiplier: D-118's rider (Pavel, 2026-08-02, "just
                                     # round it to integer") fixes today's value growth at exactly
                                     # x2.19/yr -- 119 %/yr, the unit it was ruled in before D-133
                                     # re-keyed the dial -- and this literal is its image through
                                     # the leader's t = 0 speed,
                                     #   nu = log10(2.19) / xdot^L(0),
                                     # xdot^L(0) = 1.0546 OOM/yr at these defaults. The re-key
                                     # moves NOTHING here: 2.19 and 1 + 119/100 are the same
                                     # double, so this literal is bitwise what D-118 shipped.
                                     # Each OOM of
                                     # capability is then worth x2.1029 more -- the approximate
                                     # gloss on a round rate, where the pre-rider default was the
                                     # exact image of the ratified x2.1/OOM (0.3222192947339193,
                                     # 118.6838 %/yr). The evidence did not move: 119 vs 118.68
                                     # %/yr is 0.14% on a quantity whose 90% band is [75, 179],
                                     # and the round number is the one a reader can hold.
                                     # x2.1/OOM remains the POOLED CENTRAL VALUE across three
                                     # independent constructions (Davidson value datum 1.86, GATE
                                     # ramp + wage-bill ceiling 2.30, revenue decomposition 2.24)
                                     # -- an equally-weighted Monte Carlo over their own input
                                     # bands, p50 = 2.08/OOM (valaut-091). NOT the median of
                                     # those three numbers: each is restated on this
                                     # calibration's ruler, and their median is 2.24.
                                     # D-088: nu = w'(0) literally, at every p0_w. Before D-088
                                     # it was the PRE-easing plateau, so this dial delivered
                                     # 2.089x today rather than the calibrated 2.1x.
    nu_inf: float = 0.1011445182196875  # EXTENSION (value slope transition, D-083),
                                       # RE-DERIVED by D-129 and AGAIN by D-125, and neither
                                       # time re-ratified: the ruled observable is still exactly
                                       # 10 %/yr (Amodei's floor, granted adversarially). This
                                       # literal is that rate's image through the leader's speed
                                       # AT THE HORIZON, so it moves whenever a ratified dial
                                       # moves that speed, and only then.
                                       #   D-129 raised the compute floor 0.13 -> 0.15, which
                                       # raised xdot^L(T) 0.4740 -> 0.5162 OOM/yr and cut the
                                       # slope 0.0873 -> 0.0802.
                                       #   D-125 flipped the base eta 1 -> 0 (Cobb-Douglas),
                                       # which SLOWS the leader -- a geometric mean of the two
                                       # research channels cannot be carried by whichever one is
                                       # running fastest -- so xdot^L(T) falls 0.5162 -> 0.4092
                                       # and the same 10 %/yr now buys a LARGER slope per OOM,
                                       # 0.1011 where it was 0.0802 (x1.262/OOM asymptotically,
                                       # against x1.223 before). Leaving either old literal in
                                       # place would have made the dial assert some other rate
                                       # while the ruling says 10, which is precisely the defect
                                       # D-118 exists to prevent. Params.nu
                                       # is untouched: its denominator is the speed TODAY, and
                                       # BOTH CES channels equal 1 at t = 0 by the D-086 anchor
                                       # (so adot^L(0) = g_a at every eta), which is why the
                                       # near-term anchor cannot move when the asymptote does.
                                       # The re-derivation is a single pass, not a fixed point --
                                       # the capability path does not read the value block.
                                       # CALIBRATED by D-107 and re-keyed by D-118/D-120.
                                       # ASYMPTOTIC value slope: each OOM is worth x1.262 more
                                       # once the transition is done (vs x2.1029 today). The L1
                                       # pin nu_inf = nu switches the transition OFF.
                                       # THE DIAL IS NOT THIS NUMBER. The observable is the
                                       # long-run growth rate of AI-attributable value, EXACTLY
                                       # x1.10/yr = 10 %/yr (Amodei's floor, granted adversarially
                                       # per D-107), and the parameter is its image through the
                                       # leader's capability speed at the horizon endpoint:
                                       # nu_inf = log10(1.10)/xdot^L(T), xdot^L(T) = 0.4092
                                       # OOM/yr at these defaults (0.5162 before D-125 flipped
                                       # the base eta; 0.474 before D-129 raised the compute
                                       # floor). So this literal is the
                                       # DEFAULT PATH's image of 10 %/yr and moves with no
                                       # other dial -- every other path re-derives its own
                                       # nu_inf from the same 10 %/yr. It replaces
                                       # log10(1.25) = 0.09691001300805642, which asserted
                                       # 11.16 %/yr here and a different rate at every other
                                       # compute setting: a rate no publication holds, at a
                                       # path the reader never chose.
                                       # D-118 rider (2026-08-02): the anchor is RE-STATED as
                                       # exactly 10 %/yr (it was already the ruled number; the
                                       # dial's shipped default had been the FORWARD image
                                       # 10.000000000000009). Re-deriving from the exact 10
                                       # returns this literal BITWISE -- the rider moves nothing
                                       # here.
                                       # D-133 (2026-08-03) re-keys the dial to x/yr, and this
                                       # literal is again bitwise unmoved: x1.10/yr and
                                       # 1 + 10/100 are the same double. What the re-key DOES
                                       # remove is the rider's one honest deviation. In %/yr the
                                       # forward image read 10.000000000000009 and could not be
                                       # made to read 10.0 -- growth_pct_of_slope ended in
                                       # 100*(10**y - 1) with 10**y near 1.1, whose ulp is
                                       # 2.2e-16, so the reachable values straddling 10 were
                                       # 9.999999999999987 and 10.000000000000009. In x/yr the
                                       # forward map IS 10**y, and BOTH anchors round-trip
                                       # exactly. The MENU row still binds the ANCHOR (see
                                       # cal_sources / model.py): clicking it restores nu_inf
                                       # bitwise, which is what D-109's rule is for.
    x_mid: float = 6.0         # EXTENSION (saturation); the base pins it huge (X_MID_EXP).
                                       # D-132 (Pavel, 2026-08-03, XM1): 10.0 -> 6.0 OOM. The old
                                       # 10 was witnessed by NOTHING -- its menu row was called
                                       # "Harvest-continues reference" and named no source -- and
                                       # it made the shipped (x_mid, p0_w) pair internally
                                       # inconsistent by 18.5x under the bounded-market reading
                                       # (see COHERENT_X_MID in model_dynamics). 6.0 is Epoch's
                                       # GATE ramp read at HALF OF TASKS AUTOMATED, arithmetic on
                                       # the playground's own live defaults (T = 36.5,
                                       # dF = 55%, C_T(0) = 1e25.70 => ramp 5.94 OOM; the
                                       # half-of-tasks point sits (0.5-f_init)/(1-f_init) = 44.4%
                                       # along it at f_init = 0.10; vintage -1.5). It is
                                       # corroborated to 0.12 OOM by a chain sharing NO input --
                                       # the ruler-free level identity x_mid = lam(q^w_0)
                                       # (1-q^w_0/100)/nu = 6.12, register valaut-090's K = 1.99.
    p0_w: float = 1.0          # D-084: the value-slope easing's OWN position dial (percent of
                                       # the nu -> nu_inf transition already done at today's
                                       # frontier, x = 0) -- see p0_c.

    # ----- C: cost -----
    # (D-093 retired B0 too. D-090 had already made it the observable cost(0) = k*R0; normalising
    # by that observable turns it into the constant 1, which no dataclass needs to carry. The
    # cost path is referenced to c^L(0) = 0, which is what makes B_0 = 1 hold at every dial.)
    # D-127 removed the two cost-side extensions that used to live here: the build lag `ell`
    # (D-123) and the R&D markup `phi_RD` (D-126). Their calibration records are archived under
    # Notes/calibration/retired/.
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
    #     B_t = 10^{c^L_t} * 10^{-g_p t}                                 =>  B_0 = 1
    #
    # HOW EXACTLY, measured -- the identities are algebraic, the arithmetic is not:
    #   * E_0 = rho is BITWISE at every base dial.
    #   * B_0 = 1 is BITWISE, unconditionally, at every dial. c^L(0) = 0 exactly and the cost
    #     path reads c^L(t) directly, so there is no de-lagging constant to divide out. (Until
    #     D-127 the path was referenced to c^L(ell) via the EXACT integral c_L_closed(ell) while
    #     simulate read c^L(t+ell) off the RK4 grid by interpolation; that gap cost 5 ulp at the
    #     shipped ell = 0.45 and 260 at ell = 2.5, and is why test_19 was written with a
    #     tolerance. Removing ell removed the gap.)
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
                               # in PERCENT (ui/state.py APP_RANGES['cov0'], envelope [26, 46.3]).
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


# ---------------------------------------------------------------------------------------------
# THE TWO VALUE ANCHORS (D-118 rider, Pavel 2026-08-02: "just round it to integer"; re-keyed to
# x/yr by D-133, Pavel 2026-08-03).
#
# The direction of derivation is RATE -> SLOPE. These are the ruled observables; Params.nu and
# Params.nu_inf above are their images through the leader's capability speed at the date each one
# describes (t = 0 for nu, the horizon endpoint for nu_inf), and `test_model` asserts that bitwise
# rather than trusting the two literals to stay in step. Before the rider the anchor was x2.1/OOM
# and the rate readings (118.6838..., 10.000000000000009 %/yr) were ITS images -- unroundable
# numbers on a dial the user reads as a rate.
#
# D-133 STATES THE SAME ANCHORS IN x/yr, for consistency with every other rate dial in the model
# (compute, effective compute, price-performance and the compute floor all speak x/yr). The map is
# affine -- x/yr = 1 + (%/yr)/100 -- and at these integer anchors it is EXACT in binary: 2.19 and
# 1.10 are, bit for bit, the doubles 1 + 119/100 and 1 + 10/100, so nu and nu_inf do not move by
# one ulp. The %/yr readings stay on the calibration cards as the side mention.
#
# They are also what the DEFAULT MENU ROW carries (bind_live_defaults, via model.py), so [choose]
# on it restores the calibrated parameter exactly. That is D-109's bitwise rule applied at the
# anchor rather than at the forward image. In x/yr the two coincide anyway: the forward map is
# 10^{slope * speed}, and both anchors come back EXACTLY (the %/yr map ended in 100*(10^y - 1),
# which near 1.1 could not return 10.0 at all -- the 5-ulp gap D-118's rider had to document).
# ---------------------------------------------------------------------------------------------
VALUE_GROWTH_ANCHOR = 2.19         # x/yr today           -> nu     (= 119 %/yr)
VALUE_GROWTH_INF_ANCHOR = 1.10     # x/yr in the long run -> nu_inf (=  10 %/yr)


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
    # D-118/D-120: BOTH value dials are stated as the GROWTH RATE of AI-attributable value, and
    # inverted through the leader's capability speed at the date each one describes -- today for
    # nu (closed form, Gamma(0) + g_a), the horizon endpoint for nu_inf (simulated). D-133 keys
    # that rate in x/yr, like every other rate dial here. The KEYS keep the _growth suffix rather
    # than taking the _x one: the suffix convention marks what the dial MEANS (a growth rate of
    # value, read at a date) and renaming them again would move eight frozen fixtures for a
    # spelling. loss_half_gC is the same case in reverse -- named for its percent.
    't_value_growth':     'nu',      # value growth today, x/yr        nu = log10(m)/xdot^L(0)
    't_value_growth_inf': 'nu_inf',  # value growth long-run, x/yr     nu_inf = log10(m)/xdot^L(T)
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
    # D-120: the ratified x/OOM envelope, re-keyed to a growth RATE. The union of the three
    # constructions' 90% bands is [1.62, 2.84] -> [1.5, 3.0] x/OOM (TV5' ratified); mapped
    # through the DEFAULT t = 0 speed xdot^L(0) = log10(11.34) = 1.05461 OOM/yr that is
    # 53.36 .. 218.55 %/yr, rounded OUTWARD per the envelope rule -- to WHOLE PERCENT since the
    # D-118 rider (Pavel, 2026-08-02, "just round it to integer"), which made the dial's anchor a
    # round rate. D-133 re-keys the unit to x/yr and carries the same grid across affinely:
    # [53, 119, 219] %/yr IS [1.53, 2.19, 3.19] x/yr, and the 1 %/yr step is a 0.01 x/yr step.
    # The mode is the anchor, x2.19/yr = 119 %/yr (x2.1029/OOM here).
    # Triangular (not uniform) so a user narrowing the range keeps the pooled mode.
    # The BOUNDS are re-derived numbers, not a re-ratified calibration: nothing about the
    # evidence moved, only the unit the dial states it in and the grid it states it on.
    't_value_growth':     ('triangular', 1.53, 2.19, 3.19),    # x/yr, today
    # D-107 (the default, Amodei's floor granted adversarially) + D-109 (the span) + D-118/D-133
    # (the unit). [x1.00, x1.30]/yr IS [0, 30] %/yr IS D-109's [x1.00, x1.75] PER OOM -- 30 %/yr
    # is x1.739/OOM at the default path -- so D-109's ruling is implemented here, not bypassed:
    # stagnation at the floor, Davidson's explosive threshold at the ceiling, and every menu row
    # reachable in between. Stated as a RATE the same span is granted at every compute path
    # rather than at one; the x1.00 floor is exactly no growth, and the dial cannot go below it.
    't_value_growth_inf': ('uniform', 1.00, 1.30),             # x/yr, long-run
    # rows 1-7 span 4 .. 10; the ratified MC CI is [4, 12] -> the envelope IS [4, 12].
    't_lag_mo':    ('lognormal', float(np.log(np.sqrt(4.0 * 12.0))),
                    float(np.log(12.0 / 4.0) / 3.29)),         # ~90% CI [4, 12] months
    # Hobbhahn's own CI [1.27, 1.54] rounded out; the model's spot 1.38 sits inside it.
    't_price_x':   ('uniform', 1.25, 1.55),                    # x/yr
    # D-129. The menu union EXACTLY, both ends: 1.30 (the broadest hardware reading, real-dollar
    # whole-node, with a budget that stops growing) and 1.38 x 1.10 = 1.5184226910631733 (the
    # trusted hardware leg with the budget at Amodei's own 10 %/yr floor). The ceiling is written
    # as the row's exact double rather than the ratified box's 4-dp gloss 1.5184, because under
    # D-128 the envelope end IS the witnessing row -- 1.5184 rounds it DOWN and would put the row
    # a hair outside the dial that offers it. Displayed "1.52" by _fmt3 either way.
    #
    # WAS [10^0.05, 10^0.30] = [1.1220, 1.9953], and the narrowing is the correction rather than
    # a side effect: the old top asserted a real training budget compounding at 10^(0.30-0.14) =
    # +44.5 %/yr FOREVER and the old bottom -18.7 %/yr, neither of which is a coherent asymptote.
    # Both were set before the two-leg decomposition existed, and the 1.12 bottom's own
    # justification (a grid-convergence story) had been withdrawn as construct-invalid with no
    # replacement. Across the new envelope the implied budget growth runs -5.8 % .. +10.0 %/yr,
    # so every point of the dial now names an economically readable scenario.
    't_floor_x':   ('uniform', 1.30, 1.5184226910631733),      # x/yr

    # D-098, PERCENT. NOT a chosen box: this IS the union of the adoptable alpha rows' intervals
    # with the left bound rounded DOWN and the right rounded UP, exactly per Pavel's envelope
    # rule (2026-07-26). display_only rows do not enter the union. test_09 asserts both that it
    # contains the union and that it is TIGHT, so the derivation cannot silently rot into an
    # arbitrary box.
    #
    # D-125 MOVED IT, [22, 45] -> [27, 47], and the move is a RE-STATEMENT of the same evidence
    # rather than a re-vetting of it. Every level/cost-share row is natively a Cobb-Douglas
    # object (under CD the output elasticity equals the cost share), so each row now converts
    # through its OWN construct's eta -- loss = 1 - 2^(-alpha) for the level rows, loss = alpha/2
    # for the growth row, whose bracket IS the eta = 1 weighted average. Not one source's alpha
    # changed; only the observable each alpha implies. The union goes [23.5, 44.5] -> [27.80,
    # 46.04]: it NARROWS on the left (Cottier's low CI end) and WIDENS on the right (Gundlach's
    # bound), so the envelope had to move in both directions at once.
    'loss_half_gC': ('uniform', 27.0, 47.0),                   # %
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
#   phi_RD, ell    -- REMOVED FROM THE MODEL OUTRIGHT (D-127: phi_RD under D-126, ell under
#                     D-123). phi_RD had already left this dict earlier, being inert under bill
#                     anchoring (it cancelled exactly inside cost_flow); ell was drawn here until
#                     D-127, which is why removing it re-phases the legacy draw stream (see the
#                     entry's note there),
#   kappa, B0      -- retired outright by D-093 (see the money block in Params).
# The money side draws ONE dimension and always did: the LEVEL is provably irrelevant (scaling
# earnings and cost by any lambda leaves the crossing year and the verdict exactly unchanged),
# so only the cost-to-earnings ratio carries uncertainty (SB6). Since D-093 that dimension is
# rho ITSELF rather than k_build standing in for it. Its entry below is in FRACTION units,
# because this dict is in parameter units; the WIDGET dials and crops the same dimension in
# percent through an app-side overlay (ui/state.py APP_RANGES['cov0']) and ui/mc.mc_prepare
# converts the crop. Two representations of one envelope, never two dimensions.
PARAM_RANGES = {
    # D-109 hugged the target envelope to its surviving menu union, [2.5, 5.3] x/yr -- 6.0 "was
    # witnessed by no row". That landed on TARGET_RANGES['t_compute_x'] and nowhere else, so this
    # twin kept sampling compute growth up to x6.0 through the legacy whole-parameter path. Bounds
    # are the exact image: log10(2.5) = 0.397940, log10(5.3) = 0.724276.
    'g_C0':     ('uniform', 0.39794, 0.724276),  # image of t_compute_x envelope [2.5, 5.3] x/yr
    # D-129: the image of t_floor_x's new envelope [1.30, 1.5184226910631733] x/yr, rounded
    # OUTWARD at 6 dp exactly as g_C0's and g_p's twins are (log10 of the ends is 0.11394335 and
    # 0.18139269). Was [0.05, 0.30] -- the parameter-space twin of the retired [1.12, 2.00] dial.
    'g_C_inf':  ('uniform', 0.113943, 0.181393),
    't_mid':    ('uniform', 0.7, 7.0),  # D-082: image of the old xi in [0.1, 1.0] under
                                        # the half-decay map t_mid = ln2/xi, rounded out
    # D-084 POSITION dials (PERCENT of the transition already done at u = 0), one per use of the
    # universal curve. The envelopes below are PROPOSALS, FLAGGED for the calibration round: from
    # 1% (the convention D-082 baked in -- "the transition has barely started") to 25% (visibly
    # under way). 50% is excluded BY CONSTRUCTION: it would put the midpoint at u = 0, and more
    # would put it in the past, contradicting what the midpoint dial means. All three are POINT
    # defaults in the MC (deliberately absent from SIM_DEFAULT), so the default fans are
    # untouched until a calibration pass widens them -- exactly how the asymptotic value dial
    # entered (it stayed out of SIM_DEFAULT until D-118 gave it Amodei's band).
    'p0_c':     ('uniform', 1.0, 25.0),
    'p0_w':     ('uniform', 1.0, 25.0),
    'p0_F':     ('uniform', 1.0, 25.0),
    'g_a':      ('uniform', 0.0, 0.9),           # residual g_eff - g_C0; floored at 0 (see
                                                 # invert_targets). Widget draws t_eff_x instead.
    # D-098: RE-WIRED, not deleted. This is the ratified alpha range [0.45, 0.90] in PARAMETER
    # units (so D-125's eta flip leaves it alone), and it is the envelope for the LEGACY
    # whole-parameter path only
    # (`sample_params`, which iterates all of PARAM_RANGES). The widget never reads it: it draws
    # the OBSERVABLE loss_half_gC from TARGET_RANGES and derives alpha per draw at that draw's
    # eta. Deleting the row would have silently pinned alpha at p_base in the legacy path, which
    # is worse than a slightly redundant envelope -- the same reason g_C0 keeps a parameter-space
    # range beside its target.
    'alpha':    ('uniform', 0.45, 0.90),
    # D-125: eta STOPS BEING A CHOICE. The discrete rail existed to make an eta -> -inf endpoint
    # selectable; R8 removed Leontief and D-122 removed the arbitrary -2 stand-in, leaving nothing
    # a continuous dial cannot express. This entry is that dial's ENVELOPE, and it is the union of
    # the eta menu's own rows (source_span('eta') == dist_bounds of this, bitwise):
    #   +1.00  the CES family's mathematical CEILING -- sigma = 1/(1-eta) is negative above it --
    #          and where D-018's outgoing default sat, so it must stay reachable;
    #   -1.20  the most complementary value any cited publication states as a NUMBER (the floor of
    #          the economy-wide CES range Davidson cites, algo-049). Anything below it is the
    #          eta -> -inf territory D-122 removed for being unwitnessed, so the envelope's left
    #          end and D-122's ruling agree rather than conflict.
    # TRIANGULAR with the mode at the ratified spot, the shape nu and loss_half_gC already use for
    # this situation: uniform over [-1.2, 1.0] would put equal mass on perfect substitution and
    # strong complements, which no source supports. This range is read by the LEGACY
    # whole-parameter path only (`sample_params`) -- the app never samples eta (it is in no
    # LEVEL_RANGED list and has no SIM_DEFAULT entry), which is Pavel's own ruling for it:
    # "There won't be interval, MC uses spot value for this parameter".
    # (D-122's deferred edit -- dropping -2.0 from the retired choice list -- lands here, in the
    # port round it was deferred to, together with the re-freeze it was deferred for.)
    'eta':      ('triangular', -1.2, 0.0, 1.0),
    # D-132 (Pavel, 2026-08-03, B3). WAS ('uniform', 0.1, 0.5), and that envelope had to go
    # whatever else was ruled: it excludes EVERY frontier-lab reading in the evidence (the
    # cross-lab controlled trial's floor 0.04, the survey median 1.00, the telemetry 1.83, the
    # research-staff poll 3.00) and it cannot reach the L3 profit sign flip, which makes the
    # fan's apparent decisiveness on question (a) an artefact of the envelope rather than a
    # finding. [0.04, 3.00] is the menu union across both tiers, both ends witnessed by rows.
    #
    # THE FAMILY IS THE BRIEF'S POINT, not a detail: the beta0 evidence is MULTIPLICATIVE and
    # its spread is a factor (a 75x envelope, a 15x disagreement between two readings of the
    # same population months apart), so a uniform would put half its mass above 1.5 on a dial
    # whose ratified spot is 0.3. lognormal_from_ci puts the ratified ENDS at the 90% CI and its
    # median at sqrt(0.04*3.00) = 0.3464, a hair above the held placeholder.
    # DEVIATION, recorded rather than smoothed: brief v2 sec.11 proposes a LOG-TRIANGULAR on
    # psi(0) in [1.04, 4.00] with mode 2.00. That shape is in no ratify-box item (the box runs
    # B1 construct / B2 spot / B3 envelope / B4 menu -- there is no B-item for the draw), the
    # codebase has no log-triangular family, and its mode beta0 = 1.00 would sample against the
    # spot B2 explicitly HELD at 0.3 -- the very defect D-118's own note rejects for nu_inf.
    # The lognormal keeps sec.11's multiplicative principle in the grammar the model already has.
    # Spelled longhand rather than as lognormal_from_ci(0.04, 3.00) for one mechanical reason:
    # that helper is defined BELOW this dict. Same arithmetic, same idiom Delta0 already uses
    # here, and the ends stay DERIVED from the two ratified literals rather than hand-typed.
    # ONE ULP OF SLACK, stated rather than papered over: dist_bounds inverts sigma through 1.645
    # while the fit divides by 3.29, so the round-trip returns (0.040000000000000008,
    # 2.9999999999999996) instead of the ratified ends. That is 3e-17 in rail fractions against
    # calpanel's 1e-9 adoptability tolerance, so both witnessing rows stay clickable and _fmt3
    # renders "3.00" either way. The D-129 fix -- carry the row's exact double -- is unavailable
    # for a lognormal, whose ends are derived from (mu, sigma) rather than written into it.
    'beta0':    ('lognormal', np.log(np.sqrt(0.04 * 3.00)), np.log(3.00 / 0.04) / 3.29),
                                        # D-084: renamed from rho0 (rho = coverage)
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
    # image of the ratified value envelope [1.5, 2.1, 3.0] x/OOM in PARAMETER space -- the
    # legacy whole-parameter path only (`sample_params`). D-120 re-keyed the DIAL to a growth
    # rate and D-133 keyed that rate in x/yr; this row is unchanged through both, because the
    # parameter-space image of the same evidence is the same numbers, and the widget never reads
    # it (it draws t_value_growth and inverts per draw).
    'nu':       ('triangular', 0.176091, 0.322219, 0.477121),
    'rho':      ('uniform', 0.26, 30.1 / 65.0),  # coverage today, the ONE money dimension
                                          # (D-093). The vetted [26, 46.3]% envelope in fraction
                                          # units -- the ceiling carried as the fraction 30.1/65
                                          # for the same reason Params.rho is 25.1/75: it is a
                                          # DERIVED ratio, and a hand-typed decimal would be an
                                          # ulp away from the menu row that witnesses it (D-128
                                          # re-added that row; the app-side twin in ui/state.py
                                          # APP_RANGES carries the same number times 100). D-104
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
    # D-132 (Pavel, 2026-08-03, XM2 "Round"). WAS ('uniform', 2.0, 20.0), an envelope the app's
    # own card admitted was "kept provisionally from the old bend". The top half of it was dead:
    # at x_mid = 12 the value transition is 8.7% complete at the horizon, and from 12 to 20 the
    # dial moves L2 cumulative profit by ~6%. Past ~12 the parameter is not IDENTIFIED at all --
    # only the horizon-average slope is, so a high x_mid cannot be told from a re-labelling of
    # nu. Narrowing therefore costs nothing and buys back resolution where the verdict lives.
    #
    # THE UNION RULE, AND PAVEL'S EXPLICIT DEPARTURE FROM IT (D-128 clause 2). The menu's
    # choosable rows span [2.7, 12.0] once row (f) is display_only; strict union would give a
    # floor of 2.7, and the brief's own proposal was [1.7, 12] with row (f) CHOOSABLE at 1.7.
    # Pavel ruled ROUND endpoints, so: the floor is 2.0 -- BELOW the union, never above it, so
    # no choosable row or default CI sits outside the dial -- and row (f)'s 1.7 moves to
    # display_only carrying that as its `why`. This is a recorded rounding choice, not a drift:
    # rounding a floor DOWN is the safe direction under the standing envelope rule ("sliders
    # slightly wider than the envelope of the CIs / spot values", Pavel 2026-07-26), and it is
    # what keeps D-128's real guarantee -- choosable <=> reachable -- true.
    'x_mid':    ('uniform', 2.0, 12.0),
    # ('ell' sat here until D-127, as ('lognormal', ln sqrt(0.25*1.3), ln(1.3/0.25)/3.29) -- EL3',
    # ~90% CI [0.25, 1.3] yr. Removing it shifts the draw order of everything after it, which is
    # the whole of the legacy re-phasing D-127 records. The envelope is kept in the archive note.)
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
    # D-120: re-keyed to a rate at the default t = 0 speed -- [1.7, 2.65] x/OOM is [75.0, 179.5]
    # %/yr, and the mode is the spot. Same band, same evidence, stated as a rate. D-118 rider
    # (2026-08-02): the ends go to whole percent with the dial's grid, [75, 179], and the mode
    # IS the anchor. D-133 states all three in x/yr -- [1.75, 2.19, 2.79] is that same band.
    # THE DRAW MOVES WITH THE DIAL, deliberately (D-133): the sampler draws in the dial's own
    # units, so there is exactly one band literal per dimension. The RNG stream does not
    # re-phase (a triangular draw consumes the same uniforms either way, asserted bit-for-bit
    # in test_model), but the affine image of a draw differs from the draw of the affine image
    # in the last place, so the MC fixtures re-freeze at ~1 ulp.
    't_value_growth': ('triangular', 1.75, 2.19, 2.79),        # x/yr
    # D-118: the asymptotic dial gets the SIM_DEFAULT it never had. Until now it had no entry,
    # so ui/state.py fell back to the full envelope and the Monte Carlo swept it -- putting most
    # of its mass on rates no publication holds. Amodei's own published band [10, 20] %/yr is
    # the ratified default draw, matching D-107's steelman (the dial's spot is that band's
    # floor). The precedent is t_compute_x, whose SIM_DEFAULT is Epoch's own [3, 4].
    # (Brief 11's NV4 proposed triangular(0, 2.5, 30) instead -- the mainstream-trend mode. That
    # would put the sampler's mode at 2.5 %/yr while the dial sits at 10, i.e. sample against
    # the steelman the same panel grants. The menu keeps the mainstream rows clickable.)
    # D-133: [10, 20] %/yr in the dial's x/yr units.
    't_value_growth_inf': ('uniform', 1.10, 1.20),             # x/yr
    # TL5(b): lognormal, 90% CI [4, 12] months, median ~7.
    't_lag_mo':    lognormal_from_ci(4.0, 12.0),
    # TC2(b) rider: the external four-model bracket on hardware price-performance.
    't_price_x':   ('uniform', 1.30, 1.45),
    # D-129: the compute FLOOR gets the SIM_DEFAULT it never had, and with it a place in the
    # default fan. Until now it had no entry, so it was a POINT default -- pinned at the spot,
    # and the moment a user ticked it on the fallback swept the whole envelope FLAT, which is
    # what the ratified box replaces. Triangular so a user who narrows the range keeps the mode
    # (the t_value_growth precedent), and because the shape is right on the merits: the floor is
    # a SUM of two roughly-flat legs, whose convolution is triangular-ish rather than flat. The
    # mode is the calibrated spot, 10^0.15, bitwise. Ends are the envelope's own.
    # NOTE this is a NEW DRAWN DIMENSION at Level 2+, not a re-shaping of an existing one --
    # see D-129, which measures what it does to the fan.
    't_floor_x':   ('triangular', 1.30, 1.4125375446227544, 1.5184226910631733),   # x/yr
    # D-132 (Pavel, 2026-08-03, XM4): the GATE gets the SIM_DEFAULT it never had. Until now
    # x_mid was a POINT default -- absent from SIM_DEFAULT, so `_default_sampled` left its tick
    # OFF and the sampler pinned it at the spot; only a user who ticked it on got the flat
    # envelope sweep. (Brief v2 sec.1.3 says instead that "the default fan sweeps all of
    # [2, 20]"; that is wrong about the app, and the correction matters because the change here
    # is ADDING A DRAWN DIMENSION, not re-shaping an existing one.) This is the third time the
    # same hole has been closed -- D-118 for nu_inf, D-129 for the compute floor -- and it is
    # being closed last on the dial D-107 calls THE GATE, the only dial in the value block whose
    # envelope contains a sign flip of the Level-2 verdict (measured at x_mid = 3.978).
    # TRIANGULAR, ends the envelope's own and MODE THE CALIBRATED SPOT, which is the house shape
    # for exactly this situation (t_value_growth, t_floor_x, loss_half_gC) and is right on the
    # merits here: the evidence clusters at 6 (two chains sharing no input land within 0.12) and
    # thins toward both the Davidson value-share floor and the no-bend marker.
    # NOT uniform(4.9, 7.5), GATE's own published band -- the cleanest choice by the nu_inf
    # precedent and the WRONG one here, because that band's floor sits above the highest
    # break-even midpoint (4.19 at r_inf = 0), so it would put EXACTLY ZERO mass on the region
    # where the headline question is in doubt and make a fan the paper leans on read as settled.
    'x_mid':       ('triangular', 2.0, 6.0, 12.0),             # OOM above the 2026 frontier
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
    # D-125 moved all three ends with the envelope and the default row. The mode is the ratified
    # spot's EXACT IMAGE at the new base eta = 0 (alpha = 0.70 => 38.4428...%), not a rounded
    # 38.44 typed beside it: a decimal an ulp away from the calibration it stands for is the
    # two-literals trap, and t_value_growth / t_floor_x carry their modes the same way.
    'loss_half_gC': ('triangular', 27.0, 100.0 * (1.0 - 2.0 ** -0.70), 47.0),   # %
}
