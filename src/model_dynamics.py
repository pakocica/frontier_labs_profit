"""Components A and B: the universal transition curve, the frontier engine, the follower.

The rate laws and the one closed-form integral the model's dynamics are built from — the
universal curve Gamma and its two identities, the leader's compute growth, the psi feedback, the
CES research aggregate and its alpha observable, the leader's algorithmic law, the follower's
compute growth and catch-up law, and c^L(t) in closed form.

This module knows nothing about value, cost or profit, and it never imports `model_params`: every
function takes the parameter bundle `p` as an argument and reads it by attribute, so the whole
component layer is one numpy dependency away from being ported as-is.

**Where the equations live.** The authority on the model's math is what the widget *renders*
(`ui/equations.py`, the "Equations & calibration" pane), with `paper/draft_v3.tex` as the written
companion. Prose here is deliberately minimal: a stale equation in a comment is worse than no
equation at all. Each function's own docstring carries its specification.
"""
import numpy as np
from collections import namedtuple


# ---------------------------------------------------------------- Component A -- frontier progress
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


def _logistic(u, y_minus_inf, y_inf, u_mid, s):
    """The RAW base-10 logistic, in plateau-and-slope form. PRIVATE: nothing outside this module
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

    Used by cost_flow (model_profit) to de-lag the normalised bill through the exact integrated
    path c^L(ell), never a g_C*ell linearization (Pavel, 2026-07-26: "use c(t+l), not
    g_C(t+l), as the growth might vary between t and t+l"). Agrees with simulate()'s RK4 path to
    integrator precision; in the base (g_C_inf = g_C0, constant growth) gamma_shape returns the
    plateau EXACTLY equal to g_C0, so the bracket multiplies zero and c^L(t) = g_C0*t exactly."""
    t = np.asarray(t, dtype=float)
    sh = gamma_shape(p.g_C0, p.g_C_inf, p.t_mid, p.p0_c)
    return sh.y_minus_inf * t + (p.g_C_inf - sh.y_minus_inf) * (
        _softplus(sh.k * (t - p.t_mid)) - _softplus(-sh.k * p.t_mid)) / sh.k


# ---------------------------------------------------------------- Component B -- follower catch-up
#
# What catches up is the follower's *algorithmic* level — compute cannot be copied. The two
# channels differ in what they run on: the **developed** channel ($\delta_{dev}$, espionage,
# papers, mobility) diffuses methods and so runs on the algorithmic gap, while the **released**
# channel ($\delta_{rel}$, distillation) copies what the served model can do and so runs on the
# capability gap. Both die at parity (N2, D-015). The observed open-vs-closed lag pins the
# **total** $\delta_{dev} + \delta_{rel}$; the *split* between them is the free parameter, and it
# is precisely what the parked question (b) turns on.


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
