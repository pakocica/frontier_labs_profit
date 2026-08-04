"""The observable <-> parameter maps, the catch-up re-anchor rule, and the base model.

Targets-first parameterization (D-037): wherever a parameter has a clean observable, the
*observable* is the primitive. `target_defaults` is the forward map Params -> observables and
`invert_targets` the inverse at t = 0; `stationary_catchup` supplies the degree of freedom that
keeps the gap stationary at t = 0 under every dial setting; `base_params` builds the calibrated
Level-1 model and the import-time self-checks below prove the two money identities on the model's
own forward path.

The slider bounds and Monte-Carlo distributions these maps run over live in `model_params`
(`TARGET_RANGES`, `PARAM_RANGES`), which is data and imports nothing.
"""
import numpy as np
from dataclasses import replace

from model_params import Params
from model_dynamics import (
    algo_growth_L, alpha_from_loss, compute_growth, follower_compute_growth, gc_today,
    loss_from_alpha,
)
from model_profit import gap_index, leader_horizon_state, simulate


# D-093: `gap_index` MOVED to the value block -- simulate calls it now, so it belongs beside W
# rather than among the calibration helpers. Since D-110 that block is `model_profit`, which is
# where this module imports it from.
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


_XDOT_T_CACHE: dict = {}

def xdot_L_T(p):
    """The leader's capability speed at the HORIZON ENDPOINT t = T (D-120), the denominator the
    ASYMPTOTIC value dial converts through:

        xdot^L(T) = g_C(T) + adot_L(T, x^L(T)),   with x^L(T) SIMULATED.

    WHY T AND NOT A TRUE ASYMPTOTE. Past the horizon the psi feedback diverges (spec N4), so
    "the long-run capability speed" has no limit to read; the horizon endpoint is the last speed
    the model actually states. At the shipped defaults it is 0.474 OOM/yr (the leader decelerates
    from 1.055 today as the compute slowdown bites, troughing near 0.43 around t = 5 before psi
    lifts it back).

    THE ORDERING THIS IMPOSES, and why it is not circular: the leader's path is a function of the
    compute and algorithmic legs alone -- nu and nu_inf enter nowhere in it -- so `invert_targets`
    settles the capability legs FIRST and reads this afterwards. Reversing the two would be the
    circularity, and the function order below is what prevents it.

    Memoised on the complete set of fields the leader path depends on, so a Params differing
    anywhere else (or agreeing here) is correctly treated as identical. The sidebar inverts
    several times per rerun at one context and hits the cache every time; the Monte Carlo draws a
    fresh context per draw and misses by construction, which is why `leader_horizon_state`
    integrates the leader only and short-circuits exactly where the path is not read at all."""
    key = (p.T, p.dt, p.g_C0, p.g_C_inf, p.t_mid, p.p0_c, p.g_a,
           p.A1, p.alpha, p.eta, p.leontief, p.beta0, p.gamma)
    v = _XDOT_T_CACHE.get(key)
    if v is None:
        if len(_XDOT_T_CACHE) > 4096:          # bound it: one entry per distinct MC draw
            _XDOT_T_CACHE.clear()
        t_T, x_T = leader_horizon_state(p)
        v = float(compute_growth(t_T, p) + algo_growth_L(t_T, x_T, p))
        _XDOT_T_CACHE[key] = v
    return v


# D-120: the value dials are stated as GROWTH RATES of value, and converted through the leader's
# capability speed at the date each one describes -- today for nu, the horizon endpoint for
# nu_inf. The two helpers below are that conversion, written once and used in both directions so
# `target_defaults` and `invert_targets` cannot drift apart.
#
# D-133 (Pavel, 2026-08-03) RE-KEYED THE UNIT from %/yr to x/yr, for consistency with the compute
# dials, which all speak x/yr. He agrees %/yr reads more naturally for small growth rates and
# prefers the consistency; the %/yr correspondence is kept as a side mention on the calibration
# cards. The map is affine and exact at the integer anchors -- x/yr = 1 + (%/yr)/100 -- so this
# changes the unit and NOTHING else: nu and nu_inf are bitwise what they were.
#
#     m x/yr  ->  slope = log10(m) / xdot^L                (value-OOMs per capability-OOM)
#     slope   ->  m = 10^{slope * xdot^L}
#
# The x/yr form is also the cleaner arithmetic, measured rather than assumed: the retired %/yr
# forward map ended in 100*(10^y - 1), which near 1.1 could not land on 10.0 at all (the reachable
# doubles straddling it are 9.999999999999987 and 10.000000000000009), so D-118's rider had to
# record the asymptotic dial's round trip as 5 ulp. In x/yr the forward map IS 10^y and both legs
# round-trip EXACTLY: 2.19 and 1.10 come back bitwise.
#
# WHY THE OBSERVABLE IS THE RATE (D-118, Pavel). A x/OOM dial states a rate only jointly with a
# compute path: the shipped x1.25 granted 6.9 %/yr at a slow path, 11.2 % at the default and
# 19.6 % at a fast one, so "we grant Amodei's 10-20 %/yr" was true at exactly one setting of dials
# the user is invited to move. Stated as a rate the same growth is granted at EVERY path by
# construction, which is what makes the paper's adversarial framing checkable rather than
# conditional on a compute setting the reader never chose.
_SPEED_FLOOR = 1e-9      # a zero speed is unreachable from any envelope; guard the division only
_MULT_FLOOR = 1e-12      # ditto for a zero multiplier: log10(0) is -inf, and x1.00 is the floor

def growth_mult_of_slope(slope, speed):
    """Value growth in x/yr implied by a value slope (per capability-OOM) at a capability speed."""
    return float(10.0**(float(slope) * float(speed)))

def slope_of_growth_mult(mult, speed):
    """The inverse: the value slope a x/yr reading implies at a capability speed. Domain
    mult > 0 (value does not vanish); the shipped envelopes floor at x1.00, i.e. no growth."""
    return float(np.log10(max(float(mult), _MULT_FLOOR)) / max(float(speed), _SPEED_FLOOR))


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
        # D-120/D-133: the two value dials are GROWTH RATES in x/yr, each read at the date it
        # describes -- nu at today's capability speed, nu_inf at the horizon endpoint's.
        't_value_growth':     growth_mult_of_slope(p.nu, speed0),
        't_value_growth_inf': growth_mult_of_slope(p.nu_inf, xdot_L_T(p)),
        't_floor_x':   float(10.0**p.g_C_inf),
        # D-098, in PERCENT (the p0_c convention: a percent dial states percent). Under
        # Leontief this reports the 50% the model actually delivers, not the weight's image.
        'loss_half_gC': float(100.0 * loss_from_alpha(p.alpha, p.eta, p.leontief)),
    }

_TD0 = target_defaults()


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
    resolved there, not here. stationary_catchup is now its ONLY caller: the app used to read
    the channel lengths off this function for the merged-delta methodology's doc numbers, and
    quoted a delta_eff 27% away from the one the model runs (audit A finding 4). The merged
    levels (1-2) never call it."""
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
    movement with the cost-SHAPE dials (t_mid, g_c_inf, and ell until D-127 removed it), which
    was pure parameterisation
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
    the lag; (delta_dev, delta_rel) from stationary_catchup (same exactness).

    THE ORDER IS LOAD-BEARING SINCE D-120, and in a new way. Three blocks, in this order:
      (1) THE CAPABILITY LEGS -- g_C_inf, g_C0, g_a, alpha. Everything the leader's own path
          depends on. `loss_half_gC` MOVED into this block: alpha enters no t = 0 rate (which is
          why it used to be documented as order-free) but it does enter adot_L at t > 0, so the
          horizon speed reads it.
      (2) THE VALUE LEGS -- nu, nu_inf -- which convert x/yr through the speeds block (1) has
          just fixed: xdot_L0 (closed form: Gamma(0) + g_a) and xdot_L_T (simulated).
      (3) THE LAG, which needs the exact t = 0 speed and re-solves the catch-up intensity.
    There is no cycle to break: no capability rate reads nu or nu_inf, so (1) never needs (2).
    g_p sits in (1) only because it is order-free and belongs with the other scalars."""
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
    if 't_price_x' in targets:
        out['g_p'] = float(np.log10(targets['t_price_x']))
    if 'loss_half_gC' in targets:
        # D-098. Order-free at t = 0 -- both CES channels are 1 there, so adot_L(0) = g_a at
        # every alpha -- but NOT order-free for the horizon speed: adot_L(t) reads alpha at
        # every t > 0, so D-120 moved this branch up into the capability block, ahead of the
        # value legs that convert through xdot_L_T. It reads the active eta, a free dial and
        # never a target, so `base` carries it.
        out['alpha'] = alpha_from_loss(float(targets['loss_half_gC']) / 100.0,
                                       base.eta, base.leontief)
    # ---- (2) the VALUE legs, converted through the speeds the block above has just fixed.
    ref = replace(base, **out)
    if 't_value_growth' in targets:
        # today's value growth, through the leader's EXACT t = 0 speed (closed form: Gamma(0)
        # + g_a, both already inverted above)
        out['nu'] = slope_of_growth_mult(targets['t_value_growth'], xdot_L0(ref))
    if 't_value_growth_inf' in targets:
        # the long-run rate, through the SIMULATED speed at the horizon endpoint (D-120)
        out['nu_inf'] = slope_of_growth_mult(targets['t_value_growth_inf'], xdot_L_T(ref))
    # ---- (3) the lag. `ref` is refreshed with the value legs for hygiene: nothing below reads
    # nu or nu_inf today, and a future mechanism that does must not silently see the base's.
    ref = replace(ref, **{k: out[k] for k in ('nu', 'nu_inf') if k in out})
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


# ---- base-model self-checks (D-076). These run at import, so an incoherent default can never
# reach the widget. The BASE = the full model with the later mechanisms pinned exactly as
# ui/levels.apply_level_pins does at Level 1.
def base_params(**kw):
    """The calibrated BASE model: constant compute growth, constant residual g_a, exponential
    value, merged catch-up. This is the Level-1 model the calibration round closed on 2026-07-26.
    (It used to pin two further extensions, the build lag ell and the R&D markup phi_RD, both at
    0; D-127 removed them from the model, so there is nothing left to pin.)"""
    # (D-088: g_C0 is no longer pinned here. It IS G_C_TODAY -- the dataclass default states
    # today's growth directly, so the old explicit pin, which existed only to undo D-086's
    # plateau-valued default, has nothing left to undo.)
    pins = dict(A1=True, gamma=0.0, x_mid=200.0, tau=0.0,
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
# D-118 rider (2026-08-02): re-fitted with nu, which is now the image of the round rate x2.19/yr
# rather than of x2.1/OOM (0.3664606328362475 before -- the gap rides nu and nothing else moved).
assert np.isclose(gap_index(_PB), 0.36699433065482256, rtol=1e-12)      # the value gap, in index units
# D-093: the two stale-literal guards on kappa and B0 are GONE WITH THE FIELDS. Nothing here
# needs re-fitting when a default moves -- E_0 = rho and B_0 = 1 are identities, and the two
# asserts below check them on the model's OWN forward path rather than on a stored constant.
_SB = simulate(replace(_PB, T=1.0))
assert np.isclose(_SB['revenue'][0], _PB.rho, rtol=1e-12)               # E_0 = rho, 33.5% coverage
assert np.isclose(_SB['cost'][0], 1.0, rtol=1e-12)                      # B_0 = 1 (today's bill)
