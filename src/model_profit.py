"""Components D and C, the integrator, and the question-(a) statistics.

The value of capability as an index (`w_log`, `W`, `gap_index`, the II.4 conduct multiplier), the
leader's model-building cost flow, the fixed-step RK4 integrator that runs the whole system
(`simulate`), and the headline statistics read off a simulated path (`headline`,
`delay_comparison`).

The layering is deliberate and acyclic: `model_dynamics` supplies the rate laws, this module turns
a path into earnings, cost and profit, and `model_calibration` / `model_montecarlo` sit on top.
"""
import numpy as np
from dataclasses import replace

from model_dynamics import (
    _logistic, _softplus, algo_growth_F, algo_growth_L, c_L_closed, compute_growth,
    follower_compute_growth, gamma_shape, psi_boost_share,
)


# ------------------------------------------- Component D -- value of capability (an INDEX),
# ------------------------------------------- and the II.4 conduct multiplier
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

    (Lives here, beside W, rather than among the calibration helpers where it sat until D-093: simulate
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


# ------------------------------------------------------------------ Component C -- cost flow
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


# ------------------------------------------------------------- Numerical integration (N1/N6)
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


# ------------------------------------------------------------------ Outputs -- question (a)
#
# Does the leader's profit flow turn positive within ~5 yr, and does it *stay* positive? The
# **reported** outcome in the widget is the coverage ratio $\rho_t = E_t / B_t$ (earnings over
# model-building cost), which is break-even at 1 and, given $\rho_0 = m/k$, invariant to $R_0$
# and $m$ separately — the money triple's only identified combination (D-080). NPV is a secondary
# statistic. `delay_comparison` sweeps the parked release delay $\tau$ (D-077).


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
