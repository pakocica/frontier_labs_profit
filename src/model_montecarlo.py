"""Monte Carlo — joint draws from the calibration ranges.

`sample_params` draws from `PARAM_RANGES` **jointly**, independently per parameter, and
`monte_carlo` runs `simulate` on each draw. Draws whose implied bill growth falls outside
`BILL_COHERENCE` are rejected and redrawn: the compute-growth and price-decline draws jointly
imply a bill-growth rate, and an incoherent pair would be a scenario no observer has seen.

`sample_params` is the legacy whole-parameter path (every `PARAM_RANGES` key); `mc_draw_batch` is
what the widget runs, drawing in TARGET space wherever a target exists and inverting per draw. The
envelopes themselves live in `model_params`.
"""
import numpy as np
from dataclasses import replace

from model_params import PARAM_RANGES, TARGET_RANGES
from model_calibration import invert_targets
from model_profit import headline, simulate


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
