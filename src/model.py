"""Frontier-AI-lab competition — the model.

**The single source of truth for all model code** (D-025). The Streamlit widget holds no model
math: `ui/model_access.py` imports this module and hands its functions to the UI. Illustrations
live in `model_demo.py`.

**This file is a FACADE** (D-110). The ~1500 lines of math it used to carry were split into
cohesive plain-Python modules, and everything here is a re-export, named one at a time so the
public surface is a written list rather than a side effect of `import *`. `import model as m` and
every `m.<name>` in `app.py`, `ui/*`, the tests and the capture harnesses reach exactly what they
always did:

    model_params.py       the Params dataclass + calibrated defaults, the target and parameter
                          envelopes, the tight default simulation ranges  (imports numpy only)
    model_dynamics.py     components A and B: the universal curve Gamma, compute growth, psi,
                          the CES research aggregate, the two algorithmic laws, c^L in closed form
    model_profit.py       components D and C: the value index W, the cost flow, the RK4
                          integrator + `simulate`, and the question-(a) statistics
    model_calibration.py  targets <-> parameters, the stationary-catchup re-anchor, `base_params`
                          and the import-time self-checks
    model_montecarlo.py   joint sampling, the bill-coherence filter, `monte_carlo`,
                          `mc_draw_batch`
    cal_sources.py        the calibration source menus (declarative evidence, D-110 first move)

The layering is acyclic and in that order, which is what makes the eventual JS/TS port (D-114) a
translation rather than a redesign.

**Where the equations live.** The authority on the model's math is what the widget *renders*
(`ui/equations.py`, the "Equations & calibration" pane), with `paper/draft_v3.tex` as the written
companion. Each function's own docstring carries its specification. The acceptance tests and the
harvest condition (N5) live in `tests/test_model.py`.
"""
# Re-exported for the `from model import *` surface `model_demo.py` relies on, unchanged since
# the file carried the math itself: numpy and the dataclass helpers were reachable as `m.np`,
# `m.replace` and friends, so they stay reachable.
import numpy as np                                                       # noqa: F401
from collections import namedtuple                                       # noqa: F401
from dataclasses import dataclass, field, replace                        # noqa: F401

from model_params import (                                               # noqa: F401
    G_C_TODAY, Params,
    VALUE_GROWTH_ANCHOR, VALUE_GROWTH_INF_ANCHOR,
    TARGET_PARAM, TARGET_RANGES, PARAM_RANGES, SIM_DEFAULT,
    lognormal_from_ci, dist_bounds,
)
from model_dynamics import (                                             # noqa: F401
    ALPHA_LOSS_CEILING, GammaShape, P0_MAX_PCT, P0_MIN_PCT,
    _GC_TODAY_CACHE, _ces_bracket, _logistic, _softplus,
    algo_growth_F, algo_growth_L, alpha_from_loss, c_L_closed, compute_growth,
    follower_compute_growth, gamma_curve, gamma_shape, gc_today, loss_from_alpha,
    psi, psi_boost_share, slope_span,
)
from model_profit import (                                               # noqa: F401
    COHERENCE_TOL_OOM, W, W_exp_approx, _rk4_follower_fine, _rk4_leader_fine, _sim_pad, _xR_of,
    coherent_x_mid, conduct_mult, cost_flow, delay_comparison, gap_index, headline,
    implied_value_multiple, leader_horizon_state, simulate, value_coherence, w_log,
)
from model_calibration import (                                          # noqa: F401
    _DELTA_ALGO_SHARE, _DELTA_DEV_DEFAULT, _PB, _SB, _TD0, _XDOT_T_CACHE, MERGED_DELTA_RANGE,
    base_params, channels_from_lag, growth_mult_of_slope, invert_targets, slope_of_growth_mult,
    split_delta, stationary_catchup, target_defaults, xdot_L0, xdot_L_T,
)
from model_montecarlo import (                                           # noqa: F401
    BILL_COHERENCE, FLOOR_COHERENCE, _COHERENCE_TRIES, _VALUE_TARGETS, _draw_dict, _draw_one,
    bill_coherent, bill_growth, floor_budget_growth, floor_coherent,
    mc_draw_batch, monte_carlo, sample_params,
)

# The source MENUS live in `cal_sources.py` since D-110. Calibration evidence is declarative data,
# and a module that holds nothing else is what turns the eventual JS/TS port into a serialization
# rather than a rewrite. Every name is re-exported here, so `import model as m` still reaches the
# menus as `m.CAL_SOURCES` / `m.source_span(...)` and not one consumer line had to change.
from cal_sources import (                                            # noqa: E402,F401
    CAL_SOURCES, source_span, bind_live_defaults,
    _LAG_SOURCES, _COVERAGE_SOURCES,
    _ALPHA_ETA_NOTE, _ALPHA_DEFAULT, _ALPHA_LEVEL, _ALPHA_GROWTH, _ALPHA_BOUND,
)


# A menu row whose value IS a live model default carries the default rather than a second copy
# of it (D-091). cal_sources.py imports nothing, model.py included, so the binding is made here —
# the first point where both the table and the model are in scope.
#
# D-120 adds the two value dials to the binding for a reason D-109 states as a rule: the DEFAULT
# row must carry the dial's value bitwise, so that clicking it restores the calibrated spot
# exactly. In ×/OOM the pooled row could hold a literal 2.1; as a growth rate the same spot was
# 118.68…%/yr, a number that depends on the leader's own t = 0 and horizon speeds, so a literal
# here would have been a second copy of a derived quantity — exactly the two-literals defect γ's
# row hit.
#
# D-118 RIDER (Pavel, 2026-08-02) INVERTS THE DERIVATION and with it what belongs on the card. The
# ruled observables are the round rates — ×2.19/yr and ×1.10/yr since D-133 re-keyed the unit,
# 119 and 10 %/yr in the unit they were ruled in — and ν / ν_∞ are THEIR images, so the anchors
# are the primitives and binding them here is not a second literal, it is the only one. `_TD0` is
# asserted to reproduce them (test_model), which is what keeps this honest. Under D-133 BOTH legs
# round-trip bitwise: the ×/yr forward map is 10^(ν·ẋ) and returns 2.19 and 1.10 exactly, where
# the %/yr map's 100·(10^y − 1) could not land on 10.0 at all and left the asymptotic leg 5 ulp
# out. Binding the anchor puts the ruling on the card and restores ν_∞ bitwise either way.
bind_live_defaults(gamma=Params().gamma,
                   t_value_growth=VALUE_GROWTH_ANCHOR,
                   t_value_growth_inf=VALUE_GROWTH_INF_ANCHOR,
                   alpha=Params().alpha, eta=Params().eta)
