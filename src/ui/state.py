"""Session-state infrastructure: the reset registry (keyed widgets, seed-if-absent),
the spot shadow store (sv_* keys) and trim-crop helpers (D-079), the user-edited MC
sampling-range override store (_range_over, D-040/41/42), distribution bounds/fitting
utilities, and the per-run level()/mc_active() reads.
The GC-proof patterns here fix real Streamlit bugs — see D-041/D-042 implementation notes.
"""
import numpy as np
import streamlit as st

from .levels import LEVEL_LABELS
from .model_access import m, P0, TDEF


def level():
    """The active level, read from session state (seeded so the selector widget can bind
    to the same key later in the run without a default= conflict). Values outside the current
    ladder (e.g. a stale "2 · Training in advance" from before the D-081 merge, or a stale
    "7 · …" from before the D-044 cap) clamp to Level 1."""
    if st.session_state.get("level") not in LEVEL_LABELS:
        st.session_state["level"] = LEVEL_LABELS[0]
    return int(st.session_state["level"].split(" ", 1)[0])


def mc_active():
    """True while the mode switch (atop the charts panel since round 2) is on Monte Carlo:
    the trim lanes render and the chart tiles show the MC fans. Accumulation itself runs in
    EITHER mode (background precalc)."""
    if "mode" not in st.session_state:
        st.session_state["mode"] = "Point forecast"
    return st.session_state.get("mode") == "Monte Carlo"


# ---- docked calibration panel (D-043, variant A2) ------------------------------------------
# The open state is a plain session key holding the PARAMETER key whose details are shown
# ("delta_total" for the merged-δ card); None/absent = closed. It survives reruns — unlike the
# old st.dialog, which any button click closed (the bug D-043 fixes).
def cal_open():
    """The parameter key the docked calibration panel is showing, or None."""
    return st.session_state.get("_cal_open") or None


def open_cal(key, pinned=False):
    """ⓘ / [details] callback: open the docked panel on this parameter."""
    st.session_state["_cal_open"] = key
    st.session_state["_cal_pinned"] = bool(pinned)


def close_cal():
    st.session_state["_cal_open"] = None
    st.session_state["_cal_pinned"] = False
    st.session_state.pop("_cal_scrolled", None)   # re-arm the panel's one-shot autoscroll


def toggle_cal(key, pinned=False):
    """» rail callback (Pavel: the button should "both open and close"): open the docked
    panel on this parameter; a second click on the SAME row's » closes it; another row's »
    switches the panel to that parameter."""
    if st.session_state.get("_cal_open") == key:
        close_cal()
    else:
        open_cal(key, pinned)


# (the right chart panel's collapse/width moved CLIENT-side in D-050 — theme.inject_frontend_js
# owns it via localStorage + a root CSS variable, so no server state exists for it anymore; the
# panel column itself renders every run and the MC component stays mounted by construction)


# ---- D-078 parameter-row accordion + equation sync -----------------------------------------
# `_p_open` holds the ONE expanded sidebar row's container key ("row_t_compute_x"), or None —
# an accordion by construction (opening a row is a plain overwrite). Clicking a row title also
# focuses the subsection that carries the parameter: `_eq_focus` (read by
# equations.visible_subsections, so the concise view reveals it WITHOUT flipping the show-all
# toggle) and `_eq_focus_bump` (a click counter — the focused expander force-opens and the
# pane autoscrolls once per bump, not on every rerun). The old forced Introduction→Equations
# pane switch retired with the pane tabs (round 2 — the pane is Equations-only now).
def toggle_param_row(row_id, sub_id):
    """Row-title click callback: accordion-toggle this row; on OPEN, focus its subsection."""
    S = st.session_state
    if S.get("_p_open") == row_id:                     # clicking the open row collapses it
        S["_p_open"] = None
        S["_eq_focus"] = None
        return
    S["_p_open"] = row_id
    close_cal()                                        # the pane only renders while cal is shut
    S["_eq_focus"] = sub_id                            # None for unmapped dials → no reveal
    if sub_id:
        S["_eq_focus_bump"] = int(S.get("_eq_focus_bump", 0)) + 1


# ---- reset infrastructure -----------------------------------------------------------------
# Widgets are KEYED (key=f"w_{param}") and seeded into session_state when first created, instead of
# passing value= — the seed-if-absent pattern lets a callback write the widget's state on reset
# without the "created with a default value but also set via Session State" warning. Every keyed
# widget registers its default in st.session_state["_wdefaults"] (rebuilt each run, so it always
# reflects exactly the controls visible at the current level); reset-all iterates that registry.
def _reg(wkey, default):
    """Seed a keyed widget's state if absent and register its default for reset-all. Returns wkey."""
    if wkey not in st.session_state:
        st.session_state[wkey] = default
    st.session_state.setdefault("_wdefaults", {})[wkey] = default
    return wkey


def _reset_one(wkey, default):
    st.session_state[wkey] = default          # callbacks run before widgets instantiate -> safe


def _reset_all():
    for wkey, dv in st.session_state.get("_wdefaults", {}).items():
        st.session_state[wkey] = dv
    st.session_state["_range_over"] = {}   # restore every narrowed MC sampling range


# ---- D-080: the coverage dial and its app-side range. Coverage ρ = E/B is the ONLY identified
# object on the finance side, so the widget dials it directly — in PERCENT.
#
# D-093 removed the hidden nominal scale entirely. COV_R0 = 100 and COV_K = 0.75 are GONE: the
# dial used to be translated into an (R₀, m, k) triple for a model that divided the triple back
# out, and now `Params.rho` IS the dialled object. The dimension stays app-side, and this overlay
# with it, for ONE remaining reason — the UNIT. The dial is a percentage because that is how
# coverage is read ("53 cents per dollar of build spend"), while the parameter is the fraction,
# so the envelope below cannot live in PARAM_RANGES, which is in parameter units. Everything
# that would read a notebook range for this key reads this overlay instead.
#
# The envelope (and the tight default — it is already a union) spans BOTH documented bases: the
# FIN4(b) run-rate restatement ρ ≈ 0.42 [0.33, 0.56] and the calendar-basis 0.533; the pending
# FIN4 basis ruling re-centres this dial and is the only open finance calibration question.
APP_RANGES = {"cov0": ("uniform", 33.0, 56.0)}
APP_SIM_DEFAULT = {"cov0": ("uniform", 33.0, 56.0)}


def _base_rng(key):
    """The vetted ENVELOPE distribution for a dimension: notebook targets, notebook free
    dials, or the app-side overlay (D-080's coverage dial)."""
    return m.TARGET_RANGES.get(key) or m.PARAM_RANGES.get(key) or APP_RANGES.get(key)


def _sim_rng(key):
    """The tight DEFAULT simulation distribution, overlay included."""
    return m.SIM_DEFAULT.get(key, APP_SIM_DEFAULT.get(key))


def _gc_sym():
    """Display symbol for compute growth TODAY. ALWAYS the plain g_c (D-077, Pavel): the base model
    has only the value at t = 0, so there is no zero index to carry; the slowdown extension
    introduces the FLOOR g_{c∞} as the second symbol and writes the path as g_c(t), leaving the
    initial value named g_c throughout. (The Params field is still `g_C0` internally — a code
    name, never shown.) This is what the compute DIAL, the g_a residual and the merged δ all
    mean, at every level."""
    return "g_c"


def _gc0_sym():
    """Display symbol for the Params FIELD `g_C0`. Plain g_c at EVERY level since D-088 — the
    field is today's compute growth, which is what its name always said.

    HISTORY, because this function existed only to paper over the gap it names. Between D-084 and
    D-086 the field held the pre-slowdown PLATEAU while the dial stated today's rate, so from
    Level 2 the two were different numbers (at p₀ᶜ = 25%, 0.637 vs 0.511 OOM/yr) and anything
    displaying g_C0 had to say `g_c^{pre}` there or contradict the caption beside it. D-088 moved
    the plateau identity inside Γ, so the field IS the observable again and the split is gone:
    the plateau is a derived intermediate that no dial and no card ever shows.

    Kept as a function rather than inlined so the one place that decides this stays one place —
    and because callers in three modules import it (D-088 chose the no-op over editing them)."""
    return _gc_sym()


def _tbounds_of(rng):
    """Natural-unit endpoints of a distribution: uniform/triangular bounds, lognormal ~90% CI.
    A scale_of band's natural units are the SHARE of its reference draw (audit X-10: the g_a_F
    dial is that share, so its bounds live here too)."""
    if rng[0] == "lognormal":
        med = float(np.exp(rng[1]))
        return med * float(np.exp(-1.645 * rng[2])), med * float(np.exp(1.645 * rng[2]))
    if rng[0] == "triangular":                # ("triangular", lo, mode, hi)
        return float(rng[1]), float(rng[3])
    if rng[0] == "scale_of":                  # ("scale_of", ref_key, lo_share, hi_share)
        return float(rng[2]), float(rng[3])
    return float(rng[1]), float(rng[2])


def _tbounds(tkey):
    """Target-slider bounds: the DEFAULT vetted envelope (range narrowing never moves these)."""
    return _tbounds_of(m.TARGET_RANGES[tkey])


# ---- user-edited MC sampling ranges (D-040/41/42): a session override dict LAYERED over the
# notebook's two-tier defaults. TARGET_RANGES/PARAM_RANGES are the ENVELOPE — the outer bounds a
# user may reach; the DEFAULT simulation range is the tight SIM_DEFAULT span (or a point, for
# dimensions with no multi-source basis). Keys are target keys or free-dial param keys; values
# are (lo, hi) endpoints in natural units (for the lognormal lag prior: the 90%-CI endpoints,
# from which mu/sigma are re-fitted).
def _fit_range(base, lo, hi):
    """Re-fit a distribution to user endpoints: uniform/triangular direct; lognormal from the
    90%-CI endpoints; others not editable."""
    kind = base[0]
    if kind == "lognormal":
        return m.lognormal_from_ci(lo, hi)
    if kind == "triangular":
        return ("triangular", float(lo), float(np.clip(base[2], lo, hi)), float(hi))
    if kind == "uniform":
        return ("uniform", float(lo), float(hi))
    if kind == "scale_of":                    # endpoints are shares of the reference draw
        return ("scale_of", base[1], float(lo), float(hi))
    return base


def _active_rng(key):
    """(distribution, edited?) actually simulated for a target/free-dial key: the user override
    if set (fitted with the ENVELOPE's distribution family), else the tight SIM_DEFAULT, else
    None — a POINT default (the dimension is pinned, not sampled)."""
    base = _base_rng(key)
    if base is None:
        return None, False
    over = st.session_state.get("_range_over", {})
    if key in over:
        return _fit_range(base, *over[key]), True
    return _sim_rng(key), False


def _active_ranges():
    """Override- and default-applied dicts for the Monte Carlo (consumed by mc_draw_batch):
    per dimension the user's range if set, else the tight SIM_DEFAULT. Point-default dims keep
    their envelope entry — they are excluded from the sampled set anyway (`_pinned_dim`)."""
    tr, pr = dict(m.TARGET_RANGES), dict(m.PARAM_RANGES)
    for dct in (tr, pr):
        for k in dct:
            arng, _ = _active_rng(k)
            if arng is not None:
                dct[k] = arng
    return tr, pr


def _use_range(ekey, ci, env):
    """[choose range] on a source row: set the MC crop to the source's documented CI (clipped
    to the vetted envelope). The trim lane re-syncs from the store next run."""
    lo, hi = max(float(ci[0]), env[0]), min(float(ci[1]), env[1])
    st.session_state.setdefault("_range_over", {})[ekey] = (lo, hi)


def _active_span(ekey):
    """(lo, hi) of the range control's CURRENT ends: the user's edit, else the tight default."""
    return st.session_state.get("_range_over", {}).get(ekey, _default_span(ekey))


# ---- trim-crop controls (D-079, replacing the D-041 range/spot mode tick): every editable
# MC-sampled dimension carries a two-handle trim CROP alongside the always-mounted spot slider.
# The crop IS the MC sampling range; a crop collapsed to a point is a PIN (the dimension leaves
# the sampled set), so point-default dimensions start with both handles on the spot. Spot values
# survive row filtering in plain `sv_` keys (widget keys are garbage-collected when their widget
# skips a run).
def _mc_dim_editable(key):
    """True when the dimension's MC distribution has editable endpoints (uniform/tri/lognormal,
    and — since the X-10 share dial — the scale_of share band, whose crop edits in share units)."""
    rng = _base_rng(key)
    return rng is not None and rng[0] in ("uniform", "triangular", "lognormal", "scale_of")


def _default_span(ekey):
    """(lo, hi) endpoints of the D-042 DEFAULT simulation range in the key's natural units —
    the tight documented-source span, or (for POINT dims) the collapsed point at the USER'S
    SPOT value: the crop rides the playhead until widened, and collapsing the handles back
    onto the spot cleanly drops the override (re-pinning the dimension)."""
    sim = _sim_rng(ekey)
    if sim is not None:
        return _tbounds_of(sim)
    fallback = TDEF[ekey] if ekey in m.TARGET_RANGES else getattr(P0, ekey)
    v = float(st.session_state.get(f"sv_{ekey}", fallback))
    return (v, v)


def _default_sampled(ekey):
    """Initial state of the per-parameter MC tick (D-079 rider, Pavel): TICKED for the
    dimensions the sampler draws by default (a ranged tight default — SIM_DEFAULT, which
    also covers the scale_of band g_a_F and the D-080 coverage overlay), UNTICKED for
    point-default dimensions."""
    return _sim_rng(ekey) is not None


def _sampled_on(ekey):
    return bool(st.session_state.get(f"smpv_{ekey}", _default_sampled(ekey)))


def _commit_sampled(ekey):
    st.session_state[f"smpv_{ekey}"] = bool(st.session_state[f"smp_{ekey}"])


def _mc_sampled(ekey):
    """True when the MC actually draws this dimension (D-079 + rider): its tick is ON and
    the crop is a REAL interval (a collapsed crop is a pin; point-default dims start with
    the handles on the spot)."""
    if not _sampled_on(ekey):
        return False
    lo, hi = _active_span(ekey)
    return float(hi) > float(lo)


def _reset_full(wkey, default, ekey):
    """↺ on a parameter row: restore the spot value AND the default crop AND the default
    MC tick (popping the override is enough for the crop — the trim lane re-syncs from the
    store every run)."""
    st.session_state[wkey] = default
    if ekey:
        st.session_state[f"sv_{ekey}"] = default
        st.session_state[f"smpv_{ekey}"] = _default_sampled(ekey)
        st.session_state[f"smp_{ekey}"] = _default_sampled(ekey)
        st.session_state.setdefault("_range_over", {}).pop(ekey, None)


def _spot_moved(ekey):
    """Playhead release: a crop COLLAPSED TO A POINT is a PIN, and a pin IS the spot — so it
    rides the playhead (returning to the default span drops the override, like a trim commit).
    That is what a pinned dimension MEANS, not a constraint on the spot, which is why D-087
    leaves it standing: a pinned dimension has no chosen MC range to protect.

    A REAL crop is left alone entirely. Under D-079 the released spot was clamped to the crop's
    nearest edge by a render-time guard in the row builders; D-087 retired that guard, so a spot
    and a real crop now move independently in both directions."""
    v = float(st.session_state[f"w_{ekey}"])
    over = st.session_state.setdefault("_range_over", {})
    cur = over.get(ekey)
    if cur is None or abs(float(cur[1]) - float(cur[0])) > 1e-12:
        return
    dflt = _default_span(ekey)
    if abs(v - dflt[0]) < 1e-9 and abs(v - dflt[1]) < 1e-9:
        over.pop(ekey, None)
    else:
        over[ekey] = (v, v)


def _commit_range_s(ekey, dflt):
    """Trim-lane commit (D-079) → the same override store the calibration modal edits. The
    crop is clamped into the vetted ENVELOPE — the hard bound: the free-dial playhead bounds
    the lane shares are hand-set and may exceed it. Returning the ends to the DEFAULT
    simulation range drops the override. D-087: the SPOT is not touched — dragging a crop handle
    across the playhead no longer pushes it along; the two controls are independent."""
    v = st.session_state[f"srng_{ekey}"]
    env_lo, env_hi = _tbounds_of(_base_rng(ekey))
    lo = float(np.clip(v[0], env_lo, env_hi))
    hi = float(np.clip(v[1], env_lo, env_hi))
    over = st.session_state.setdefault("_range_over", {})
    if abs(lo - dflt[0]) < 1e-9 and abs(hi - dflt[1]) < 1e-9:
        over.pop(ekey, None)
    else:
        over[ekey] = (lo, hi)


def _use_source(wkey, value, lo=None, hi=None):
    """Adopt a source's POINT: write its value into the destination control. Targets get the
    natural-units value clipped to the slider bounds — the vetted ENVELOPE, which still bounds
    the spot (D-087) — and free dials are set directly. Runs as a callback, so writing widget
    state is safe.

    D-087: the value lands EXACTLY. It used to be clipped a second time, at render, into the MC
    crop, so adopting a lab figure outside today's crop silently produced the crop's edge rather
    than the number the user clicked. That was the corner Pavel saw; independence removes it
    instead of special-casing it."""
    if isinstance(value, str):
        st.session_state[wkey] = value
    else:
        v = float(value)
        if lo is not None:
            v = float(np.clip(v, lo, hi))
        st.session_state[wkey] = v
        st.session_state["sv_" + wkey[2:]] = v   # keep the range-mode spot memory in step
