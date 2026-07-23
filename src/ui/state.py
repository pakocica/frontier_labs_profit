"""Session-state infrastructure: the reset registry (keyed widgets, seed-if-absent),
dual-mode range/spot stores (D-041: sv_*/rmv_* shadow keys, _spoton_ transition flag),
the user-edited MC sampling-range override store (_range_over, D-040/41/42), distribution
bounds/fitting utilities, and the per-run level()/mc_active() reads.
The GC-proof patterns here fix real Streamlit bugs — see D-041/D-042 implementation notes.
"""
import numpy as np
import streamlit as st

from .levels import LEVEL_LABELS
from .model_access import m, P0, TDEF


def level():
    """The active level, read from session state (seeded so the selector widget can bind
    to the same key later in the run without a default= conflict). Values outside the current
    ladder (e.g. a stale "7 · …" from before the D-044 cap) clamp to Level 1."""
    if st.session_state.get("level") not in LEVEL_LABELS:
        st.session_state["level"] = LEVEL_LABELS[0]
    return int(st.session_state["level"].split(" ", 1)[0])


def mc_active():
    """True while the top-bar mode switch (D-043) is on Monte Carlo: the dual-mode sidebar
    rows render and the chart tiles show the MC fans. Accumulation itself runs in EITHER
    mode (background precalc)."""
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


# (the right chart panel's collapse/width moved CLIENT-side in D-050 — theme.inject_frontend_js
# owns it via localStorage + a root CSS variable, so no server state exists for it anymore; the
# panel column itself renders every run and the MC component stays mounted by construction)


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


def _gc_sym():
    """Display symbol for today's compute growth: plain g_c below the slowdown level (L1-3, where
    there is no g_c0 vs g_c∞ distinction to draw), subscripted g_c0 from L4 (display only)."""
    return "g_c" if level() < 4 else "g_{c0}"


def _tbounds_of(rng):
    """Natural-unit endpoints of a distribution: uniform/triangular bounds, lognormal ~90% CI."""
    if rng[0] == "lognormal":
        med = float(np.exp(rng[1]))
        return med * float(np.exp(-1.645 * rng[2])), med * float(np.exp(1.645 * rng[2]))
    if rng[0] == "triangular":                # ("triangular", lo, mode, hi)
        return float(rng[1]), float(rng[3])
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
    return base


def _active_rng(key):
    """(distribution, edited?) actually simulated for a target/free-dial key: the user override
    if set (fitted with the ENVELOPE's distribution family), else the tight SIM_DEFAULT, else
    None — a POINT default (the dimension is pinned, not sampled)."""
    base = m.TARGET_RANGES.get(key) or m.PARAM_RANGES.get(key)
    if base is None:
        return None, False
    over = st.session_state.get("_range_over", {})
    if key in over:
        return _fit_range(base, *over[key]), True
    return m.SIM_DEFAULT.get(key), False


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


def _commit_range(ekey, v, dflt):
    """Commit the modal range editor's current endpoints to the override store (returning the
    ends to the DEFAULT simulation range drops the override). Called inline with the slider's
    return value — see the note at the editor. Returns True when the store actually changed."""
    over = st.session_state.setdefault("_range_over", {})
    before = over.get(ekey)
    if abs(v[0] - dflt[0]) < 1e-9 and abs(v[1] - dflt[1]) < 1e-9:
        over.pop(ekey, None)
    else:
        over[ekey] = (float(v[0]), float(v[1]))
    return over.get(ekey) != before


def _bump_rng_widget(ekey):
    """Invalidate the modal range editor's widget so it re-seeds from the store (see the
    versioned-key note in _cal_dialog)."""
    S = st.session_state
    S[f"_rngv_{ekey}"] = int(S.get(f"_rngv_{ekey}", 0)) + 1


def _reset_range(ekey, dflt):
    st.session_state.setdefault("_range_over", {}).pop(ekey, None)
    _bump_rng_widget(ekey)


def _use_range(ekey, ci, env):
    """[use as range] on a source row: set the MC range to the source's documented CI (clipped
    to the vetted envelope)."""
    lo, hi = max(float(ci[0]), env[0]), min(float(ci[1]), env[1])
    st.session_state.setdefault("_range_over", {})[ekey] = (lo, hi)
    _bump_rng_widget(ekey)


def _active_span(ekey):
    """(lo, hi) of the range control's CURRENT ends: the user's edit, else the tight default."""
    return st.session_state.get("_range_over", {}).get(ekey, _default_span(ekey))


# ---- dual-mode controls (D-041): every MC-sampled dimension has a minimal tick that switches
# between RANGE mode (default — the control is the two-ended MC sampling range; the deterministic
# paths use the remembered spot value, shown gently) and SPOT mode (single value used everywhere;
# the dimension is pinned out of the MC). Spot values survive mode switches in plain `sv_` keys
# (widget keys are garbage-collected when their widget skips a run).
def _mc_dim_editable(key):
    """True when the dimension's MC distribution has editable endpoints (uniform/tri/lognormal)."""
    rng = m.TARGET_RANGES.get(key) or m.PARAM_RANGES.get(key)
    return rng is not None and rng[0] in ("uniform", "triangular", "lognormal")


def _default_mode(ekey):
    """Initial range/spot mode (D-042): RANGE for dimensions with a tight ranged default
    (SIM_DEFAULT), SPOT for point-default dimensions (pinned until the user widens them)."""
    return ekey in m.SIM_DEFAULT


def _default_span(ekey):
    """(lo, hi) endpoints of the D-042 DEFAULT simulation range in the key's natural units —
    the tight documented-source span, or (for POINT dims) the collapsed point at the USER'S
    SPOT value: it matches the ghost bullet and the pin-at-spot MC behavior, and collapsing
    the handles back onto the spot cleanly drops the override (re-pinning the dimension)."""
    sim = m.SIM_DEFAULT.get(ekey)
    if sim is not None and sim[0] != "scale_of":
        return _tbounds_of(sim)
    fallback = TDEF[ekey] if ekey in m.TARGET_RANGES else getattr(P0, ekey)
    v = float(st.session_state.get(f"sv_{ekey}", fallback))
    return (v, v)


def _range_mode(ekey):
    return bool(st.session_state.get(f"rmv_{ekey}", _default_mode(ekey)))


def _commit_mode(ekey):
    st.session_state[f"rmv_{ekey}"] = bool(st.session_state[f"rm_{ekey}"])


def _reset_full(wkey, default, ekey):
    """↺ on a dual-mode row: restore the spot value AND the default mode AND the default range."""
    st.session_state[wkey] = default
    if ekey:
        st.session_state[f"sv_{ekey}"] = default
        st.session_state[f"rmv_{ekey}"] = _default_mode(ekey)
        st.session_state[f"rm_{ekey}"] = _default_mode(ekey)
        st.session_state.setdefault("_range_over", {}).pop(ekey, None)


def _commit_range_s(ekey, dflt):
    """Sidebar range-slider callback → the same override store the calibration modal edits.
    Returning the ends to the DEFAULT simulation range drops the override."""
    v = st.session_state[f"srng_{ekey}"]
    over = st.session_state.setdefault("_range_over", {})
    if abs(v[0] - dflt[0]) < 1e-9 and abs(v[1] - dflt[1]) < 1e-9:
        over.pop(ekey, None)
    else:
        over[ekey] = (float(v[0]), float(v[1]))


def _use_source(wkey, value, lo=None, hi=None):
    """[use] in the calibration modal: write the source's value into the destination control.
    Targets get the natural-units value (clipped to the slider bounds); free dials are set
    directly. Runs as a callback, so writing widget state is safe."""
    if isinstance(value, str):
        st.session_state[wkey] = value
    else:
        v = float(value)
        if lo is not None:
            v = float(np.clip(v, lo, hi))
        st.session_state[wkey] = v
        st.session_state["sv_" + wkey[2:]] = v   # keep the range-mode spot memory in step
