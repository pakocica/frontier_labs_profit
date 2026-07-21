"""Per-parameter calibration cards (the equations panel's right column): value + interval
head, plain-language calibration caption, and a [details] button that opens the DOCKED
calibration panel (ui/calpanel.py — D-043; the old st.dialog modal is gone). Calibration DATA
(CAL_SOURCES, ranges) lives in the notebook.
"""
import numpy as np
import streamlit as st

from .content import TSPEC, _CAL_TARGET, _MATH_LABEL
from .model_access import m, P0, TDEF, _PARAM_TO_TARGET
from .state import _active_rng, _gc_sym, _tbounds_of, level, open_cal


def _bare_interval(rng):
    """Bare '[lo, hi]' interval for the right side of a calibration card — no distribution letter
    (the distribution's full description stays in the details popover via _fmt_range)."""
    if not rng:
        return None
    k = rng[0]
    if k == "uniform":
        return f"[{rng[1]:g}, {rng[2]:g}]"
    if k == "lognormal":
        med = float(np.exp(rng[1]))
        lo, hi = med * float(np.exp(-1.645 * rng[2])), med * float(np.exp(1.645 * rng[2]))
        return f"[{lo:.2g}, {hi:.2g}]"
    if k == "triangular":
        return f"[{rng[1]:g}, {rng[3]:g}]"
    if k == "scale_of":
        base = getattr(P0, rng[1], None)
        if isinstance(base, (int, float)):
            return f"[{rng[2] * base:.2g}, {rng[3] * base:.2g}]"
        return f"[{rng[2]:g}, {rng[3]:g}]×{rng[1]}"
    if k == "choice":
        return "{" + ", ".join(f"{v:g}" for v in rng[1]) + "}"
    return None


def _card_head(col, head, interval, key=None, pinned=False):
    """First row of a calibration card: 'symbol = value' left, the bare interval right, and —
    when `key` is given — the same small ⓘ the sidebar rows carry at the far right (opens the
    docked calibration panel; replaces the old full-width [details] button)."""
    cols = ([1.2, 1, 0.28] if interval else [2.2, 0.28]) if key else ([1.2, 1] if interval else None)
    if cols is None:
        col.markdown(head)
        return
    cs = col.columns(cols, vertical_alignment="top")
    cs[0].markdown(head)
    if interval:
        # the interval cell is raw HTML (for the alignment), so translate the *(...)* markdown
        # italics the interval strings carry into <em> tags; nowrap keeps a tag like
        # "(edited)" from breaking mid-word in the narrow cell (QA N1)
        iv = interval.replace("*(", "<em style='white-space:nowrap'>(") \
                     .replace(")*", ")</em>")
        cs[1].markdown(f"<div style='text-align: right'>{iv}</div>", unsafe_allow_html=True)
    if key:
        cs[-1].button("ⓘ", key=f"ieq_{key}", on_click=open_cal, args=(key, pinned),
                      help="calibration details — sources, ranges, methodology")


def _fmt3(v):
    """3-significant-digit number without scientific notation (1000 stays '1000')."""
    return f"{float(f'{v:.3g}'):g}"


def _target_interval(pkey, p):
    """For a target-driven parameter: the image of the ACTIVE target range (the tight D-042
    default, or the user's edit) under the inversion at the CURRENT parameter context. A POINT
    default (dimension not sampled) shows as the single inverted default value."""
    tkey = _PARAM_TO_TARGET.get(pkey)
    if tkey is None:
        return None
    arng, edited = _active_rng(tkey)
    try:
        if arng is None:   # POINT default: the dimension is pinned, not sampled
            v = m.invert_targets({tkey: TDEF[tkey]}, p, merged=(level() <= 5))[pkey]
            return f"{_fmt3(v)} *(point)*"
        lo, hi = _tbounds_of(arng)
        a, b = sorted(m.invert_targets({tkey: v}, p, merged=(level() <= 5))[pkey]
                      for v in (lo, hi))
    except Exception:
        return None
    return f"[{_fmt3(a)}, {_fmt3(b)}]" + (" *(edited)*" if edited else "")


def _cal_entry(col, key, d, p, pinned=False):
    """One compact calibration card in a right-hand column: value + interval (image of the target
    range where a target drives the parameter), plain-language calibration caption, details."""
    default = getattr(P0, key, None)
    tkey = _PARAM_TO_TARGET.get(key)
    dval = f"{default:g}" if isinstance(default, (int, float)) else str(default)
    sym = _gc_sym() if key == "g_C0" else _MATH_LABEL.get(key, key)
    head = f"${sym}$ **=** {dval}"
    if pinned:
        head += " · *(pinned)*"
    iv = None
    if not pinned:
        iv = _target_interval(key, p)
        if iv is None:                      # free dial: active (default-tight or edited) range
            arng, edited = _active_rng(key)
            if arng is not None:
                iv = _bare_interval(arng)
                if iv and edited:
                    iv += " *(edited)*"
            elif key in m.PARAM_RANGES:     # MC dimension whose D-042 default is a POINT
                iv = f"{dval} *(point)*"
    _card_head(col, head, iv, key=key, pinned=pinned)
    tgt = _CAL_TARGET.get(key)
    if tgt:
        if tkey:
            tgt += f" — the *{TSPEC[tkey][0].split(' (')[0]}* slider"
        col.caption(f"→ {tgt}")


def _cal_cards(col, entries, d, p):
    for key, pinned in entries:
        _cal_entry(col, key, d, p, pinned)


def _cal_delta_merged(col, d, p):
    """The merged single-δ calibration card (levels ≤ 5, where the lag target pins one rate)."""
    dstar = (P0.g_C0 + P0.g_a) / P0.Delta0
    _lrng, _led = _active_rng("t_lag_mo")
    lag_lo, lag_hi = _tbounds_of(_lrng)
    _card_head(col, f"$\\delta$ **=** {dstar:.2f}",
               f"[{_fmt3(12.0 / lag_hi)}, {_fmt3(12.0 / lag_lo)}]"
               + (" *(edited)*" if _led else ""), key="delta_total")
    col.caption(f"→ the ~8-month follower lag stays constant: "
                f"$\\delta = ({_gc_sym()}+g_a)/\\Delta_0$ — the *Follower lag* slider")
