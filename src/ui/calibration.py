"""Per-parameter calibration cards (the equations panel's right column): value + interval
head, plain-language calibration caption, and a [details] button that opens the DOCKED
calibration panel (ui/calpanel.py — D-043; the old st.dialog modal is gone). Calibration DATA
(CAL_SOURCES, ranges) lives in the notebook.
"""
import numpy as np
import streamlit as st

from .content import PCT_KEYS, TSPEC, _CAL_TARGET, _MATH_LABEL, fmt_dial_value
from .model_access import m, P0, TDEF, _PARAM_TO_TARGET
from .state import _active_rng, _gc0_sym, _gc_sym, _tbounds_of, level, open_cal
from .levels import merged_delta


def _bare_interval(rng, key=None):
    """Bare '[lo, hi]' interval for the right side of a calibration card — no distribution letter
    (the distribution's full description stays in the details popover via _fmt_range).

    `key` is optional and carries the UNIT (D-092): a percent-valued dial's range reads
    "[1, 25]%" rather than "[1, 25]", for the same reason its value reads "1%" — one quantity
    must not appear on screen in two conventions."""
    if not rng:
        return None
    pct = "%" if key in PCT_KEYS else ""
    k = rng[0]
    if k == "uniform":
        return f"[{rng[1]:g}, {rng[2]:g}]{pct}"
    if k == "lognormal":
        med = float(np.exp(rng[1]))
        lo, hi = med * float(np.exp(-1.645 * rng[2])), med * float(np.exp(1.645 * rng[2]))
        return f"[{lo:.2g}, {hi:.2g}]{pct}"
    if k == "triangular":
        return f"[{rng[1]:g}, {rng[3]:g}]{pct}"
    if k == "scale_of":
        base = getattr(P0, rng[1], None)
        if isinstance(base, (int, float)):
            return f"[{rng[2] * base:.2g}, {rng[3] * base:.2g}]{pct}"
        return f"[{rng[2]:g}, {rng[3]:g}]×{rng[1]}"
    if k == "choice":
        return "{" + ", ".join(f"{v:g}" for v in rng[1]) + "}"
    return None


def _card_head(col, head, interval, key=None, pinned=False):
    """First row of a calibration card: 'symbol = value' left, the bare interval right, and —
    when `key` is given — a small ⓘ at the far right that opens the docked calibration panel.
    D-078 follow-up (Pavel): the ⓘ is gone from every card whose parameter HAS a sidebar row —
    its » is the single route to detail now — so callers pass `key` only for the PINNED derived
    cards (κ, B₀, g_p), which have no row and would otherwise lose their panels entirely."""
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


def _effective(key, d, p):
    """The value the SIMULATION is running at for `key`, at the current level and dial state —
    what a calibration-card head shows (2026-07-28 functionality-test fix F-2).

    The head used to read `getattr(Params(), key)`, a frozen module default that ignored the
    level, the dials and the » panel's [choose] buttons entirely. `p` is Params(**d), so it
    carries the level pins and the derived anchors (κ, B₀) as well as the dialled values.

    Why a frozen default is wrong even at the app's own defaults, structurally: `Params()` is the
    FULL model and a card shows one LEVEL of it, so the two part company on every field whose
    value depends on a mechanism the level pins off. Live instance today: **κ**, 109.0745 against
    a Level-1 effective 109.1480, because κ is fitted through W(−Δ₀) and the value transition
    x_mid is pinned at Level 1.

    Do not maintain a list of instances here — they dissolve. Two of the three this fix was
    written for are already gone: g_c (0.514389 vs 0.510545) closed when D-088 made the field
    today's growth at every level, and S₀ (22.1483 vs 75.0, under a gloss stating S₀ = kR₀ = 75.0
    exactly) closed when D-090 re-based the cost path so B₀ IS today's bill. Both were fixed by
    making the DEFAULT correct — which is the better fix where it is available, and orthogonal to
    this one. A card that asks the simulation is right before, during and after each of those
    rounds, which is the whole reason it should ask.

    cov0 (D-080) is dialled in PERCENT while its Params field `rho` is the fraction, so it is
    the one key whose effective value is a unit conversion rather than a lookup. (Until D-093 it
    was a genuine derivation — the dial had no field at all and was read back off the three money
    primitives the sidebar derived from it.)
    """
    if key == "cov0":
        return 100.0 * float(d.get("rho", P0.rho))
    v = getattr(p, key, None)
    return v if isinstance(v, (int, float)) else getattr(P0, key, None)


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
            v = m.invert_targets({tkey: TDEF[tkey]}, p, merged=merged_delta(level()))[pkey]
            return f"{_fmt3(v)} *(point)*"
        lo, hi = _tbounds_of(arng)
        a, b = sorted(m.invert_targets({tkey: v}, p, merged=merged_delta(level()))[pkey]
                      for v in (lo, hi))
    except Exception:
        return None
    return f"[{_fmt3(a)}, {_fmt3(b)}]" + (" *(edited)*" if edited else "")


def _cal_entry(col, key, d, p, pinned=False):
    """One compact calibration card in a right-hand column: value + interval (image of the target
    range where a target drives the parameter), plain-language calibration caption, details."""
    val = _effective(key, d, p)
    tkey = _PARAM_TO_TARGET.get(key)
    dval = fmt_dial_value(key, val)   # D-092: percent dials carry their sign
    sym = _gc0_sym() if key == "g_C0" else _MATH_LABEL.get(key, key)
    head = f"${sym}$ **=** {dval}"
    if pinned:
        head += " · *(pinned)*"
    iv = None
    if not pinned:
        iv = _target_interval(key, p)
        if iv is None:                      # free dial: active (default-tight or edited) range
            arng, edited = _active_rng(key)
            if arng is not None:
                iv = _bare_interval(arng, key)
                if iv and edited:
                    iv += " *(edited)*"
            elif key in m.PARAM_RANGES:     # MC dimension whose D-042 default is a POINT
                iv = f"{dval} *(point)*"
    _card_head(col, head, iv, key=key if pinned else None, pinned=pinned)
    tgt = _CAL_TARGET.get(key)
    if tgt:
        if tkey:
            tgt += f" — the *{TSPEC[tkey][0].split(' (')[0]}* slider"
        col.caption(f"→ {tgt}")


def _cal_cards(col, entries, d, p):
    for key, pinned in entries:
        _cal_entry(col, key, d, p, pinned)


def merged_lag_months(p):
    """The fringe lag the current context is running at, in months — Δ₀ divided by the leader's
    EXACT t = 0 speed (D-086 P1-2), which is the inverse of the sidebar's own lag conversion.
    Quoted by the merged-δ card and its docked panel; a literal "~7 months" there would go stale
    the moment the user moves the Fringe-lag slider (and had already gone stale once, at D-076)."""
    speed = m.xdot_L0(p)
    return 12.0 * p.Delta0 / speed if speed > 0 else float("nan")


def _cal_delta_merged(col, d, p):
    """The merged single-δ calibration card (levels ≤ 2, where the lag target pins one rate)."""
    dstar = d.get("delta_rel", p.delta_rel)   # merged δ routes through δ_rel (split_delta)
    _lrng, _led = _active_rng("t_lag_mo")
    lag_lo, lag_hi = _tbounds_of(_lrng)
    # no ⓘ: the merged δ rides the Fringe-lag row, whose » opens this same delta_total panel
    _card_head(col, f"$\\delta$ **=** {dstar:.2f}",
               f"[{_fmt3(12.0 / lag_hi)}, {_fmt3(12.0 / lag_lo)}]"
               + (" *(edited)*" if _led else ""))
    col.caption(f"→ the ~{merged_lag_months(p):.0f}-month fringe lag stays "
                f"constant: $\\delta = ({_gc_sym()}+g_a)/\\Delta_0$ — the *Fringe lag* slider")
