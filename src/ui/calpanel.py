"""The docked calibration panel (D-043, variant A2 — replaces the old st.dialog modal, whose
any-click-closes rerun behavior hid the [use] feedback).

Opens as a fixed ~300 px column right of the sidebar (theme.inject_layout_css pins the width);
while open, the Equations pane (Introduction tab retired — Pavel, round 2) folds into a thin
strip and the sidebar
auto-scrolls to the parameter (emphasis CSS + a same-origin JS shim). Content: current value +
interval header, the plain-language calibration target, per-source cards (reputation-ranked
order from the notebook's CAL_SOURCES) with a [choose]/[choose range] button that updates the
sidebar LIVE while the panel STAYS OPEN, and the methodology. (The MC sampling-range EDITOR
left with D-079: the row's trim lane is the single place the crop is set — ranges here are
documentation plus the [choose range] shortcuts.)
"""
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from .calibration import _bare_interval, _effective, _fmt3, _target_interval, merged_lag_months
from .content import (GRADES, INTERP, TSPEC, _CAL_ALT, _CAL_TARGET, _DELTA_MERGED_DOC,
                      fmt_dial_value,
                      _MATH_LABEL, _fmt_range, _sub_live)
from .model_access import m, _PARAM_TO_TARGET
from .state import (_active_rng, _active_span, _base_rng, _gc0_sym, _tbounds, _tbounds_of,
                    _use_range, _use_source, cal_open, close_cal, mc_active)


def param_row_key(key):
    """The sidebar row-container key that hosts this parameter's control (targets live on
    their target row; the lag family shares the Follower-lag row). Used for the emphasis CSS
    and the auto-scroll."""
    tkey = _PARAM_TO_TARGET.get(key)
    if tkey is None and key in ("delta_total", "Delta0"):
        tkey = "t_lag_mo"
    return f"row_{tkey or key}"


def _autoscroll(row_key):
    """Same-origin JS shim: scroll the sidebar to the emphasized row once per panel-open
    (guarded by a session key so reruns while the panel stays open don't re-scroll). Also
    scrolls the MAIN pane back to the top — the docked panel renders at the top of the main
    area, so an ⓘ clicked far down the page would otherwise open it off-screen (QA S3)."""
    components.html(
        f"""<script>
        function go(tries) {{
          const doc = window.parent.document;
          const el = doc.querySelector('section[data-testid="stSidebar"] .st-key-{row_key}');
          if (el) el.scrollIntoView({{behavior: "smooth", block: "center"}});
          else if (tries > 0) {{ setTimeout(function () {{ go(tries - 1); }}, 250); return; }}
          const main = doc.querySelector('section[data-testid="stMain"]') ||
                       doc.querySelector('.main');
          // direct assignment: scrollTo({{behavior: "smooth"}}) is silently ignored on this
          // container (verified in-browser), scrollTop is not
          if (main) main.scrollTop = 0;
          window.parent.scrollTo(0, 0);
        }}
        go(6);
        </script>""",
        height=0,
    )


# Per-source reputation chips (D-043, mockup layout_mockups.html): A green · B amber ·
# C light grey · D mid grey · F dark grey — grades ride on the notebook's CAL_SOURCES rows and
# follow the evidence register's own scale (D = vendor claim / press report / unverifiable, added
# with the D-080 coverage menu, whose per-lab restatements are genuinely D-grade).
_GRADE_COLORS = {"A": "#38a169", "B": "#d69e2e", "C": "#a0aec0", "D": "#8b95a3",
                 "F": "#718096"}

# Accounting-basis chips (D-080 follow-up / FIN4): outlined, so they read as a different KIND of
# label from the filled grade chips. Calendar and run-rate figures are incommensurable and mixing
# them inside one ratio is the defect FIN4 §3 found to dominate the money side.
_BASIS_COLORS = {"calendar": "#3182ce", "run-rate": "#805ad5", "mixed": "#c05621"}

_BASIS_DISCLOSURE = (
    "Each row divides a before-model-building profit by a model-building outlay and shows the "
    "implied coverage; the outlined chip is the **accounting basis** of the two legs. "
    "*Calendar*, *run-rate* and *mixed* readings are **not commensurable** — dividing a "
    "calendar-year cost by an annualized run-rate profit is worth ≈1.4× on the ratio by itself, "
    "more than the margin-definition question. (FIN4)")


def _grade_chip(rw):
    g = rw.get("grade")
    if not g:
        return ""
    return (f" <span title='source reputation grade {g}' style='font-size:10px;"
            f"border-radius:4px;padding:1px 5px;margin-left:4px;color:#fff;"
            f"background:{_GRADE_COLORS.get(g, '#a0aec0')};vertical-align:middle;'>{g}</span>")


def _basis_chip(rw):
    b = rw.get("basis")
    if not b:
        return ""
    c = _BASIS_COLORS.get(b, "#718096")
    return (f" <span title='accounting basis of the two legs: {b}' style='font-size:10px;"
            f"border-radius:4px;padding:0 5px;margin-left:4px;color:{c};"
            f"border:1px solid {c};vertical-align:middle;'>{b}</span>")


# ============================================================ the mini rail (D-089, variant 3)
# Pavel reviewed four interactive prototypes of a redesigned source row
# (Notes/prototypes/calibration_row_prototypes.html, commit 3a94c00) and chose variant 3,
# "Mini rail per row". His instruction, which the variant realises: "Instead of 'choose' being
# the button, I would like there to be the value as a button. What is more, I would like the spot
# value choice to be on the left and in case that source also offers a confidence interval, then
# there should be button on the right with clickable range."
#
# So each row draws the parameter's whole ENVELOPE as a short number line and puts the source on
# it: its POINT as a dot (adopts the spot) and, where it reports one, its INTERVAL as a bracket
# (adopts the MC crop). The rail also carries the parameter's CURRENT crop as a faint band, so a
# source that falls outside today's sampling range is visibly outside before you click it.
#
# THE CONSTRAINT AND HOW IT IS SOLVED. Streamlit cannot take clicks from injected HTML without a
# bidirectional custom component, and we are not building one (parked with the 3-point slider).
# So the two click targets are NATIVE `st.button`s, absolutely positioned from server-computed
# percentages, and everything that is not a click target — track, crop band, envelope end labels,
# interval end labels, dead markers, out-of-envelope chevrons — is ONE injected markdown block
# behind them. This is the same idiom D-079's collapsed-row crop band already uses: the server
# knows the fractions, CSS draws them, and no client state exists. Adoption marks are likewise
# server-side: the row's own CSS is emitted in the adopted variant, and the button LABEL carries a
# ✓ so the state is readable without reading stylesheets (and assertable in tests).
_RAIL_INSET = 6.0   # px inset at each rail end, so a marker at f = 0 or 1 sits fully on track


def _mx(f):
    """Left offset of a fraction along the mini rail. Same construction as the sidebar's crop
    band: a percentage of the track, pulled back by the inset so both ends land on the rail."""
    return f"calc({_RAIL_INSET:g}px + {100.0 * f:.3f}% - {2.0 * _RAIL_INSET * f:.3f}px)"


def _mw(d):
    return f"calc({100.0 * d:.3f}% - {2.0 * _RAIL_INSET * d:.3f}px)"


def _f01(x):
    return float(np.clip(x, 0.0, 1.0))


# Static half of the rail's CSS (the per-row half — positions and adopted marks — is computed
# server-side and appended per row). Colours come from the app's own variables so the panel and
# the sidebar's brackets read as one system.
_RAIL_CSS = """
/* the rail CONTAINER is the positioning context: the furniture markdown and the two buttons are
   SIBLINGS inside it, so the buttons' `position:absolute` must resolve against this box and not
   against the inner .mrail div (which they are not inside). The furniture is the only child left
   in flow, so it is also what gives the container its height. */
.st-key-calpanel [class*="st-key-srail_"]{position:relative;}
.st-key-calpanel [class*="st-key-srail_"] > div[data-testid="stVerticalBlock"]{gap:0!important;}
.st-key-calpanel [class*="st-key-srail_"] [data-testid="stElementContainer"]{margin:0!important;}
.st-key-calpanel .mrail{position:relative;height:46px;margin:2px 0 0;}
.st-key-calpanel .mtrack{position:absolute;left:6px;right:6px;top:20px;height:3px;
  background:rgba(var(--accent-rgb),0.10);border-radius:2px;}
.st-key-calpanel .mcrop{position:absolute;top:19px;height:5px;border-radius:2px;
  background:linear-gradient(90deg,rgba(var(--accent-rgb),0.75) 0 2px,
    rgba(var(--accent-rgb),0.20) 2px calc(100% - 2px),
    rgba(var(--accent-rgb),0.75) calc(100% - 2px) 100%);}
.st-key-calpanel .mends{position:absolute;top:31px;font-size:9px;opacity:0.40;
  font-variant-numeric:tabular-nums;}
.st-key-calpanel .mends.l{left:2px} .st-key-calpanel .mends.r{right:2px}
.st-key-calpanel .mci{position:absolute;top:31px;transform:translateX(-50%);font-size:9px;
  opacity:0.65;font-variant-numeric:tabular-nums;white-space:nowrap;}
.st-key-calpanel .mchev{position:absolute;top:12px;font-size:10px;color:#e2a33a;
  font-variant-numeric:tabular-nums;white-space:nowrap;}
.st-key-calpanel .mchev.l{left:2px} .st-key-calpanel .mchev.r{right:2px}
/* non-adoptable markers — same shapes, drawn rather than clickable */
.st-key-calpanel .mdead{position:absolute;top:16px;transform:translateX(-50%);}
.st-key-calpanel .mdead i{display:block;width:9px;height:9px;border-radius:50%;
  border:1.5px dashed #8b95a3;}
.st-key-calpanel .mdead b{position:absolute;left:50%;top:-13px;transform:translateX(-50%);
  font-size:10.5px;font-weight:600;opacity:0.55;font-variant-numeric:tabular-nums;
  white-space:nowrap;}
.st-key-calpanel .mbrkd{position:absolute;top:25px;height:6px;
  border:1px dashed rgba(var(--accent-rgb),0.35);border-top:0;border-radius:0 0 2px 2px;}
/* the two NATIVE buttons: the element container is positioned, the button is drawn */
.st-key-calpanel [class*="st-key-sdot_"],.st-key-calpanel [class*="st-key-sbrk_"]
  {position:absolute;z-index:3;}
.st-key-calpanel [class*="st-key-sdot_"]{top:14px;width:24px;transform:translateX(-50%);}
.st-key-calpanel [class*="st-key-sbrk_"]{top:25px;height:16px;z-index:2;}
.st-key-calpanel [class*="st-key-sdot_"] button,
.st-key-calpanel [class*="st-key-sbrk_"] button
  {background:none!important;border:0!important;box-shadow:none!important;padding:0!important;
   min-height:0!important;width:100%;}
.st-key-calpanel [class*="st-key-sdot_"] button{height:20px;position:relative;}
.st-key-calpanel [class*="st-key-sdot_"] button::after
  {content:"";position:absolute;left:50%;top:5px;margin-left:-5px;width:10px;height:10px;
   border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px rgba(var(--accent-rgb),0.16);
   transition:box-shadow .12s ease;}
.st-key-calpanel [class*="st-key-sdot_"] button:hover::after
  {box-shadow:0 0 0 6px rgba(var(--accent-rgb),0.22);}
.st-key-calpanel [class*="st-key-sdot_"] button p
  {position:absolute!important;left:50%;top:-13px;transform:translateX(-50%);margin:0!important;
   font-size:10.5px!important;font-weight:600!important;color:var(--accent)!important;
   font-variant-numeric:tabular-nums;white-space:nowrap;}
.st-key-calpanel [class*="st-key-sbrk_"] button{height:16px;position:relative;}
.st-key-calpanel [class*="st-key-sbrk_"] button::after
  {content:"";position:absolute;left:0;right:0;top:0;height:6px;border:1px solid #525b6e;
   border-top:0;border-radius:0 0 2px 2px;transition:all .12s ease;}
.st-key-calpanel [class*="st-key-sbrk_"] button:hover::after
  {border-color:var(--accent);background:rgba(var(--accent-rgb),0.10);}
.st-key-calpanel [class*="st-key-sbrk_"] button p{display:none!important;}
"""


def _mini_rail(pkey, i, rw, env, ekey, wkey, lo, hi, mode_mc, css):
    """One source row's rail. Returns nothing; appends per-row CSS to `css` and renders the
    furniture plus up to two native buttons into the current container.

    Row states, all four required by the brief:
      · point + interval  — dot and bracket, both adoptable (the bracket only in Monte-Carlo
        mode, where a crop is a live object; in Point mode it renders as furniture so the row
        still says what the source reports);
      · point only        — dot only, and NO dead second button;
      · display-only      — the value is outside the vetted envelope, or is a different object:
        the marker is drawn dashed and grey, or pinned to the rail edge as a chevron when it has
        nowhere to land, and the row's own `why` says so. Nothing is clickable;
      · interval wider than the envelope — the bracket is clipped to the rail with a warn
        chevron at the offending end, and it is NOT adoptable. Adoption must land EXACTLY
        (D-087), so an interval that cannot be taken whole is not offered. In practice this is
        reachable only on display-only rows: every adoptable interval in CAL_SOURCES sits inside
        its envelope by construction, because the envelope was DERIVED as the union of the
        menu's own intervals.
    """
    e_lo, e_hi = env
    span = (e_hi - e_lo) or 1.0

    def frac(v):
        return (float(v) - e_lo) / span

    rid = f"{pkey}_{i}"
    val = rw["value"]
    numeric = isinstance(val, (int, float))
    dead = bool(rw.get("display_only") or rw.get("triple") is not None)
    fv = frac(val) if numeric else None
    point_ok = numeric and not dead and -1e-9 <= fv <= 1 + 1e-9

    ci = rw.get("ci")
    ci_a = ci_b = None
    ci_inside = False
    if ci is not None:
        ci_a, ci_b = frac(ci[0]), frac(ci[1])
        ci_inside = ci_a >= -1e-9 and ci_b <= 1 + 1e-9
    ci_ok = ci is not None and ci_inside and not dead and mode_mc and ci_b > ci_a

    clo, chi = _active_span(ekey)
    h = ['<div class="mrail"><div class="mtrack"></div>',
         f'<div class="mcrop" style="left:{_mx(_f01(frac(clo)))};'
         f'width:{_mw(_f01(frac(chi)) - _f01(frac(clo)))}"></div>',
         f'<span class="mends l">{_fmt3(e_lo)}</span>',
         f'<span class="mends r">{_fmt3(e_hi)}</span>']
    if ci is not None and ci_b > ci_a:
        a, b = _f01(ci_a), _f01(ci_b)
        if not ci_ok:                       # furniture bracket: shown, not offered
            h.append(f'<span class="mbrkd" style="left:{_mx(a)};width:{_mw(b - a)}"></span>')
        h.append(f'<span class="mci" style="left:{_mx(a)}">{_fmt3(ci[0])}</span>')
        h.append(f'<span class="mci" style="left:{_mx(b)}">{_fmt3(ci[1])}</span>')
        if ci_a < -1e-9:
            h.append('<span class="mchev l">‹</span>')
        if ci_b > 1 + 1e-9:
            h.append('<span class="mchev r">›</span>')
    if numeric and not point_ok:
        if fv < 0 or fv > 1:                # nowhere to land: pin it to the rail's edge
            side, txt = ("r", f"{_fmt3(val)} ›") if fv > 1 else ("l", f"‹ {_fmt3(val)}")
            h.append(f'<span class="mchev {side}">{txt}</span>')
        else:                               # inside, but not adoptable (display-only / context)
            h.append(f'<span class="mdead" style="left:{_mx(_f01(fv))}">'
                     f'<b>{_fmt3(val)}</b><i></i></span>')
    st.markdown("".join(h) + "</div>", unsafe_allow_html=True)

    # ---- the click targets, positioned over the furniture -----------------------------------
    if point_ok:
        cur = st.session_state.get(wkey)
        on = isinstance(cur, (int, float)) and abs(float(cur) - float(val)) < 1e-9
        css.append(f'.st-key-sdot_{rid}{{left:{_mx(_f01(fv))};}}')
        if on:
            css.append(f'.st-key-sdot_{rid} button::after{{background:#fff;'
                       f'box-shadow:0 0 0 4px var(--accent);}}')
        st.button(("✓ " if on else "") + _fmt3(val), key=f"sdot_{rid}",
                  on_click=_use_source, args=(wkey, val, lo, hi),
                  help=f"Set the spot value to this source's {_fmt3(val)} {rw['unit']}"
                       " — exactly, whether or not it falls inside the sampling range.")
    if ci_ok:
        on = np.allclose(_active_span(ekey), (float(ci[0]), float(ci[1])), rtol=0, atol=1e-9)
        css.append(f'.st-key-sbrk_{rid}{{left:{_mx(_f01(ci_a))};'
                   f'width:{_mw(_f01(ci_b) - _f01(ci_a))};}}')
        if on:
            css.append(f'.st-key-sbrk_{rid} button::after{{border-color:var(--accent);'
                       f'background:rgba(var(--accent-rgb),0.55);}}')
        st.button(("✓ " if on else "") + f"[{_fmt3(ci[0])}, {_fmt3(ci[1])}]", key=f"sbrk_{rid}",
                  on_click=_use_range, args=(ekey, ci, env),
                  help=f"Set the Monte-Carlo sampling range to this source's interval "
                       f"[{_fmt3(ci[0])}, {_fmt3(ci[1])}] — the spot value is left alone.")


def _adopted_by(rows, wkey, ekey):
    """(spot source, crop source) currently in force, by name — or None where nothing matches.

    This is the panel's state line, and it is the honest half of the adoption marks: it names
    what the two controls are set to and which row, if any, put them there. It drops the moment
    the user drags either control away, because it is recomputed from the live values every run
    rather than remembered."""
    cur = st.session_state.get(wkey)
    span = _active_span(ekey) if ekey else None
    spot_src = crop_src = None
    for rw in rows:
        if rw.get("display_only") or rw.get("triple") is not None:
            continue
        v = rw["value"]
        if (spot_src is None and isinstance(v, (int, float))
                and isinstance(cur, (int, float)) and abs(float(cur) - float(v)) < 1e-9):
            spot_src = rw["source"]
        ci = rw.get("ci")
        if (crop_src is None and ci is not None and span is not None
                and np.allclose(span, (float(ci[0]), float(ci[1])), rtol=0, atol=1e-9)):
            crop_src = rw["source"]
    return spot_src, crop_src


def _source_cards(key, merged, tkey, pinned, mode_mc):
    """Per-source rows, each carrying the D-089 mini rail: the source's point is a dot that
    adopts the SPOT, its interval is a bracket that adopts the MC CROP, and the parameter's
    current crop rides behind them as a band. The panel stays open through either click."""
    rows = m.CAL_SOURCES.get("delta_total" if merged else key, [])
    if not rows:
        return
    if tkey:
        lo, hi = _tbounds(tkey)
        wkey = f"w_{tkey}"
    else:
        lo = hi = None
        wkey = f"w_{key}"
    # the app-side coverage dial (D-080) has no Params field, so its envelope lives in the
    # session overlay — _base_rng is the one lookup that spans both
    ekey = tkey if tkey else (key if _base_rng(key) is not None else None)
    env = _tbounds_of(_base_rng(ekey)) if ekey else None
    # the header names the affordances the rows ACTUALLY carry: a menu whose sources document no
    # interval (the D-080 coverage rows are single derived ratios) has dots and no brackets
    any_range = mode_mc and ekey is not None and any(
        "ci" in rw and not rw.get("display_only") for rw in rows)
    dial = f"*{TSPEC[tkey][0].split(' (')[0]}* slider" if tkey else "dial"
    st.markdown(f"**Sources** — each row places itself on the {'dial' if tkey else 'parameter'}'s "
                f"range. Click the **dot** to set the {dial} to that source's value"
                + (", or the **bracket** under it to set the Monte-Carlo sampling range to its "
                   "interval." if any_range else ".")
                + " This panel stays open.")
    rail_ok = env is not None and not pinned
    if rail_ok:
        _sp, _cr = _adopted_by(rows, wkey, ekey)
        _cur = st.session_state.get(wkey)
        bits = [f"spot **{_fmt3(_cur)}**" + (f" — *{_sp}*" if _sp else " — *not from a source*")
                if isinstance(_cur, (int, float)) else "spot —"]
        if mode_mc and ekey is not None:
            _lo2, _hi2 = _active_span(ekey)
            bits.append(f"range **[{_fmt3(_lo2)}, {_fmt3(_hi2)}]**"
                        + (f" — *{_cr}*" if _cr else " — *not from a source*"))
        st.caption(" · ".join(bits))
    if any(rw.get("basis") for rw in rows):
        st.caption(_BASIS_DISCLOSURE)
    css = [_RAIL_CSS] if rail_ok else []
    last_group = None
    for i, rw in enumerate(rows):
        # D-076: rows are GROUPED, because a menu that mixes objects teaches the wrong thing.
        # The subheading names what each block measures ("capability frontier" vs "largest run",
        # "lower bound — pretraining only" vs "upper bound — test-time included", …).
        grp = rw.get("group")
        if grp and grp != last_group:
            st.markdown(f"<div style='margin:0.6rem 0 0.2rem;font-size:0.78rem;"
                        f"letter-spacing:0.02em;text-transform:uppercase;opacity:0.65;'>"
                        f"{grp}</div>", unsafe_allow_html=True)
            last_group = grp
        with st.container(border=True, key=f"srow_{key}_{i}"):
            # `disp` overrides the rendered figure where the row's machine value is an exact
            # fraction (the coverage menu's 160/3) that must not be shown at full precision
            shown = rw.get("disp") or f"{rw['value']} {rw['unit']}"
            st.markdown(f"{rw['source']}{_basis_chip(rw)}{_grade_chip(rw)}  \n**{shown}**"
                        + (f" &nbsp; :gray[{rw['note']}]" if rw["note"] else ""),
                        unsafe_allow_html=True)
            # THE RAIL (D-089), between the row's head and its prose — every row that has an
            # envelope to sit on gets one, adoptable or not: placing a display-only reading on
            # the same line as the adoptable ones is what answers "why can't I take this?"
            # spatially, which is the argument that won variant 3.
            if rail_ok:
                with st.container(key=f"srail_{key}_{i}"):
                    _mini_rail(key, i, rw, env, ekey, wkey, lo, hi, mode_mc, css)
            if rw.get("display_only"):
                # a different object, a bound, or a retired reading: shown for context, never
                # clickable, and excluded from the default sampling span (source_span skips it).
                # `why` states THIS row's own reason where the generic sentence would mislead —
                # the coverage menu's rows are the same quantity, just outside the envelope.
                st.caption("— " + (rw.get("why")
                                   or "context only: not a competing estimate of this quantity."))
                continue
            # (the `triple` branch went with the money menus, D-093: no row carries the field
            # any more, and a guard against an impossible case reads as if one might.)
            if pinned:
                if key == "g_p":
                    st.caption("— fixed at this level: a measured leg, not a dial. The rows "
                               "above are for interpretation.")
                else:
                    st.caption("— pinned at this level; the spot value is fixed.")
                continue
            if rw.get("ci") is not None and not mode_mc:
                st.caption("— its interval is drawn under the rail; switch to **Monte Carlo** "
                           "to adopt it as the sampling range.")
    if css:
        st.markdown("<style>" + "".join(css) + "</style>", unsafe_allow_html=True)


def render(d, p):
    """The whole panel for the currently open parameter (state.cal_open()). Renders inside
    the caller's fixed-width column; returns nothing."""
    key = cal_open()
    pinned = bool(st.session_state.get("_cal_pinned", False))
    mode_mc = mc_active()
    with st.container(key="calpanel"):
        merged = key == "delta_total"
        tkey = "t_lag_mo" if merged else _PARAM_TO_TARGET.get(key)
        # ---- header: symbol = value · interval, with the ✕ close on the right --------------
        if merged:
            dval = f"{d.get('delta_rel', p.delta_rel):.2f}"   # merged δ routes through δ_rel
            _lr, _le = _active_rng("t_lag_mo")
            lag_lo, lag_hi = _tbounds_of(_lr)
            iv = (f"[{_fmt3(12.0 / lag_hi)}, {_fmt3(12.0 / lag_lo)}]"
                  + (" *(edited)*" if _le else ""))
            sym = "\\delta"
        else:
            # F-2: the EFFECTIVE value, same source as the equations-pane card head
            val = _effective(key, d, p)
            dval = fmt_dial_value(key, val)   # D-092: percent dials carry their sign
            sym = _gc0_sym() if key == "g_C0" else _MATH_LABEL.get(key, key)
            iv = None
            if not pinned:
                iv = _target_interval(key, p)
                if iv is None:
                    arng, edited = _active_rng(key)
                    if arng is not None:
                        iv = _bare_interval(arng, key)
                        if iv and edited:
                            iv += " *(edited)*"
                    elif key in m.PARAM_RANGES:   # MC dimension whose D-042 default is a POINT
                        iv = f"{dval} *(point)*"
        hc, xc = st.columns([6, 1], vertical_alignment="top")
        hc.markdown(f"#### ${sym}$ = {dval}"
                    + (f" &nbsp;·&nbsp; {iv}" if iv else "")
                    + (" &nbsp;·&nbsp; *(pinned)*" if pinned else ""))
        with xc:
            with st.container(key="calclose"):
                st.button("✕", key="cal_close_btn", on_click=close_cal,
                          help="close the calibration panel")
        tgt = _CAL_TARGET.get("delta_total" if merged else key)
        if merged:
            tgt = f"the ~{merged_lag_months(p):.0f}-month fringe lag stays constant"
        if tgt:
            st.caption(f"→ {tgt}" + (f" — set by the *{TSPEC[tkey][0].split(' (')[0]}* slider"
                                     if tkey else ""))
        # ---- source cards + methodology (D-079 rider: the panel's MC range EDITOR is gone —
        # the row's trim lane is the single place the crop is set; the panel keeps ranges as
        # DOCUMENTATION in the header interval, the [choose range] buttons and the grade
        # captions below) -------------------------------------------------------------------
        _source_cards(key, merged, tkey, pinned, mode_mc)
        st.markdown("**Methodology**")
        st.markdown(_sub_live(_DELTA_MERGED_DOC, d) if merged
                    else _sub_live(INTERP.get(key, f"**{key}** — see the calibration notes."),
                                   d))

        def _sim_desc(ek):
            arng, _ = _active_rng(ek)
            return (_fmt_range(arng) if arng is not None
                    else "point — not sampled by default (widen the range to sample it)")

        if merged:
            st.caption("grade **C** · MC samples the *Follower lag* target: "
                       + _sim_desc("t_lag_mo") + " months · envelope "
                       + _fmt_range(m.TARGET_RANGES["t_lag_mo"]))
        elif tkey:
            st.caption(f"grade **{GRADES.get(key, '—')}** · MC samples the "
                       f"*{TSPEC[tkey][0].split(' (')[0]}* target: {_sim_desc(tkey)} · "
                       f"envelope {_fmt_range(m.TARGET_RANGES[tkey])}")
        elif _base_rng(key) is not None:      # free dial, incl. the app-side coverage envelope
            st.caption(f"grade **{GRADES.get(key, '—')}** · MC {_sim_desc(key)} · "
                       f"envelope {_fmt_range(_base_rng(key))}")
        else:
            st.caption(f"grade **{GRADES.get(key, '—')}** · MC —")
        alt = _CAL_ALT.get("delta_total" if merged else key)
        if alt:
            st.caption(alt)
    # ---- one-shot sidebar auto-scroll for THIS open key --------------------------------
    if st.session_state.get("_cal_scrolled") != key:
        _autoscroll(param_row_key(key))
        st.session_state["_cal_scrolled"] = key
