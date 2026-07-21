"""The docked calibration panel (D-043, variant A2 — replaces the old st.dialog modal, whose
any-click-closes rerun behavior hid the [use] feedback).

Opens as a fixed ~300 px column right of the sidebar (theme.inject_layout_css pins the width);
while open, the tabbed Introduction|Equations pane folds into a thin strip and the sidebar
auto-scrolls to the parameter (emphasis CSS + a same-origin JS shim). Content: current value +
interval header, the plain-language calibration target, per-source cards (reputation-ranked
order from the notebook's CAL_SOURCES) with a [choose]/[choose range] button that updates the
sidebar LIVE while the panel STAYS OPEN, the MC sampling-range editor, and the methodology.
"""
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from .calibration import _bare_interval, _fmt3, _target_interval
from .content import (GRADES, INTERP, TSPEC, _CAL_ALT, _CAL_TARGET, _DELTA_MERGED_DOC,
                      _MATH_LABEL, _fmt_range, _sub_live)
from .model_access import m, P0, _PARAM_TO_TARGET
from .state import (_active_rng, _commit_range, _default_span, _gc_sym, _range_mode,
                    _reset_range, _tbounds, _tbounds_of, _use_range, _use_source, cal_open,
                    close_cal, mc_active)


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
# C light grey · F dark grey — grades ride on the notebook's CAL_SOURCES rows.
_GRADE_COLORS = {"A": "#38a169", "B": "#d69e2e", "C": "#a0aec0", "F": "#718096"}


def _grade_chip(rw):
    g = rw.get("grade")
    if not g:
        return ""
    return (f" <span title='source reputation grade {g}' style='font-size:10px;"
            f"border-radius:4px;padding:1px 5px;margin-left:4px;color:#fff;"
            f"background:{_GRADE_COLORS.get(g, '#a0aec0')};vertical-align:middle;'>{g}</span>")


def _source_cards(key, merged, tkey, pinned, mode_mc):
    """Per-source cards with the single adaptive button: [choose] sets the spot value live;
    in Monte-Carlo mode it reads [choose range] where the source documents an interval (the
    range becomes the MC sampling range). The panel stays open either way."""
    rows = m.CAL_SOURCES.get("delta_total" if merged else key, [])
    if not rows:
        return
    btn_word = "choose range" if mode_mc else "choose"
    st.markdown(f"**Sources** — *{btn_word}* moves the "
                + (f"*{TSPEC[tkey][0].split(' (')[0]}* slider" if tkey else "dial")
                + " live; this panel stays open.")
    if tkey:
        lo, hi = _tbounds(tkey)
        wkey = f"w_{tkey}"
    else:
        lo = hi = None
        wkey = f"w_{key}"
    ekey = tkey if tkey else (key if key in m.PARAM_RANGES else None)
    env = _tbounds_of(m.TARGET_RANGES[ekey] if ekey in m.TARGET_RANGES
                      else m.PARAM_RANGES[ekey]) if ekey else None
    over = st.session_state.get("_range_over", {})
    for i, rw in enumerate(rows):
        with st.container(border=True):
            st.markdown(f"{rw['source']}{_grade_chip(rw)}  \n**{rw['value']} {rw['unit']}**"
                        + (f" &nbsp; :gray[{rw['note']}]" if rw["note"] else ""),
                        unsafe_allow_html=True)
            if pinned:
                st.caption("— pinned at this level; the spot value is fixed.")
                continue
            as_range = mode_mc and "ci" in rw and ekey is not None
            if as_range:
                ci_cl = (max(float(rw["ci"][0]), env[0]), min(float(rw["ci"][1]), env[1]))
                chosen = over.get(ekey) is not None and \
                    np.allclose(over[ekey], ci_cl, rtol=0, atol=1e-9)
                st.button(("✓ " if chosen else "") + f"choose range [{rw['ci'][0]:g}, "
                          f"{rw['ci'][1]:g}]",
                          key=f"choose_{key}_{i}", on_click=_use_range,
                          args=(ekey, rw["ci"], env),
                          help="Set the MC sampling range to this source's documented "
                               "interval (clipped to the vetted envelope).")
            else:
                cur = st.session_state.get(wkey)
                chosen = (isinstance(rw["value"], (int, float))
                          and isinstance(cur, (int, float))
                          and abs(float(cur) - float(rw["value"])) < 1e-9) \
                    or (isinstance(rw["value"], str) and cur == rw["value"])
                st.button(("✓ " if chosen else "") + "choose",
                          key=f"choose_{key}_{i}", on_click=_use_source,
                          args=(wkey, rw["value"], lo, hi),
                          help="Move the slider to this source's value"
                               + (" (the ghost spot thumb follows; the sampling range is "
                                  "unchanged)." if mode_mc else "."))


def _range_editor(key, merged, tkey):
    """MC sampling-range editor over the vetted envelope (D-042 two-tier ranges) — same
    override store as the sidebar range sliders, so edits show up there and vice versa."""
    ekey = ("t_lag_mo" if merged else tkey) or (key if key in m.PARAM_RANGES else None)
    base = (m.TARGET_RANGES.get(ekey) or m.PARAM_RANGES.get(ekey)) if ekey else None
    if base is None:
        return
    if base[0] not in ("uniform", "triangular", "lognormal"):
        st.caption("MC sampling range: not editable (a derived or discrete-choice dimension).")
        return
    over = st.session_state.get("_range_over", {})
    tag = " — *edited*" if ekey in over else (
        "" if ekey in m.SIM_DEFAULT else " — *point (not sampled by default)*")
    st.markdown("**MC sampling range**" + tag)
    env_lo, env_hi = _tbounds_of(base)
    dflt = _default_span(ekey)
    cur = over.get(ekey, dflt)
    # VERSIONED widget key + seed-if-absent + INLINE commit from the return value (see the
    # D-042 note: programmatic writes to an already-rendered slider do not reliably reach its
    # frontend instance; bumping the version mints a fresh widget at the seeded value).
    rkey = f"rng_{ekey}_{st.session_state.setdefault(f'_rngv_{ekey}', 0)}"
    st.session_state.setdefault(rkey, (float(cur[0]), float(cur[1])))
    step = float(f"{(env_hi - env_lo) / 200:.1g}")
    rc1, rc2 = st.columns([5, 1], vertical_alignment="bottom")
    v = rc1.slider("sampling range (the slider bounds are the vetted envelope)",
                   float(env_lo), float(env_hi), step=step, key=rkey,
                   help="The default simulation range is TIGHT — the span of the documented "
                        "sources (a single point where only one source exists). Widen it up "
                        "to the vetted envelope, narrow it, or collapse it to a point."
                        + (" The lognormal lag prior is re-fitted from these 90%-CI "
                           "endpoints." if base[0] == "lognormal" else ""))
    _commit_range(ekey, v, dflt)
    rc2.button("↺", key=f"rrst_{ekey}", help="reset the sampling range to the default",
               on_click=_reset_range, args=(ekey, dflt))
    st.caption(("Endpoints are the lognormal prior's 90% CI (mu/sigma are re-fitted). "
                if base[0] == "lognormal" else "")
               + "Any change applies immediately and restarts the Monte Carlo.")


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
            dval = f"{(P0.g_C0 + P0.g_a) / P0.Delta0:.2f}"
            _lr, _le = _active_rng("t_lag_mo")
            lag_lo, lag_hi = _tbounds_of(_lr)
            iv = (f"[{_fmt3(12.0 / lag_hi)}, {_fmt3(12.0 / lag_lo)}]"
                  + (" *(edited)*" if _le else ""))
            sym = "\\delta"
        else:
            default = getattr(P0, key, None)
            dval = f"{default:g}" if isinstance(default, (int, float)) else str(default)
            sym = _gc_sym() if key == "g_C0" else _MATH_LABEL.get(key, key)
            iv = None
            if not pinned:
                iv = _target_interval(key, p)
                if iv is None:
                    arng, edited = _active_rng(key)
                    if arng is not None:
                        iv = _bare_interval(arng)
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
            tgt = "the ~8-month follower lag stays constant"
        if tgt:
            st.caption(f"→ {tgt}" + (f" — set by the *{TSPEC[tkey][0].split(' (')[0]}* slider"
                                     if tkey else ""))
        # ---- source cards + range editor + methodology -------------------------------------
        _source_cards(key, merged, tkey, pinned, mode_mc)
        if not pinned:
            _range_editor(key, merged, tkey)
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
        elif key in m.PARAM_RANGES:
            st.caption(f"grade **{GRADES.get(key, '—')}** · MC {_sim_desc(key)} · "
                       f"envelope {_fmt_range(m.PARAM_RANGES[key])}")
        else:
            st.caption(f"grade **{GRADES.get(key, '—')}** · MC —")
        alt = _CAL_ALT.get("delta_total" if merged else key)
        if alt:
            st.caption(alt)
    # ---- one-shot sidebar auto-scroll for THIS open key --------------------------------
    if st.session_state.get("_cal_scrolled") != key:
        _autoscroll(param_row_key(key))
        st.session_state["_cal_scrolled"] = key
