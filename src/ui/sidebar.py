"""The sidebar: targets-first controls (D-037) with trim-scrubber rows (D-079: the spot
"playhead" slider always mounted, a slim two-handle MC "trim" crop fused beneath it in
Monte-Carlo mode), building the effective parameter dict d for the run. Row builders are
closures over (d, LEVEL, MC_ACTIVE); level pins come from levels.apply_level_pins.

TWO PASSES (2026-07-28 functionality-test fix F-1). Every row is MOUNTED first and its dial
value collected — free dials straight into `d`, observable targets into `tg` — and only then,
once the whole level's dial state is in hand, is the ONE target->parameter inversion run
(`m.invert_targets`). The captions that show an implied parameter are deferred to that point
and written back into their row containers out of order (a standard Streamlit pattern, already
used for the lag row). The sidebar used to invert inline, row by row, in render order: the
Basics solve therefore read `g_C_inf` and `p0_c` out of a `d` the Dynamics rows had not written
yet and silently fell back to the Params defaults, so the D-086 guarantee ("every dial means
what it says at every p0_c") failed for exactly the two dials D-086 was written about. There is
now ONE inversion in the app, the one `test_24` guards.
"""
import numpy as np
import streamlit as st

from .content import INTERP, INTERP_T, SHORT_TIP, SHORT_TIP_T, TSPEC, _sub_live, lag_note
from .equations import param_subsection, sidebar_filter_keys
from .levels import (apply_level_pins, apply_post_inversion_pins, level_sample_keys,
                     merged_delta)
from .mc import _inspected_params
from .model_access import m, P0, TDEF
from .state import (ADVANCED_DIALS, APP_RANGES, _active_span, _commit_range_s,
                    _commit_sampled, _default_sampled, _default_span, _gc0_sym, _gc_sym,
                    _mc_dim_editable, _mc_sampled, _reg, _reset_all, _reset_full,
                    _sampled_on, _spot_moved, _tbounds, _tbounds_of, adv_hidden, adv_open,
                    cal_open, dial, mc_active, toggle_adv, toggle_cal, toggle_param_row)
from .theme import C_SAMPLE, inject_row_expand_css


def render(LEVEL):
    """Render the whole sidebar for the current level; returns the effective dict d that
    Params(**d) consumes (every parameter explicitly set — pins included, no fallbacks)."""
    MC_ACTIVE = mc_active()
    d = {}    # parameter dict passed to Params(**d) — free dials in pass 1, inverted in pass 2
    tg = {}   # observable-target dial values, inverted ONCE at the end of pass 1
    _caps = []  # deferred row captions: (row, fn) run in pass 3, after the inversion
    S = st.session_state

    # ---- level parameter filter (D-065, was D-048): by default only the parameters NEW at this
    # level get a row (the `_CHANGED_AT[level]` subsection). Independent of the pane tab and of
    # "show all equations" — the "show all parameters" toggle below is the only widener. Hidden
    # rows fall back to their remembered spot values (sv_ shadows), so the effective dict d stays
    # complete and MC ranges/modes survive. None = show all params up to this level (L1, or the
    # toggle-on override below).
    allowed = sidebar_filter_keys(LEVEL)

    def _lvl_vis(*keys):
        """The LEVEL filter alone: would this level show the row at all? (D-065.)"""
        return allowed is None or any(k in allowed for k in keys)

    def _vis(*keys):
        """Does any of these rows render? The level filter AND — for the S-curve midpoint and
        position dials — that variable's advanced-calibration button (D-111). Two independent
        gates, both of which must pass for a given key: the button can only reveal a dial the
        level has already introduced."""
        return any(_lvl_vis(k) and not adv_hidden(k) for k in keys)

    def _adv_button(container, fam, what):
        """D-111: the per-variable 'advanced calibration' toggle, rendered inside that
        variable's own section, directly above the dials it reveals. Suppressed entirely when
        the level would not show those dials anyway — a button that reveals nothing is worse
        than no button. `what` names the two dials in the reader's language, so the closed
        state still says what is behind it."""
        if not _lvl_vis(*ADVANCED_DIALS[fam]):
            return
        open_ = adv_open(fam)
        container.button("advanced calibration ▾" if not open_ else "advanced calibration ▴",
                         key=f"adv_{fam}", on_click=toggle_adv, args=(fam,),
                         help=("Hide" if open_ else "Show") + f" this curve's {what}. The two "
                              "endpoint dials above — the rate today and the rate it tends to — "
                              "are the calibration; these two say how the curve gets from one "
                              "to the other.")

    # ---- row builders (closures over d / LEVEL / MC_ACTIVE) --------------------------
    _track_css = []   # per-row playhead-track adornment CSS, injected ONCE at the sidebar end
    _SAMPLED_HERE = set(level_sample_keys(LEVEL))   # what the MC actually draws at this level

    def _sample_tick(container, ekey):
        """The per-parameter MC switch (D-079 rider, Pavel): ticked, the draws sweep this
        dimension's crop; unticked, every draw pins it at the SPOT value and the trim lane
        greys out. A semantic choice about which dimensions are drawn — not the retired
        display flip. The mode of record lives in the plain `smpv_` key (registered for
        Reset-all) so it survives Point-mode GC; the checkbox itself renders only in
        Monte-Carlo mode. (No help=: a collapsed-label widget can never render a tooltip —
        the tickTip saga; the row title's tooltip carries the orientation.)"""
        _reg(f"smpv_{ekey}", _default_sampled(ekey))
        wk = f"smp_{ekey}"
        st.session_state[wk] = _sampled_on(ekey)                         # pre-inst sync
        return container.checkbox("sample in MC", key=wk, label_visibility="collapsed",
                                  on_change=_commit_sampled, args=(ekey,))

    def _trim_lane(row, ekey, label, lo, hi, step, fmt=None, active=True):
        """D-079 trim lane: the two-handle MC crop rendered directly beneath the playhead, in
        an IDENTICAL column split with the playhead's OWN bounds and step, so the two tracks
        align position-for-position and read as one filmstrip control. The crop IS the MC
        sampling range (the `_range_over` store); collapsing the handles to a point re-pins
        the dimension, and the row's tick pins it outright (`active` False → the lane renders
        DISABLED, the crop held but ignored). Rendered every run in MC mode — collapsed rows
        hide it in CSS (an active crop then shows as the shaded band `_crop_band` paints on
        the playhead track) and the D-078 expand rule reveals it. The vetted envelope stays
        the hard bound via the commit clamp (_commit_range_s): the free-dial playhead bounds
        this lane shares are hand-set and may exceed it.

        D-087: the crop no longer constrains the playhead above it, so the two tracks can
        legitimately disagree — the spot is the scenario on screen, the crop is what the MC
        draws. The alignment is what makes that readable rather than confusing: a playhead
        sitting outside the lane's shaded span says exactly what it looks like."""
        dflt = _default_span(ekey)
        cur = _active_span(ekey)
        skey = f"srng_{ekey}"
        # pre-instantiation sync w/ store, clipped into the lane's bounds: a panel-edited
        # range can hold ends this lane can't show
        st.session_state[skey] = (float(np.clip(cur[0], lo, hi)),
                                  float(np.clip(cur[1], lo, hi)))
        kw = {"format": fmt} if fmt else {}
        with row.container(key=f"trim_{ekey}"):
            tc = st.columns([6.05, 0.55, 0.4], vertical_alignment="center")[0]
            tc.slider(label, lo, hi, step=step, key=skey,
                      on_change=_commit_range_s, args=(ekey, dflt),
                      label_visibility="collapsed", disabled=not active, **kw)

    def _crop_band(ekey, lo, hi):
        """READ-ONLY crop read-out on the playhead track (D-079): a shaded band, plus the
        inspected-draw tick while ⊙ inspection is active — pure per-row CSS on the track
        group's pseudo-elements (no extra DOM element, so the row's flex rhythm is untouched
        and a COLLAPSED row still shows its crop at zero height cost). The stop positions are
        calc()'d with the ±8px half-thumb inset the old ghost overlay verified aligned within
        ~1px at both ends of the font range (D-065).

        D-087 restyle: the band carries CRISP END-CAPS (a gradient inside the same ::after, so
        no extra pseudo-element is needed — ::before is the inspected tick). Under D-079 the
        thumb could never leave the band, so a soft translucent fill read as "the range the
        thumb is in". Now the spot is independent, and a soft fill under a thumb parked outside
        it reads as a bug. Capped, the band is a BRACKETED SPAN with its own two ends — the same
        object the trim lane's two handles and the calibration panel's mini-rail bracket show —
        and a thumb outside it reads as what it is: a scenario outside the sampled range. The
        thumb needs no change; Streamlit paints it opaque above the band at full contrast."""
        span = (hi - lo) or 1.0

        def _pos(v):
            f = float(np.clip((float(v) - lo) / span, 0.0, 1.0))
            return f"calc({f * 100.0:.2f}% + {8.0 - 16.0 * f:.1f}px)"

        clo, chi = _active_span(ekey)
        sel = (f'section[data-testid="stSidebar"] .st-key-w_{ekey} '
               f'[data-testid="stSlider"] div[role="group"]')
        fhi = float(np.clip((float(chi) - lo) / span, 0.0, 1.0))
        right = f"calc({(1.0 - fhi) * 100.0:.2f}% + {8.0 - 16.0 * (1.0 - fhi):.1f}px)"
        _track_css.append(
            f'{sel}::after{{content:"";position:absolute;top:calc(50% - 5px);height:10px;'
            f'left:{_pos(clo)};right:{right};border-radius:2px;pointer-events:none;'
            f'background:linear-gradient(90deg,rgba(var(--accent-rgb),0.85) 0 2px,'
            f'rgba(var(--accent-rgb),0.22) 2px calc(100% - 2px),'
            f'rgba(var(--accent-rgb),0.85) calc(100% - 2px) 100%);}}')
        ins = _inspected_params()
        if ekey in ins:
            _track_css.append(
                f'{sel}::before{{content:"";position:absolute;top:-0.25rem;bottom:-0.25rem;'
                f'width:0;left:{_pos(ins[ekey])};border-left:2px dashed {C_SAMPLE};'
                f'opacity:0.9;pointer-events:none;z-index:6;}}')

    def _row_head(row, label, panel_key, pinned=False, tip=None, row_id=None):
        """Line 1 of a sidebar row (D-043): the title, plus the » that opens the docked
        calibration panel (absolutely positioned on the block's right edge — out of flow, so
        the grid is untouched). Round 2 (Pavel): the row-head VALUE cell is gone — the value
        floats above the slider thumb (the re-enabled native stSliderThumbValue), identically
        for playhead and trim, point and MC. D-078: the title is a BUTTON (CSS styles it back
        to plain text) — clicking it accordion-expands the row and focuses the equation
        subsection carrying the parameter; `help=` carries the SHORT orientation tooltip
        (D-078 follow-up) and renders because the button label is VISIBLE."""
        # the » TOGGLES the panel (Pavel); while this row's panel is open the glyph points
        # back («) so the same button reads as the way out
        _open_here = cal_open() == panel_key
        row.button("«" if _open_here else "»", key=f"i_{panel_key}",
                   on_click=toggle_cal, args=(panel_key, pinned),
                   help="Open / close detailed calibration — sources, ranges, methodology")
        if row_id is None:
            row_id = f"row_{panel_key}"
        mark = "▾" if S.get("_p_open") == row_id else "▸"
        row.button(f"{mark} {label}", key=f"t_{row_id}", help=tip,
                   on_click=toggle_param_row, args=(row_id, param_subsection(panel_key, LEVEL)))

    def _param(container, key, label, lo, hi, step, dial_default=None, cap=None, **kw):
        """Compact 3-line row (D-043): label+value+ⓘ, keyed playhead slider (collapsed label)
        with a per-slider ↺ reset; in Monte-Carlo mode an editable sampled dimension adds the
        trim lane beneath it (D-079). Never hardcodes a default: the seed value is
        `getattr(P0, key)`, clipped into range. Filtered-out rows (D-048) render nothing but
        still feed d from the sv_ shadow (widget keys are GC'd when a widget skips a run;
        sv_ is not). The dial may live in DIFFERENT units than the Params field (audit X-10,
        g_a_F's follower/leader share): every stored key (w_/sv_/crop) holds the DIAL value,
        `dial_default` seeds it (getattr(P0, key) would be in Params units), and the units
        conversion happens in the DERIVE pass below — it reads an inverted parameter, which
        pass 1 does not have. `cap` renders a live ⇒ caption from the dial value, deferred to
        pass 3 for the same reason."""
        # Tooltips are substituted against this pass-1 `d`, which is thinner than the final one:
        # a token whose inputs are only derived later (⟪BE_VM⟫ reads g_C0/g_a/g_p, all emitted by
        # the inversion) falls back to the pinned defaults here, and reads live in the » panel,
        # which renders against the FINAL d. A tooltip cannot be deferred: help= is set at mount.
        interp = _sub_live(INTERP.get(key), d)
        default = float(np.clip(getattr(P0, key) if dial_default is None else dial_default,
                                lo, hi))
        # An MC dimension — a parameter-space envelope AND membership of the level's drawn set.
        # Both clauses are load-bearing: `mc_draw_batch` consumes `level_sample_keys(LEVEL)`, so a
        # dial with an envelope but no LEVEL_RANGED entry is never drawn however the tick is set,
        # and offering it a tick and a trim lane would be a control that silently does nothing.
        # η is the one such dial (D-125: continuous, so it has an envelope; deliberately unsampled,
        # per Pavel's "MC uses spot value for this parameter"), and every other `_param` dial is in
        # its level's list, so this clause changes nothing else.
        sampled = key in m.PARAM_RANGES and key in _SAMPLED_HERE
        wkey = f"w_{key}"
        if not _vis(key):
            cur = float(np.clip(float(S.get(wkey, S.get(f"sv_{key}", default))), lo, hi))
            d[key] = cur
            S[f"sv_{key}"] = cur
            return None
        _reg(f"sv_{key}", default)
        st.session_state.setdefault("_wdefaults", {})[wkey] = default
        if wkey in S:
            S[f"sv_{key}"] = float(S[wkey])   # mounted widget state is authoritative
        else:
            # remount after the level filter's GC: restore the remembered spot BEFORE the
            # slider instantiates (a skipped keyed widget loses its session-state key)
            S[wkey] = float(np.clip(S[f"sv_{key}"], lo, hi))
        crop_dim = sampled and _mc_dim_editable(key)   # carries a crop (trim lane in MC)
        ticked = sampled and MC_ACTIVE                 # every sampled dim gets the MC tick
        dual = crop_dim and MC_ACTIVE
        smp = _sampled_on(key)
        # D-087: NO spot∈crop guard. Spot and crop are independent controls — the spot is the
        # deterministic scenario on screen, the crop is what the Monte Carlo samples — so
        # dragging either leaves the other exactly where it was. (The ENVELOPE still bounds
        # both; only the nesting relation is retired.)
        row = container.container(key=f"row_{key}")
        _fmt = kw.get("format")
        # D-078 follow-up (Pavel): the title tooltip is the SHORT orientation line — the long
        # `interp` stays in the » panel; a key with no short tip gets NO tooltip, never the blob
        _row_head(row, label, key, tip=SHORT_TIP.get(key))
        # the tick column keeps the ↺ at the same x in both modes (6.05+0.55 = 6.6)
        cs = row.columns([6.05, 0.55, 0.4] if ticked else [6.6, 0.4],
                         vertical_alignment="center")
        raw = cs[0].slider(label, lo, hi, step=step, key=wkey, help=interp,
                           label_visibility="collapsed",
                           on_change=_spot_moved if crop_dim else None,
                           args=(key,) if crop_dim else None, **kw)
        d[key] = raw
        st.session_state[f"sv_{key}"] = float(raw)   # shadow survives filtering/GC
        if ticked:
            _sample_tick(cs[1], key)
        cs[-1].button("↺", key=f"r_{key}", help="reset value, MC tick and sampling range",
                      on_click=_reset_full, args=(wkey, default, key if sampled else None))
        if dual:
            _trim_lane(row, key, label, lo, hi, step, fmt=_fmt, active=smp)
            if smp:
                _crop_band(key, lo, hi)
        if cap:
            _caps.append((row, lambda v=raw: cap(v)))
        return row

    def _target_row(container, tkey, panel_key, visible=True):
        """Target control (bounds/default from the notebook's TARGET_RANGES/target_defaults)
        + ↺; in Monte-Carlo mode the trim lane rides beneath it (every target is an editable
        MC dimension, and the playhead bounds ARE the vetted envelope here by construction).
        Compact 3-line row (D-043); `panel_key` is the parameter the row's ⓘ opens in the
        docked calibration panel. Returns (value, row_container) — row is None when the row
        is filtered out (D-048), with the value served from the sv_ shadow."""
        label, step, fmt = TSPEC[tkey]
        lo, hi = _tbounds(tkey)
        default = float(np.clip(TDEF[tkey], lo, hi))
        wkey = f"w_{tkey}"
        if not visible:
            v = float(np.clip(float(S.get(wkey, S.get(f"sv_{tkey}", default))), lo, hi))
            S[f"sv_{tkey}"] = v
            return v, None
        _reg(f"sv_{tkey}", default)
        st.session_state.setdefault("_wdefaults", {})[wkey] = default   # ↺ / Reset-all target
        if wkey in S:
            S[f"sv_{tkey}"] = float(S[wkey])   # mounted widget state is authoritative
        else:
            # remount after the level filter's GC: restore the remembered spot BEFORE the
            # slider instantiates (a skipped keyed widget loses its session-state key)
            S[wkey] = float(np.clip(S[f"sv_{tkey}"], lo, hi))
        smp = _sampled_on(tkey)
        # D-087: no spot∈crop guard — see the note in _param
        row = container.container(key=f"row_{tkey}")
        _row_head(row, label, panel_key, tip=SHORT_TIP_T.get(tkey), row_id=f"row_{tkey}")
        cs = row.columns([6.05, 0.55, 0.4] if MC_ACTIVE else [6.6, 0.4],
                         vertical_alignment="center")
        v = cs[0].slider(label, lo, hi, step=step, key=wkey, format=fmt,
                         help=_sub_live(INTERP_T.get(tkey), d), label_visibility="collapsed",
                         on_change=_spot_moved, args=(tkey,))
        st.session_state[f"sv_{tkey}"] = float(v)
        if MC_ACTIVE:
            _sample_tick(cs[1], tkey)
        cs[-1].button("↺", key=f"r_{tkey}", help="reset value, MC tick and sampling range",
                      on_click=_reset_full, args=(wkey, default, tkey))
        if MC_ACTIVE:
            _trim_lane(row, tkey, label, lo, hi, step, fmt=fmt, active=smp)
            if smp:
                _crop_band(tkey, lo, hi)
        return v, row


    def _coverage_row(container):
        """The ONE money dial (D-080, Pavel): coverage ρ at t = 0, in PERCENT — and since
        D-093 the model's ONLY finance parameter, not a stand-in for three hidden ones.
        A base-level (Basics) control. Mirrors _target_row (MC tick, trim lane, band) with
        the APP-SIDE envelope (state.APP_RANGES — [26, 46.3]: D-104 dated every leg,
        D-128 gave the ceiling its witnessing row) instead
        of a notebook target; the dimension stays app-side because it is dialled in percent
        while Params.rho is the fraction. The default seed is EXACT (100·ρ, off the 0.1
        display grid) so the round trip is bit-exact. Returns (ρ %, row)."""
        ekey = "cov0"
        label = "$\\rho_0$  coverage at $t = 0$ (%)"
        lo, hi = _tbounds_of(APP_RANGES[ekey])
        default = float(np.clip(100.0 * P0.rho, lo, hi))
        wkey = f"w_{ekey}"
        if not _vis(ekey):
            v = float(np.clip(float(S.get(wkey, S.get(f"sv_{ekey}", default))), lo, hi))
            S[f"sv_{ekey}"] = v
            return v, None
        _reg(f"sv_{ekey}", default)
        st.session_state.setdefault("_wdefaults", {})[wkey] = default
        if wkey in S:
            S[f"sv_{ekey}"] = float(S[wkey])   # mounted widget state is authoritative
        else:
            S[wkey] = float(np.clip(S[f"sv_{ekey}"], lo, hi))
        smp = _sampled_on(ekey)
        # D-087: no spot∈crop guard — see the note in _param
        row = container.container(key=f"row_{ekey}")
        _row_head(row, label, "cov0", tip=SHORT_TIP.get("cov0"), row_id=f"row_{ekey}")
        cs = row.columns([6.05, 0.55, 0.4] if MC_ACTIVE else [6.6, 0.4],
                         vertical_alignment="center")
        v = cs[0].slider(label, lo, hi, step=0.1, key=wkey, format="%.1f",
                         label_visibility="collapsed", on_change=_spot_moved, args=(ekey,))
        S[f"sv_{ekey}"] = float(v)
        if MC_ACTIVE:
            _sample_tick(cs[1], ekey)
        cs[-1].button("↺", key=f"r_{ekey}", help="reset value, MC tick and sampling range",
                      on_click=_reset_full, args=(wkey, default, ekey))
        if MC_ACTIVE:
            _trim_lane(row, ekey, label, lo, hi, 0.1, fmt="%.1f", active=smp)
            if smp:
                _crop_band(ekey, lo, hi)
        return v, row

    def _tparam(container, tkey, pkey, capfn, rngcapfn=None):
        """Targets-first control for ONE parameter (D-037; users interact ONLY with observables —
        the ⚙ raw-parameter unlock was removed by Pavel's D-042 ruling). The target VALUE is
        collected into `tg`; the parameter is produced by the ONE inversion below, and the
        caption showing it is deferred to pass 3. The caption shows the implied parameter live:
        the IMAGE OF THE CROP while the MC samples the dimension (endpoints ordered numerically;
        the row head carries the spot), a point otherwise — and the endpoints go through the
        SAME inversion as the spot, so the two can never disagree."""
        v, row = _target_row(container, tkey, panel_key=pkey, visible=_vis(pkey))
        tg[tkey] = float(v)
        if row is None:                     # filtered out (D-048) — value only, no caption
            return
        if MC_ACTIVE and _mc_sampled(tkey) and rngcapfn is not None:
            rlo, rhi = _active_span(tkey)
            _caps.append((row, lambda: rngcapfn(*sorted((_invert({tkey: float(rlo)})[pkey],
                                                         _invert({tkey: float(rhi)})[pkey])))))
        else:
            _caps.append((row, lambda: capfn(d[pkey])))

    def _invert(over=None):
        """THE target → parameter inversion — the model's own (`m.invert_targets`, the function
        `test_24` guards), run at the FINAL dial context `_basep`. `over` replaces single target
        values, which is how a crop endpoint's implied parameter is read. Callable only from
        pass 2 onwards: `_basep` and `_merged` are assigned there and read at call time."""
        return m.invert_targets({**tg, **over} if over else tg, _basep, merged=_merged)

    # ---- sidebar body ----------------------------------------------------------------
    # D-050: the panel is titled by what it HOLDS (parameter controls). The control-orientation
    # prose now rides a native help= tooltip on the title (replacing the D-054 expander, which
    # was outdated) — one affordance pattern across the app.
    # The compact "Reset ↺" control sits to the RIGHT of the title (replacing the old full-width
    # "↺ Reset all to defaults" button) and reuses the same reset-all logic.
    _tc, _rc = st.sidebar.columns([4.4, 1.6], vertical_alignment="bottom")
    _tc.title(
        "Parameters",
        help="These controls set the observables and parameters of the model level you picked "
             "above, grouped by the mechanism they drive. Where a clean observable exists the "
             "control is in natural units and the caption beneath shows the implied parameter "
             "live; dials without one are stated directly. Defaults are the calibrated values — "
             "**Reset ↺** restores them.")
    _rc.button("Reset ↺", key="resetall_btn", on_click=_reset_all,
               help="Restore all controls to their calibrated defaults.")


    # The reset registry is rebuilt from scratch each run, so it always lists exactly the controls the
    # current level shows. (Clear BEFORE any keyed widget is created.)
    st.session_state["_wdefaults"] = {}
    # D-065: the widener for the level filter — ALWAYS shown (on every pane tab), default OFF.
    # `allowed is None` now means "L1 (all params are new)", where there is nothing to widen, so
    # the toggle is suppressed there only. A plain mem key preserves the preference across the
    # widget's GC. Ticked → allowed = None → every parameter UP TO this level shows.
    if allowed is not None:
        _reg("w_all_params", bool(S.get("_all_params_mem", False)))
        show_all_params = st.sidebar.checkbox(
            "show all parameters", key="w_all_params",
            help="By default the panel shows only the parameters **introduced at this level**. "
                 "Tick to see every parameter up to this level (equal model — only the visible "
                 "controls change).")
        S["_all_params_mem"] = bool(show_all_params)
        if show_all_params:
            allowed = None
    # -------------------------------------------------- global horizon (round 2, Pavel): the
    # 5/10-yr switch moved to the top of the CHARTS panel beside the mode switch — it
    # configures the graphs, not the model. Same "w_hz" key; the widget instantiates later in
    # the run (views._charts_column), so the sidebar reads the session key.
    d["T"] = 5.0 if S.get("w_hz") == "5 yr" else 10.0  # absent/deselected → 10 yr

    sb = st.sidebar

    # -------------------------------------------------- Level 1: Basics (always visible)
    # D-037 targets-first, D-076 base calibration: the Basics controls ARE the observables of the
    # calibrated base model — the two speed dials, the value dial, the money triple and the fringe
    # lag. The caption under each shows the implied parameter(s), recomputed live.
    if _vis("g_C0", "g_a", "nu", "delta_total", "Delta0", "delta_dev", "delta_rel"):
        sb.subheader("Basics")
    # This dial states TODAY'S compute growth and the parameter it sets IS that (D-088: g_C0
    # means g_C(0), and Γ derives the pre-slowdown plateau from it internally), so this one
    # inversion is the identity map. The row still defers to the DERIVE pass at the bottom of
    # this function, because the observables BELOW it do not: g_a is a residual against Γ(0),
    # and the lag converts at the leader's exact t = 0 speed — both of which need the Dynamics
    # dials that the group under Basics has not been read yet. That was F-1.
    _tparam(sb, "t_compute_x", "g_C0",
            lambda x: f"⇒ ${_gc0_sym()}$ = {x:.3f} OOM/yr",
            lambda a, b: f"⇒ ${_gc0_sym()} \\in$ [{a:.3f}, {b:.3f}] OOM/yr")
    # ---- effective compute (physical × everything else). g_a is its RESIDUAL against the compute
    # dial above, so the two dials can never double-count the same progress (Pavel's ruling). The
    # floor at 0 is the "algorithms cannot get worse" guard: it only binds in the corner where a
    # user drags compute above effective compute, and the caption says so out loud.
    def _eff_cap(x):
        txt = (f"⇒ $g_a$ = {x:.3f} OOM/yr (×{10.0**x:.2f}/yr) — the residual: "
               f"{10.0**(_gc_today + x):.2f} = {10.0**_gc_today:.2f} × {10.0**x:.2f}")
        if x <= 1e-12:
            txt += "  ⚠︎ **clamped at 0** — effective compute cannot grow slower than physical"
        return txt

    _tparam(sb, "t_eff_x", "g_a", _eff_cap,
            lambda a, b: (f"⇒ $g_a \\in$ [{a:.3f}, {b:.3f}] OOM/yr "
                          f"(×{10.0**a:.2f}–×{10.0**b:.2f}/yr), residual of "
                          f"${_gc_sym()}$ = {_gc_today:.3f} (today)"))
    # ---- fringe lag (months): ONE observable driving Δ0 AND the catch-up rate(s) (D-037).
    # Δ0 = lag × the leader's EXACT t = 0 speed and the catch-up intensity are both produced by
    # the derive pass (Pavel's refined re-anchor rule, D-081 addendum) — they need the FULL
    # effective context (engine, slowdown, η, follower engine, pins) — and this row's caption
    # writes back into its container out of order (a standard Streamlit pattern).
    _lag, _lag_row = _target_row(sb, "t_lag_mo",
                                 panel_key=("delta_total" if merged_delta(LEVEL) else "Delta0"),
                                 visible=_vis("delta_total", "Delta0", "delta_dev",
                                              "delta_rel"))
    tg["t_lag_mo"] = float(_lag)
    _lag_rng = MC_ACTIVE and _mc_sampled("t_lag_mo")

    # (the value target lives in Basics at every level — the old extensions level that used to
    # take it over is retired, D-081 ladder amendment)
    # D-120: the dial is the value GROWTH RATE, in ×/yr since D-133 (the compute dials' unit;
    # the %/yr reading rides on the calibration card), inverted through the leader's exact
    # t = 0 speed. The caption carries both readings of the parameter it sets — the slope ν and
    # the ×/OOM multiplier the calibration is quoted in — so nothing the ×/OOM dial used to show
    # is lost, and the ×/OOM number is a read-out that moves with the speed dials rather than a
    # second control that silently means a different rate at each of their settings.
    _tparam(sb, "t_value_growth", "nu",
            lambda x: (f"⇒ $\\nu$ = {x:.2f} value-OOM per OOM (×{10.0**x:.2f} per OOM) — at "
                       f"today's frontier speed $\\dot x^L_0$ = {_xdot0:.2f} OOM/yr"),
            lambda a, b: (f"⇒ $\\nu \\in$ [{a:.2f}, {b:.2f}] value-OOM per OOM "
                          f"(×{10.0**a:.2f}–×{10.0**b:.2f} per OOM)"))

    # ---- the compute PRICE leg (D-106). A full Level-1 row, not a display-only card: D-105 put
    # the base model's break-even test in closed form, ν(g_c+g_a) > g_c − g_p, so g_p is one of
    # exactly four numbers that decide whether the leader is ever profitable — and Epoch's grade-A
    # interval [×1.27, ×1.54] straddles the ×1.482 threshold that flips the verdict. It sits here,
    # between the value dial and the money dial, because that is where the equations put it (the
    # cost block runs between value and coverage). The caption carries the implied BILL growth,
    # which is the read-out this leg is trusted against (2.35 vs Cottier's observed 2.4×/yr) —
    # `_gc_today` is the model's own Γ(0) and is resolved in pass 3, like the residual caption.
    _tparam(sb, "t_price_x", "g_p",
            lambda x: (f"⇒ $g_p$ = {x:.3f} OOM/yr (×{10.0**x:.2f}/yr cheaper) — implied "
                       f"training-bill growth ×{10.0**(_gc_today - x):.2f}/yr "
                       "(observed ×2.4)"),
            lambda a, b: (f"⇒ $g_p \\in$ [{a:.3f}, {b:.3f}] OOM/yr "
                          f"(×{10.0**a:.2f}–×{10.0**b:.2f}/yr cheaper)"))

    # ---- the money side: ONE dial, and since D-093 ONE PARAMETER behind it. The dial has
    # always been the coverage ratio ρ (the only identified combination — scaling earnings and
    # cost jointly moves no verdict), but until D-093 it was translated here into a hidden
    # (R₀, m, k) triple for a model that then divided the triple back out. The model now takes
    # ρ directly, so this is a unit conversion and nothing else: the slider is in PERCENT
    # because that is how coverage is read, the parameter is the fraction.
    _cov, _crow = _coverage_row(sb)
    d["rho"] = float(_cov) / 100.0
    if _crow is not None:
        _crow.caption(f"labs currently earn **~{float(_cov):.0f} cents** per dollar of "
                      "model-building spend · break-even at **100%**")

    # -------------------------------------------------- Level 2: Dynamics (D-081 merge)
    # ONE level, two opposing forces: (1) compute growth slows toward the floor — which is also
    # what makes the training lead time ℓ matter (under Level 1's steady growth the ℓ-timing
    # only re-anchors the internal cost constant; a bending compute curve makes it bite), and
    # (2) algorithmic progress speeds up via the ψ RSI feedback, plus the value bend x_mid.
    # There is NO speed dial here — the LEVEL of algorithmic progress is a base-model quantity
    # (D-076: g_a is the Basics effective-compute residual); this level changes how it is
    # PRODUCED.
    # Pavel's addendum (2026-07-27): the Level-2 dials are ORDERED AS THEY APPEAR IN THE
    # EQUATIONS — the groups mirror the changed subsections in STORY order (slowdown → engine →
    # cost rider → value rider), each listing its dials in the order its subsection's equations
    # introduce them (first appearance wins). This supersedes the observables-above/
    # internals-below split for the Dynamics dials: γ, β₀, t_mid and x_mid leave "Model
    # internals" and sit with their equations. (MC sampling order is untouched — LEVEL_RANGED
    # stays the frozen old-ladder concatenation; mc_draw_batch consumes it in order.)
    if LEVEL >= 2:
        # (1) leader compute:  g_c(t) = S(t; g_c, g_c∞, t_mid)  →  g_c∞, then t_mid (D-082)
        if _vis("g_C_inf", "t_mid", "p0_c"):
            sb.subheader("Dynamics — compute slows down")
        _tparam(sb, "t_floor_x", "g_C_inf",
                lambda x: f"⇒ $g_{{c\\infty}}$ = {x:.2f} OOM/yr",
                lambda a, b: f"⇒ $g_{{c\\infty}} \\in$ [{a:.2f}, {b:.2f}] OOM/yr")
        # D-111: the two ENDPOINTS above (today's growth in Basics, the floor here) are the
        # calibrated dials; the midpoint and the position below are behind this button.
        _adv_button(sb, "leader_compute", "slowdown midpoint and how far into it we are today")
        _param(sb, "t_mid", "$t_{mid}$  slowdown midpoint (yr)", *dial("t_mid"))
        # D-084: the curve's POSITION today — the dial that resolves its slope, so it comes
        # right after the midpoint it is stated against (and reads it live in the caption).
        # Bounds: R11 (state.dial) — the envelope [1, 25]% padded to [1, 28]%. The FLOOR does not
        # move, and that is the model's constraint rather than a convention: slope_span raises
        # outside (0, 50)%, so a padded 0 or below would be an unstarted transition with infinite
        # slope. X-12's lesson still holds inside this — the far end stays reachable.
        _param(sb, "p0_c", "$q^c_0$  how far into the slowdown today (%)", *dial("p0_c"),
               cap=lambda v: (f"⇒ we are in the bottom **{v:.0f}%** of the curve today; it "
                              f"flattens (midpoint) in {d['t_mid']:.1f} yr and is "
                              f"{100.0 - v:.0f}% done by {2.0 * d['t_mid']:.1f} yr"))
        # (2) leader algo:  ȧᴸ = g_a[(1−α)(ψ/ψ(0))^η + α(g_c,t/g_c)^η]^{1/η}, ψ = 1+β₀e^{γx}
        #     →  β₀, then γ (with its freeze switch), and under them, behind this section's
        #     "Advanced calibration" expander, the two dials that shape the bracket the two of
        #     them sit in: η (the CES exponent) and the α drag.
        #
        # D-135 IS WHY THIS SECTION NO LONGER RUNS IN EQUATION ORDER. Pavel's 2026-07-27 addendum
        # orders each Level-2 group as its subsection's equations introduce their dials, and by
        # that rule η comes first (the ψ definition is stated against it). The other two groups
        # still obey it. Here the rule was putting the section's two least readable dials at the
        # top of the RSI story, and Pavel ruled the cut ("Agree with the cut, go ahead"): what is
        # visible is now the story — how much faster AI R&D goes today (β₀) and how fast that
        # speedup grows (γ) — and the machinery is one click below it.
        #
        # WHY β₀ AND γ ARE NOT SPLIT, though only γ is the "growth" half the section is named for:
        # β₀ is not separately identified. D-132 made it an INPUT level with a 75× envelope and no
        # calibration of its own, and D-132's `cal_alt` states the pair jointly — the evidence
        # constrains the ψ path, which is β₀ and γ together. Hiding either one would leave the
        # other looking like a measured quantity.
        #
        # WHY AN EXPANDER RATHER THAN D-111's per-variable "advanced calibration" BUTTON, which is
        # the house pattern in the two neighbouring sections: `_adv_button`'s copy is written for
        # an S-CURVE family — "this curve's … the two endpoint dials above — the rate today and
        # the rate it tends to — are the calibration; these two say how the curve gets from one to
        # the other". η and the α drag are not a midpoint and a position, and there is no curve
        # here whose endpoints β₀ and γ are, so that button would have to lie or be re-parameterised.
        # The mechanisms also differ in a way that matters to D-065: `_adv_button` hides by not
        # MOUNTING the row (the sv_ shadow carries the value), while an expander keeps its children
        # mounted and merely folds them away — so here the dial itself is alive whether the reader
        # has opened the fold or not, and the model input cannot depend on the fold at all.
        if _vis("eta", "beta0", "gamma"):
            sb.subheader("Dynamics — algorithms speed up")
        # β₀ and γ are the LEVEL and the GROWTH of the same object, so the two rows are
        # written to read as a pair (Pavel: the old "ψ compounding" "is not understandable…
        # how about something in the sense of RSI growth"). β₀ is dimensionless and was NOT
        # touched by D-091's base-10 rescale.
        _param(sb, "beta0", "$\\beta_0$  AI R&D speedup today", *dial("beta0"))
        if _vis("gamma"):
            freeze = sb.checkbox("Freeze AI assistance ($\\gamma = 0$)", key=_reg("w_freeze",
                                 bool(S.get("_freeze_mem", P0.gamma == 0.0))),
                                 help="Turns off the $\\psi$ RSI feedback. $\\gamma$ above "
                                      "~0.182 (at the default $\\eta = 0$, $\\alpha = 0.7$ mix) "
                                      "goes super-exponential inside the horizon; "
                                      "the runtime blow-up warning reads the simulated path, "
                                      "so it holds for any $\\eta$.")
            S["_freeze_mem"] = bool(freeze)   # survives the widget's GC while filtered out
        else:
            freeze = bool(S.get("_freeze_mem", S.get("w_freeze", P0.gamma == 0.0)))
        if freeze:
            d["gamma"] = 0.0
        else:
            # BOUNDS: R11, like every other free dial (state.dial). The step stays 0.02 —
            # D-091's gap was that γ went to base 10 while the slider kept its nats bounds, so
            # the whole meaningful region (blow-up at ≈0.182) sat in the first fifth of a track
            # running to 0.45. R11 finishes that fix from the other end: the reach is now the
            # envelope [0, 0.174] padded to 0.20, so the blow-up threshold is one step from the
            # top instead of two-fifths along, and the 0.45 tail — which no calibration defends —
            # is gone. The floor stays 0, which is the freeze switch's value and meaningful.
            _param(sb, "gamma", "$\\gamma$  how fast that speedup grows (/OOM)",
                   *dial("gamma"))
        # ---- ADVANCED CALIBRATION (D-135): η and the α drag ------------------------------
        # What is behind the fold is the CES BRACKET the RSI story runs inside, and both of its
        # settings are conventions rather than measurements — which is the argument for folding
        # them: η = 0 is the Cobb-Douglas grade-A convention D-125 moved the base to, and the
        # shipped drag is the one that delivers the conventional α = 0.70. Neither is a number
        # this project measured, and the α caption's η-dependence — the whole point of D-098 — is
        # expert furniture: it answers a question a reader who has not opened this fold is not
        # asking.
        #
        # THE FOLD IS VIEW-ONLY (D-065), and by a stronger route than the level filter's. Both
        # calls below are UNCONDITIONAL and their containers are the only thing that changed:
        # Streamlit keeps an expander's children mounted whether it is open or shut (the open
        # state is client-side), so `d["eta"]` and `d["alpha"]` are written by exactly the same
        # code on exactly the same run either way. There is no shadow-restore path to get wrong
        # here — the sv_ shadows still carry these dials through the LEVEL filter, as before.
        #
        # The expander is built only when it would hold something: from Level 3 the default view
        # filters both rows out, and an empty "Advanced calibration" would promise dials that are
        # not there. `_param` and `_target_row` both return before touching their container in
        # that case, so `sb` is never drawn into — it stands in so the two calls can stay
        # unconditional, which is the contract above.
        _adv = (sb.expander("Advanced calibration", expanded=False)
                if _vis("eta") or _vis("alpha") else sb)
        # THE SELECTBOX IS GONE (D-125, Pavel: the discreteness "just complicates things"). It
        # existed to make an η → −∞ endpoint selectable; R8 removed Leontief ("Remove Leontief
        # from the options") and D-122 removed the arbitrary η = −2 stand-in, and with both gone
        # the choice dimension had nothing left a slider cannot express. What replaces it is an
        # ORDINARY `_param` dial over the vetted envelope [−1.20, +1.00] at step 0.05 — the right
        # end being the CES family's own ceiling (σ = 1/(1−η) is negative beyond it), which
        # `_DIAL_SPEC`'s new `ceil` field holds against R11's padding.
        #
        # Both retirements were DIAL changes, not model changes, and stay so: `Params.leontief`
        # and `_ces_bracket`'s min() branch are the CES family's endpoint and are tested there
        # (test_alpha_observable's test_03), and η = −2 is still a legal `Params.eta`. The widget
        # pins the Leontief flag False explicitly, the way it pins τ: `render` promises every
        # parameter is set with no dataclass fallback. That pin, and the `w_eta` migration below
        # it, are plain statements in no container at all — the fold holds the two ROWS, and
        # nothing else in this block is inside anything.
        #
        # A session carried over from before this change holds a STRING in `w_eta` (an option
        # label like "1 (weighted avg)"), which a slider cannot mount. It is cleared before the
        # widget instantiates, the same way `state.level()` clamps a retired level.
        for _k in ("w_eta", "sv_eta"):
            if not isinstance(S.get(_k, 0.0), (int, float)):
                S.pop(_k, None)
        S.pop("_eta_mem", None)      # the retired label memo; the sv_ shadow serves this now
        d["leontief"] = False
        _param(_adv, "eta", "$\\eta$  CES exponent (compute–labor)", *dial("eta"))
        _eta_disp = f"{d['eta']:.2f}".rstrip("0").rstrip(".")   # "0" / "0.61" / "-1.2", for α
        # α sits directly beneath η because they are the two halves of one bracket: η says how
        # substitutable the inputs are, α how much weight the compute one carries. D-098 dials α
        # through the OBSERVABLE (the drag), not the weight, so that holding the drag fixed while
        # η moves counts the bottleneck evidence exactly once. That pairing is why D-135 folds the
        # two together rather than splitting them across the fold.
        # VIEW-ONLY contract (D-065): the level filter hides the ROW, never changes the model
        # input. So d["alpha"] is assigned on every path -- _target_row serves the sv_ shadow
        # when it is filtered out, exactly as the eta row writes d["eta"] outside its own _vis
        # guard. Gating the assignment instead silently dropped the key from d and test_ui's
        # tab-independence check caught it.
        # Gated on the PARAMETER key, not the observable's key: `_vis` reads
        # `sidebar_filter_keys`, which is derived from `subsection_param_entries`, and that table
        # is keyed by parameter throughout (t_compute_x is gated by g_C0 the same way). Gating on
        # "loss_half_gC" made this row unreachable in the default Level-2 view — D-098 follow-up.
        #
        # ROUTED THROUGH `_tparam` (audit A/6, FM-5). R8 removed the Leontief fork above, and with
        # it the disabled placeholder slider that fork rendered — a widget keyed
        # `w_loss_half_gC_leontief`, seeded at 50.0, above its own vetted envelope (then [22, 45]). What
        # is left is the ordinary target row, and it now goes through the SAME path every other
        # observable takes. Before this it called `m.alpha_from_loss` INLINE, so loss_half_gC never
        # entered `tg` and never reached the one `_invert()` below, even though `m.invert_targets`
        # has carried the branch since D-098 — two implementations of one inversion, and the row
        # was the only target with no ⇒ caption. It gains one, and the caption names the active η:
        # D-098's headline property is that the delivered α MOVES with the substitution setting
        # (the shipped 38.44% drag delivers α = 0.70 at the base η = 0, 0.74 at η = 0.61 and
        # 0.77 at η = 1), and the widget was showing nothing that moved.
        _tparam(_adv, "loss_half_gC", "alpha",
                lambda a: (f"⇒ $\\alpha$ = {a:.2f} at $\\eta$ = {_eta_disp} — the weight the "
                           "model delivers for the drag you stated; it moves with $\\eta$, so "
                           "the bottleneck evidence is counted once"),
                lambda a, b: (f"⇒ $\\alpha \\in$ [{a:.2f}, {b:.2f}] at $\\eta$ = "
                              f"{_eta_disp}"))
        # (3) THE COST BLOCK HAS NO LEVEL-2 RIDER. B_t = 10^{c^L_t}·10^{−g_p t} at every level,
        # and its one dial g_p lives at Level 1. Until D-127 the build lag ℓ rendered here under
        # a "training paid in advance" subheader; removing it is why the Level-2 story is now
        # THREE blocks rather than four (D-123 addendum, and equations.py's caption).
        # (4) value rider (D-083):  w'(x) = S(x; ν, ν_∞, x_mid)  →  ν_∞ + the transition
        # midpoint x_mid (ν is a Basics dial; the ceiling W̄ is retired)
        if _vis("x_mid", "nu_inf", "p0_w"):
            sb.subheader("Dynamics — the value slope eases")
        # D-120: the dial is the LONG-RUN value growth rate, in ×/yr since D-133, inverted through the
        # leader's capability speed at the horizon endpoint — the last speed the model states
        # (past T the ψ feedback diverges, so there is no true asymptote to read). The caption
        # names that denominator explicitly, because it is the one genuinely new modelling
        # commitment behind the dial and the reader cannot check the number without it.
        _tparam(sb, "t_value_growth_inf", "nu_inf",
                lambda x: (f"⇒ $\\nu_\\infty$ = {x:.2f} value-OOM per OOM (×{10.0**x:.2f} per "
                           f"OOM) — at the horizon's frontier speed $\\dot x^L_T$ = "
                           f"{_xdotT:.2f} OOM/yr, down from {_xdot0:.2f} today"),
                lambda a, b: (f"⇒ $\\nu_\\infty \\in$ [{a:.2f}, {b:.2f}] value-OOM per OOM "
                              f"(×{10.0**a:.2f}–×{10.0**b:.2f} per OOM)"))
        # D-111: ν (Basics) and ν_∞ (above) are this curve's endpoints; the bend's midpoint and
        # position are advanced — and its OWN button, independent of the compute curve's.
        _adv_button(sb, "value", "easing midpoint and how far into it we are today")
        # Bounds: R11 (state.dial) — the envelope [2, 20] padded to [0.5, 22]. The floor is the
        # first grid point above 0 because the curve's slope is span/x_mid, so 0 divides by zero.
        # This is the widest padding in the table in RATIO terms (0.5 vs a floor of 2), which is
        # inherent to an additive 10%-of-width rule on an envelope 10× wider than its own floor —
        # and it matters here more than elsewhere: D-107 measured x_mid as the GATE on whether
        # ν_∞ can bite at all. (Envelope [2, 20] itself stays flagged → calibration round.)
        _param(sb, "x_mid", "$x_{mid}$  transition midpoint (OOM)", *dial("x_mid"))
        # D-084: this curve's OWN position dial (the value easing is a different empirical
        # claim from the compute slowdown, so it gets its own), after the midpoint it reads.
        _param(sb, "p0_w", "$q^w_0$  how far into the easing today (%)", *dial("p0_w"),
               cap=lambda v: (f"⇒ today's slope is already **{v:.0f}%** of the way from "
                              f"$\\nu$ down to $\\nu_\\infty$; half-done at "
                              f"{d['x_mid']:.1f} OOM, {100.0 - v:.0f}% by "
                              f"{2.0 * d['x_mid']:.1f} OOM"))

    # -------------------------------------------------- Level 3: Catch-up channels
    # R7 (Pavel: "This is correct division"). ONE rung of the ladder, TWO labelled groups — and
    # the division IS the question the level exists to answer: does the follower keep up by BUYING
    # COMPUTE, or by ABSORBING THE LEADER'S PROGRESS? The two subheadings are the two answers, so
    # a reader who only reads the headings has still learned what Level 3 is for.
    #
    # Grouping only: no level machinery, pins, MC envelopes or fixtures move. In particular
    # LEVEL_RANGED[3] keeps its frozen order (mc_draw_batch consumes it in order) — this reorders
    # what is on SCREEN, not what is drawn.
    #
    # The nine parameters R7 divides are exactly `sidebar_filter_keys(3)`, but only six of them
    # have a row here: Δ0, δ_dev and δ_rel are all set by the ONE Fringe-lag observable up in
    # Basics (D-037's wedge-split inversion), so the second group says so out loud rather than
    # promising five dials and showing two. The follower's own engine stays raw and feeds the lag
    # caption's wedge live.
    #
    # `split` moves here from "Model internals (no clean observable)", which it was the last
    # occupant of and which therefore disappears. That heading grouped dials by GRADE; R7 groups
    # them by MECHANISM, and split — how much of the initial gap is algorithmic — is a spillover
    # question, not a leftover. It is still a grade-F judgment call, and its own card says so.
    if LEVEL >= 3:
        if _vis("g_CF0", "g_CF_inf", "t_mid_F", "p0_F"):
            sb.subheader("The fringe's own compute")
            sb.caption("It **buys its own compute** and runs its own curve — same shape as the "
                       "leader's, its own floor, midpoint and position.")
        _param(sb, "g_CF0", "$g_c^F$  fringe compute growth (OOM/yr)", *dial("g_CF0"))
        _param(sb, "g_CF_inf", "$g_{c\\infty}^F$  growth floor (OOM/yr)", *dial("g_CF_inf"))
        # D-111: same rule for the fringe's own curve — its two endpoints stay, its midpoint and
        # position sit behind its own button.
        _adv_button(sb, "follower", "slowdown midpoint and how far into it the fringe is today")
        _param(sb, "t_mid_F", "$t_{mid}^F$  slowdown midpoint (yr)", *dial("t_mid_F"))
        # D-084: the fringe curve's own position — never silently tied to the leader's q^c_0
        _param(sb, "p0_F", "$q^F_0$  how far into the slowdown today (%)", *dial("p0_F"),
               cap=lambda v: (f"⇒ the fringe is in the bottom **{v:.0f}%** of its curve today; "
                              f"midpoint in {d['t_mid_F']:.1f} yr, {100.0 - v:.0f}% done by "
                              f"{2.0 * d['t_mid_F']:.1f} yr"))
        if _vis("g_a_F", "split", "Delta0", "delta_dev", "delta_rel"):
            sb.subheader("Spillovers & catch-up")
            sb.caption("It **absorbs the leader's progress** — through talent and published "
                       "methods ($\\delta_{dev}$) and by distilling the released model "
                       "($\\delta_{rel}$). Both channels, and the initial gap $\\Delta_0$, are "
                       "set by the one **Fringe lag** dial in *Basics*; these two dials say how "
                       "much of the leader's algorithmic progress the fringe reproduces by "
                       "itself, and how the gap divides between algorithms and compute.")
        # X-10 (extensions-sync): the dial is the follower/leader SHARE — the same object the
        # MC prior draws (scale_of g_a) and the evidence states (Gundlach's 0.6–0.8× band), so
        # moving the effective-compute dial keeps the documented relation g_a^F = share·g_a
        # instead of silently breaking an absolute rate. Bounds: R11 (state.dial) — the envelope
        # [0.5, 0.9] padded to [0.46, 0.94].
        # d["g_a_F"] holds the SHARE until the derive pass turns it into the rate (it needs the
        # inverted g_a, which pass 1 does not have).
        _param(sb, "g_a_F", "$g_a^F$  algo rate — share of the leader's", *dial("g_a_F"),
               dial_default=0.7,
               cap=lambda s_: (f"⇒ $g_a^F$ = {s_ * d['g_a']:.2f} OOM/yr "
                               f"({100 * s_:.0f}% of the leader's $g_a$)"))
        _param(sb, "split", "algo share of $\\Delta_0$", *dial("split"))

    # ---- RETIRED levels (Pavel's ladder amendment, 2026-07-27) -------------------------
    # Release delay (old L7): x^R-parked in the spec (N9). Cost mechanism (old L8): retired —
    # φ_RD is provably inert under the observed-bill anchor; g_p survives as the calibrated
    # base constant (pinned in apply_level_pins). Extensions (old L9): retired — χ, the
    # conduct multiplier, own-compute (II.6) and the labor line (II.7) leave the widget; their
    # machinery stays in the notebook, always off (Params defaults), parked in the SPEC with
    # N9-style revival notes. τ/φ_RD/g_p are pinned unconditionally by apply_level_pins; the
    # extension flags are simply never set here, so Params(**d) keeps them off.

    # ========================================================= PASS 2 — the ONE inversion (F-1)
    # Every dial has been read; nothing above this line derives a model parameter from an
    # observable. `m.invert_targets` — the model's own inversion, the one `test_24` guards —
    # now produces g_C0, g_a, ν, ν_∞, g_c∞, Δ0, the catch-up channels and the money anchors in a
    # single call, at the FULL effective context. That is what makes the D-086 guarantee hold in
    # the app: today's compute growth, today's effective growth and the fringe lag stay on their
    # dialled values at every q₀ᶜ and every compute floor, because the residual and the lag
    # conversion finally see the Dynamics dials that the group below Basics writes.
    #
    # Seeds. `apply_level_pins` ties g_c∞ := g_c at Level 1, so the base needs g_C0 before the
    # inversion runs — and that seed is EXACT, not an approximation: it is literally what the
    # inversion returns, since D-088 made the compute inversion the identity map. From Level 2
    # the pin touches it not at all and the target supplies it anyway.
    #
    # D-120: the OTHER Level-1 tie, ν_∞ := ν, can no longer be seeded this way. ν is now the
    # log₁₀ of the value dial's ×/yr rate over the leader's exact t = 0 speed, which needs the Dynamics
    # context and the inverted g_a — precisely what pass 2 exists to supply — so seeding it here
    # would be a second, approximate copy of the inversion, and a tie made from it would leave a
    # faint value easing in Level 1's supposedly-exponential paths. The seed below is a
    # PLACEHOLDER (nothing `invert_targets` reads depends on ν or ν_∞: no capability rate, no lag,
    # no catch-up intensity), and the real tie is `apply_post_inversion_pins`, below the
    # inversion.
    d["g_C0"] = float(np.log10(tg["t_compute_x"]))
    d["nu"] = P0.nu
    apply_level_pins(d, LEVEL)
    _merged = merged_delta(LEVEL)
    _basep = m.Params(**d)
    if LEVEL >= 3:
        # X-10: the follower's algo dial is a SHARE of the leader's g_a, and the lag inversion
        # reads the follower engine (stationary_catchup), so the base needs the absolute rate
        # first. g_a is a pure function of the two speed dials and the leader's compute curve —
        # never of the follower — so taking it from the SAME inversion is exact, not iterative.
        d["g_a_F"] = float(d["g_a_F"]) * _invert()["g_a"]
        _basep = m.Params(**d)
    # Pavel's refined re-anchor rule (D-081 addendum) and the money anchors (D-076) are both
    # inside this call: EVERY dial configuration reproduces, at t = 0, the observed gap Δ0 AND
    # gap stationarity Δ̇(0) = 0 exactly (the catch-up intensity is the absorbing degree of
    # freedom), and revenue(0) = m·R₀ and cost(0) = k·R₀ hold exactly at every level. Dials
    # shape the trajectory only FORWARD of t = 0.
    d.update(_invert())
    # D-120: the Level-1 value tie, applied to the INVERTED ν (see the seed note above).
    apply_post_inversion_pins(d, LEVEL)

    # ========================================================= PASS 3 — the deferred captions
    # Each ⇒ caption writes back into its own row container, so the DOM order inside a row is
    # unchanged (caption last, after the trim lane). `_gc_today` is the model's own Γ(0) —
    # today's realised compute growth — which several captions read.
    _gc_today = m.gc_today(m.Params(**d))
    # D-120: the two capability speeds the value dials convert through — today's (closed form)
    # and the horizon endpoint's (simulated), both read off the FINAL effective parameters, so
    # the captions quote the denominator the inversion actually used.
    _effp = m.Params(**d)
    _xdot0 = m.xdot_L0(_effp)
    _xdotT = m.xdot_L_T(_effp)
    for _row, _fn in _caps:
        _row.caption(_fn())
    if _lag_row is not None:
        # X-04 (extensions-sync): the stationary construction holds the lag constant for ALL t
        # only under Level 1's steady growth; once the Dynamics bend the paths (L2+) it is
        # guaranteed AT t = 0 only — the caption must not overclaim (the forward drift is the
        # two slowdowns diverging, not an inversion defect). `content.lag_note` is the ONE place
        # that decides how strong the claim may be: this was the only site X-04 reached, and the
        # merged-δ card and its » header kept asserting "stays constant" at L2 (audit A/3).
        _lagnote = f"the lag {lag_note(LEVEL)}"
        if _lag_rng:
            _llo, _lhi = _active_span("t_lag_mo")
            _elo, _ehi = _invert({"t_lag_mo": float(_llo)}), _invert({"t_lag_mo": float(_lhi)})
            _d0s = sorted((_elo["Delta0"], _ehi["Delta0"]))
        if _merged:
            if _lag_rng:
                _des = sorted((_elo["delta_rel"], _ehi["delta_rel"]))
                _cap = (f"⇒ $\\Delta_0 \\in$ [{_d0s[0]:.2f}, {_d0s[1]:.2f}] OOM · $\\delta \\in$ "
                        f"[{_des[0]:.2f}, {_des[1]:.2f}]/yr — {_lagnote}")
            else:
                # merged δ routes through δ_rel (split_delta)
                _cap = (f"⇒ $\\Delta_0$ = {d['Delta0']:.2f} OOM · $\\delta$ = "
                        f"{d['delta_rel']:.2f}/yr $= \\dot x^L_0/\\Delta_0$ — {_lagnote}")
        else:
            if _lag_rng:
                _dv = sorted((_elo["delta_dev"], _ehi["delta_dev"]))
                _dr = sorted((_elo["delta_rel"], _ehi["delta_rel"]))
                _cap = (f"⇒ $\\Delta_0 \\in$ [{_d0s[0]:.2f}, {_d0s[1]:.2f}] OOM · "
                        f"$\\delta_{{dev}} \\in$ [{_dv[0]:.2f}, {_dv[1]:.2f}] · "
                        f"$\\delta_{{rel}} \\in$ [{_dr[0]:.2f}, {_dr[1]:.2f}]/yr — "
                        f"{_lagnote}")
            else:
                _cap = (f"⇒ $\\Delta_0$ = {d['Delta0']:.2f} OOM · $\\delta_{{dev}}$ = "
                        f"{d['delta_dev']:.2f} · $\\delta_{{rel}}$ = {d['delta_rel']:.2f}/yr — "
                        f"{_lagnote}")
        _lag_row.caption(_cap)

    # -------------------------------------------------- D-078: expand the open accordion row
    # A plain session key survives reruns; re-injecting the style each run keeps the row
    # expanded through the background-MC rerun churn. If the level filter hid the row, the
    # selector simply matches nothing.
    if S.get("_p_open"):
        inject_row_expand_css(S["_p_open"])
    # D-079: the per-row playhead-track adornments (crop bands + inspected ticks), batched
    # into ONE style element at the sidebar end — a per-row markdown would be a flex item
    # and shift each row's vertical rhythm.
    if _track_css:
        st.sidebar.markdown("<style>" + "".join(_track_css) + "</style>",
                            unsafe_allow_html=True)
    return d
