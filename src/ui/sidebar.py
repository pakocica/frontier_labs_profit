"""The sidebar: targets-first controls (D-037) with dual-mode range/spot rows (D-041),
building the effective parameter dict d for the run. Row builders are closures over
(d, LEVEL, MC_ACTIVE); level pins come from levels.apply_level_pins.
"""
import numpy as np
import streamlit as st

from .content import INTERP, INTERP_T, TSPEC, _sub_live
from .levels import apply_level_pins
from .mc import _inspected_params
from .model_access import m, P0, TDEF
from .state import (_active_span, _commit_mode, _commit_range_s, _default_mode,
                    _default_span, _gc_sym, _mc_dim_editable, _range_mode, _reg,
                    _reset_all, _reset_full, _reset_one, _tbounds, _tbounds_of,
                    mc_active, open_cal)
from .theme import C_SAMPLE


def render(LEVEL):
    """Render the whole sidebar for the current level; returns the effective dict d that
    Params(**d) consumes (every parameter explicitly set — pins included, no fallbacks)."""
    MC_ACTIVE = mc_active()
    d = {}  # parameter dict passed to Params(**d)

    # ---- row builders (closures over d / LEVEL / MC_ACTIVE) --------------------------
    def _mode_tick(container, ekey):
        """Icon-only range/spot toggle (minimal space; meaning on hover). Ticked = RANGE mode.
        The mode of record lives in the plain `rmv_` key (registered for Reset-all), so it survives
        view switches while the checkbox widget itself is only rendered on the Monte-Carlo view."""
        _reg(f"rmv_{ekey}", _default_mode(ekey))
        mkey = f"rm_{ekey}"
        st.session_state[mkey] = _range_mode(ekey)                                # pre-inst sync
        return container.checkbox("range mode", key=mkey, label_visibility="collapsed",
                                  on_change=_commit_mode, args=(ekey,),
                                  help="Ticked: **range mode** — the Monte Carlo samples this range; "
                                       "the deterministic paths use the spot value. Unticked: **spot "
                                       "mode** — this exact value is used everywhere and the Monte "
                                       "Carlo pins the dimension.")

    def _range_control(container, ekey, label, fmt=None):
        """Two-ended MC sampling-range slider bounded by the vetted ENVELOPE, starting at the tight
        D-042 default (or its collapsed point), synced with `_range_over` (modal edits show up here
        and vice versa), plus the inspected-draw tick when ⊙ is active."""
        base = m.TARGET_RANGES.get(ekey) or m.PARAM_RANGES.get(ekey)
        env_lo, env_hi = _tbounds_of(base)
        dflt = _default_span(ekey)
        cur = st.session_state.get("_range_over", {}).get(ekey, dflt)
        skey = f"srng_{ekey}"
        st.session_state[skey] = (float(cur[0]), float(cur[1]))   # pre-instantiation sync w/ store
        step = float(f"{(env_hi - env_lo) / 200:.1g}")
        kw = {"format": fmt} if fmt else {}
        container.slider(label, float(env_lo), float(env_hi), step=step, key=skey,
                         on_change=_commit_range_s, args=(ekey, dflt),
                         label_visibility="collapsed",
                         help="MC sampling range (synced with the range editor in the calibration "
                              "details). The default is TIGHT — the span of the documented sources "
                              "(a point where only one source exists); the slider bounds are the "
                              "vetted envelope. The faint ghost bullet marks the SPOT value the "
                              "deterministic paths use."
                              + (" Endpoints are the lognormal prior's 90% CI."
                                 if base[0] == "lognormal" else ""), **kw)
        # ---- ONE zero-height overlay pulled UP onto the slider's track line (the +/− margins
        # cancel, so layout below is unaffected; a SINGLE element, so the flex gap between Streamlit
        # elements doesn't shift it). It carries BOTH track adornments:
        #   · the GHOST SPOT THUMB (D-042 addendum) — the deterministic spot value as a faint
        #     thumb-styled bullet + value label;
        #   · the INSPECTED-DRAW TICK — a thin dashed line, only while ⊙ inspection is active.
        # Different shapes/opacity keep the two distinct when both are visible.
        parts = []
        sv = st.session_state.get(f"sv_{ekey}")
        if sv is not None:
            sfrac = float(np.clip((float(sv) - env_lo) / (env_hi - env_lo), 0.0, 1.0)) * 100.0
            slab = (fmt % float(sv)) if fmt else f"{float(sv):g}"
            # geometry (measured in-browser): the overlay's zero-height line sits ~5px above the
            # track center-line; the bullet must match the real thumbs (12px, center ON the
            # track), so label(12px)+gap(4px)+bullet-half(6px) puts the container at -17px
            parts.append(
                f"<div title='spot value {slab} — used by the deterministic paths' "
                f"style='position:absolute;left:{sfrac:.1f}%;transform:translateX(-50%);"
                f"top:-17px;opacity:0.30;text-align:center;pointer-events:auto;z-index:5;'>"
                f"<div style='font-size:12px;color:#4c8dff;line-height:1;"
                f"margin-bottom:4px;white-space:nowrap;'>{slab}</div>"
                f"<div style='width:12px;height:12px;border-radius:50%;background:#4c8dff;"
                f"margin:0 auto;'></div></div>")
        ins = _inspected_params()
        if ekey in ins:
            v = float(ins[ekey])
            frac = float(np.clip((v - env_lo) / (env_hi - env_lo), 0.0, 1.0)) * 100.0
            parts.append(
                f"<div title='inspected draw: {v:.3g}' style='position:absolute;left:{frac:.1f}%;"
                f"top:-3px;height:16px;border-left:2px dashed {C_SAMPLE};opacity:0.9;"
                f"pointer-events:auto;z-index:6;'></div>")
        if parts:
            container.markdown(
                "<div style='position:relative;height:0;margin:-42px 8px 42px 8px;"
                "pointer-events:none;'>" + "".join(parts) + "</div>",
                unsafe_allow_html=True)

    def _row_head(row, label, val_txt, panel_key, pinned=False):
        """Line 1 of a compact 3-line sidebar row (D-043): label + current value + a small ⓘ
        at the far right that opens the docked calibration panel (replaces the old full-width
        interpretation popover / [details] button)."""
        # label gets ~73% of the row (values are short); one line per label is the S7 contract
        lc, vc, ic = row.columns([7.2, 1.6, 1.0], vertical_alignment="center")
        lc.markdown(label)
        vc.markdown(f"<div style='text-align:right;font-size:0.85rem;opacity:0.85;"
                    f"white-space:nowrap'>{val_txt}</div>", unsafe_allow_html=True)
        ic.button("ⓘ", key=f"i_{panel_key}", on_click=open_cal, args=(panel_key, pinned),
                  help="calibration details — sources, ranges, methodology")

    def _param(container, key, label, lo, hi, step, **kw):
        """Compact 3-line row (D-043): label+value+ⓘ, keyed slider (collapsed label) with a
        per-slider ↺ reset. Never hardcodes a default: the seed value is `getattr(P0, key)`,
        clipped into range."""
        interp = _sub_live(INTERP.get(key), d)
        default = float(np.clip(getattr(P0, key), lo, hi))
        sampled = key in m.PARAM_RANGES     # an MC dimension
        wkey = f"w_{key}"
        if sampled:
            _reg(f"sv_{key}", default)
            st.session_state.setdefault("_wdefaults", {})[wkey] = default
        else:
            _reg(wkey, default)
        row = container.container(key=f"row_{key}")
        cur = st.session_state.get(wkey, st.session_state.get(f"sv_{key}", default))
        _row_head(row, label, f"{float(cur):g}", key)
        dual = sampled and MC_ACTIVE        # dual-mode UI renders only in Monte-Carlo mode
        cols = [5.3, 0.55, 1.0] if dual else [6, 1]
        cs = row.columns(cols, vertical_alignment="bottom")
        sc, rc = cs[0], cs[-1]
        rmode = _mode_tick(cs[1], key) if dual else False
        if dual and rmode and _mc_dim_editable(key):
            st.session_state[f"_spoton_{key}"] = False
            _range_control(sc, key, label)   # the ghost bullet on the track carries the spot value
            d[key] = float(st.session_state.get(f"sv_{key}", default))
        else:
            if sampled and not st.session_state.get(f"_spoton_{key}", False):
                # transition back into spot rendering: restore the remembered spot (see _target_row)
                st.session_state[wkey] = float(np.clip(st.session_state[f"sv_{key}"], lo, hi))
            if sampled:
                st.session_state[f"_spoton_{key}"] = True
            d[key] = sc.slider(label, lo, hi, step=step, key=wkey, help=interp,
                               label_visibility="collapsed", **kw)
            if sampled:
                st.session_state[f"sv_{key}"] = float(d[key])
        rc.button("↺", key=f"r_{key}", help="reset value, mode and sampling range",
                  on_click=_reset_full, args=(wkey, default, key if sampled else None))
        return row

    def _target_row(container, tkey, panel_key):
        """Target control (bounds/default from the notebook's TARGET_RANGES/target_defaults) + ↺,
        dual-mode (D-041): RANGE mode renders the two-ended MC sampling range and returns the
        remembered spot value (which drives the deterministic paths, and renders as the ghost
        bullet on the track); SPOT mode is the single-value slider used everywhere.
        Compact 3-line row (D-043); `panel_key` is the parameter the row's ⓘ opens in the
        docked calibration panel. Returns (value, row_container)."""
        label, step, fmt = TSPEC[tkey]
        lo, hi = _tbounds(tkey)
        default = float(np.clip(TDEF[tkey], lo, hi))
        _reg(f"sv_{tkey}", default)
        wkey = f"w_{tkey}"
        st.session_state.setdefault("_wdefaults", {})[wkey] = default   # ↺ / Reset-all target
        row = container.container(key=f"row_{tkey}")
        cur = st.session_state.get(wkey, st.session_state.get(f"sv_{tkey}", default))
        _row_head(row, label, fmt % float(cur), panel_key)
        cols = [5.3, 0.55, 1.0] if MC_ACTIVE else [6, 1]
        cs = row.columns(cols, vertical_alignment="bottom")
        rmode = _mode_tick(cs[1], tkey) if MC_ACTIVE else False
        if rmode:
            st.session_state[f"_spoton_{tkey}"] = False
            _range_control(cs[0], tkey, label, fmt=fmt)
            v = float(st.session_state.get(f"sv_{tkey}", default))
        else:
            # (Re)entering spot rendering (untick, view switch, first run): restore the remembered
            # spot from sv_ BEFORE the slider instantiates. Streamlit resurrects a skipped widget's
            # key with its REGISTERED DEFAULT (its first-seed value), NOT its last value — which
            # made unticking range mode land on the default (read as "the interval's min" where the
            # two coincide, e.g. t_value_x). The _spoton_ flag limits the force-sync to transition
            # runs, so it can never clobber an incoming user drag on consecutive spot runs.
            if not st.session_state.get(f"_spoton_{tkey}", False):
                st.session_state[wkey] = float(np.clip(st.session_state[f"sv_{tkey}"], lo, hi))
            st.session_state[f"_spoton_{tkey}"] = True
            v = cs[0].slider(label, lo, hi, step=step, key=wkey, format=fmt,
                             help=INTERP_T.get(tkey), label_visibility="collapsed")
            st.session_state[f"sv_{tkey}"] = float(v)
        cs[-1].button("↺", key=f"r_{tkey}", help="reset value, mode and sampling range",
                      on_click=_reset_full, args=(wkey, default, tkey))
        return v, row


    def _tparam(container, tkey, pkey, derive, capfn, rngcapfn=None):
        """Targets-first control for ONE parameter (D-037; users interact ONLY with observables —
        the ⚙ raw-parameter unlock was removed by Pavel's D-042 ruling). The caption shows the
        implied parameter live: a point in SPOT mode, the IMAGE OF THE CURRENT SAMPLING RANGE in
        range mode (endpoints ordered numerically; the ghost bullet on the track carries the spot)."""
        v, row = _target_row(container, tkey, panel_key=pkey)
        d[pkey] = derive(v)
        if MC_ACTIVE and _range_mode(tkey) and rngcapfn is not None:
            rlo, rhi = _active_span(tkey)
            a, b = sorted((derive(float(rlo)), derive(float(rhi))))
            row.caption(rngcapfn(a, b))
        else:
            row.caption(capfn(d[pkey]))

    # ---- sidebar body ----------------------------------------------------------------
    st.sidebar.title("Assumptions")
    st.sidebar.caption(f"**Level {LEVEL}** — the controls are **observables** where one exists; the "
                       "small caption under each shows the implied parameter live. Dials without a "
                       "clean observable sit under *Model internals*. Raise the level (top of the "
                       "page) to add the next mechanism. Defaults are provisional → calibration.")


    # The reset registry is rebuilt from scratch each run, so it always lists exactly the controls the
    # current level shows. (Clear BEFORE any keyed widget is created.)
    st.session_state["_wdefaults"] = {}
    st.sidebar.button("↺ Reset all to defaults", use_container_width=True, on_click=_reset_all,
                      help="Return every visible control to its notebook default.")

    # -------------------------------------------------- global horizon (applies to EVERY graph)
    _hz = st.sidebar.segmented_control("Horizon — applies to all graphs", ["5 yr", "10 yr"],
                                       key=_reg("w_hz", "10 yr"), help=INTERP["T"])
    d["T"] = 5.0 if _hz == "5 yr" else 10.0  # None (deselected) falls back to 10 yr

    sb = st.sidebar

    # -------------------------------------------------- Level 1: Basics (always visible)
    # D-037 targets-first: the Basics controls are OBSERVABLES; the caption under each shows the
    # implied parameter(s), recomputed live.
    sb.subheader("Basics")
    _tparam(sb, "t_compute_x", "g_C0",
            lambda v: float(np.log10(v)),
            lambda x: f"⇒ ${_gc_sym()}$ = {x:.3f} OOM/yr",
            lambda a, b: f"⇒ ${_gc_sym()} \\in$ [{a:.3f}, {b:.3f}] OOM/yr")
    if LEVEL <= 8:   # the value target stays in Basics until L9 moves it into Value & demand
        _tparam(sb, "t_value_x", "nu",
                lambda v: float(np.log10(v)),
                lambda x: f"⇒ $\\nu$ = {x:.2f} value-OOM per OOM",
                lambda a, b: f"⇒ $\\nu \\in$ [{a:.2f}, {b:.2f}] value-OOM per OOM")
    _param(sb, "theta", "$\\theta$  operating margin", 0.1, 0.7, 0.05)
    _param(sb, "S0", "$S_0$  train spend (\\$B/yr)", 15.0, 100.0, 5.0)
    # ---- follower lag (months): ONE observable driving Δ0 AND the catch-up rate(s) (D-037).
    # Reading later-rendered widgets via session_state is safe: keyed widget state is final at the
    # start of the script run, so the caption is live and consistent within the run.
    _ga_now = (P0.g_a if LEVEL < 3 else
               float(np.log10(st.session_state.get("sv_t_algo_x",
                              st.session_state.get("w_t_algo_x", TDEF["t_algo_x"])))))
    _lag, _lag_row = _target_row(sb, "t_lag_mo",
                                 panel_key=("delta_total" if LEVEL <= 5 else "Delta0"))
    _speed = d["g_C0"] + _ga_now
    d["Delta0"] = _lag / 12.0 * _speed
    _lag_rng = MC_ACTIVE and _range_mode("t_lag_mo")
    if LEVEL <= 5:
        # merged inversion: δ = 12/lag, so δ·Δ0 = leader speed EXACTLY — the gap is stationary at
        # the base levels for ANY lag position (supersedes the hand-computed δ default, D-036).
        d["delta_dev"], d["delta_rel"] = m.split_delta(12.0 / _lag)
        if _lag_rng:
            _llo, _lhi = _active_span("t_lag_mo")
            _d0 = sorted((_llo / 12.0 * _speed, _lhi / 12.0 * _speed))
            _de = sorted((12.0 / _llo, 12.0 / _lhi))      # δ = 12/lag is DECREASING in the lag
            _cap = (f"⇒ $\\Delta_0 \\in$ [{_d0[0]:.2f}, {_d0[1]:.2f}] OOM · $\\delta \\in$ "
                    f"[{_de[0]:.2f}, {_de[1]:.2f}]/yr — the lag stays constant")
        else:
            _cap = (f"⇒ $\\Delta_0$ = {d['Delta0']:.2f} OOM · $\\delta$ = {12.0 / _lag:.2f}/yr "
                    f"$= ({_gc_sym()}+g_a)/\\Delta_0$ — the lag stays constant")
    else:
        # channels inversion: Δ0 from the lag; (δ_dev, δ_rel) rescaled from the joint calibration
        # so the lag is stationary at the current speeds (wedge-split, D-037).
        _own = (st.session_state.get("w_g_a_F", P0.g_a_F)
                + st.session_state.get("w_g_CF0", P0.g_CF0))
        d["delta_dev"], d["delta_rel"] = m.channels_from_lag(_lag / 12.0, _speed, _own)
        if _lag_rng:
            _llo, _lhi = _active_span("t_lag_mo")
            _d0 = sorted((_llo / 12.0 * _speed, _lhi / 12.0 * _speed))
            _clo = m.channels_from_lag(_llo / 12.0, _speed, _own)
            _chi = m.channels_from_lag(_lhi / 12.0, _speed, _own)
            _dv = sorted((_clo[0], _chi[0]))
            _dr = sorted((_clo[1], _chi[1]))
            _cap = (f"⇒ $\\Delta_0 \\in$ [{_d0[0]:.2f}, {_d0[1]:.2f}] OOM · "
                    f"$\\delta_{{dev}} \\in$ [{_dv[0]:.2f}, {_dv[1]:.2f}] · "
                    f"$\\delta_{{rel}} \\in$ [{_dr[0]:.2f}, {_dr[1]:.2f}]/yr — "
                    "the lag stays constant")
        else:
            _cap = (f"⇒ $\\Delta_0$ = {d['Delta0']:.2f} OOM · $\\delta_{{dev}}$ = "
                    f"{d['delta_dev']:.2f} · $\\delta_{{rel}}$ = {d['delta_rel']:.2f}/yr — "
                    "the lag stays constant")
    _lag_row.caption(_cap)

    # -------------------------------------------------- Level 2: Training in advance
    if 2 <= LEVEL <= 7:   # ℓ unpins here; at L8 it moves into the Cost mechanism group
        sb.subheader("Training in advance")
        _param(sb, "ell", "$\\ell$  lead time (yr)", 0.25, 3.0, 0.05)

    apply_level_pins(d, LEVEL)

    # -------------------------------------------------- Level 3: Growth engine
    if LEVEL >= 3:
        sb.subheader("Growth engine")
        _tparam(sb, "t_algo_x", "g_a",
                lambda v: float(np.log10(v)),
                lambda x: f"⇒ $g_a$ = {x:.3f} OOM/yr",
                lambda a, b: f"⇒ $g_a \\in$ [{a:.3f}, {b:.3f}] OOM/yr")

    # -------------------------------------------------- Level 4: Compute slowdown
    if LEVEL >= 4:
        sb.subheader("Compute slowdown")
        _tparam(sb, "t_floor_x", "g_C_inf",
                lambda v: float(np.log10(v)),
                lambda x: f"⇒ $g_{{c\\infty}}$ = {x:.2f} OOM/yr",
                lambda a, b: f"⇒ $g_{{c\\infty}} \\in$ [{a:.2f}, {b:.2f}] OOM/yr")

    # -------------------------------------------------- Level 6: Catch-up channels
    # Δ0, δ_dev, δ_rel are driven by the Follower-lag target in Basics (wedge-split inversion); the
    # follower's OWN engine stays raw and feeds the lag caption's wedge live.
    if LEVEL >= 6:
        sb.subheader("Catch-up channels — follower engine")
        _param(sb, "g_a_F", "$g_a^F$  algo rate (OOM/yr)", 0.1, 0.6, 0.01)
        _param(sb, "g_CF0", "$g_{c0}^F$  growth today (OOM/yr)", 0.2, 0.8, 0.05)
        _param(sb, "g_CF_inf", "$g_{c\\infty}^F$  growth floor (OOM/yr)", 0.0, 0.4, 0.01)
        _param(sb, "xi_F", "$\\xi^F$  slowdown decay (/yr)", 0.05, 1.5, 0.05)

    # -------------------------------------------------- Level 7: Release delay
    if LEVEL >= 7:
        sb.subheader("Release delay")
        _tau_def = int(round(np.clip(P0.tau * 12, 0, 3)))
        _twk = _reg("w_tau_mo", _tau_def)
        _trow = sb.container(key="row_tau")
        _row_head(_trow, "$\\tau$  release delay (months)",
                  f"{int(st.session_state.get(_twk, _tau_def))} mo", "tau")
        _tsc, _trc = _trow.columns([6, 1], vertical_alignment="bottom")
        tau_mo = _tsc.slider("$\\tau$  release delay (months)", 0, 3, step=1, key=_twk,
                             help=INTERP["tau"], label_visibility="collapsed")
        _trc.button("↺", key="r_tau", help="reset to default", on_click=_reset_one,
                    args=(_twk, _tau_def))
        d["tau"] = tau_mo / 12.0

    # -------------------------------------------------- Level 8: Cost mechanism
    # The base cost is ALREADY the compute-path mechanism with φ_RD pinned to 0 (see the pins above).
    # Level 8 ADDS the R&D overhead φ_RD and frees the price-decline dial g_p; ℓ moves here from the
    # Training-in-advance group.
    if LEVEL >= 8:
        sb.subheader("Cost mechanism")
        _param(sb, "phi_RD", "$\\phi_{RD}$  R&D markup", 0.0, 2.5, 0.1)
        _param(sb, "ell", "$\\ell$  lead time (yr)", 0.25, 3.0, 0.05)
        _tparam(sb, "t_bill_x", "g_p",
                lambda v: float(d["g_C0"] - np.log10(v)),
                lambda x: f"⇒ $g_p$ = {x:.2f} OOM/yr (given the compute-scaling target)",
                lambda a, b: f"⇒ $g_p \\in$ [{a:.2f}, {b:.2f}] OOM/yr (given the compute-scaling "
                             "target)")
        sb.caption("The discount rate $r$ is hidden (profit flows are undiscounted, no NPV); it "
                   "reappears under II.6 ownership in Extensions.")

    # -------------------------------------------------- Model internals (free dials, D-037)
    # Dials with NO clean observable (grades C/F) — judgment calls, not targets. Grouped here so the
    # targets-first sidebar reads intentional: observables above, internals below.
    if LEVEL >= 3:
        sb.subheader("Model internals (no clean observable)")
        freeze = sb.checkbox("Freeze AI assistance ($\\gamma = 0$)", key=_reg("w_freeze",
                             bool(P0.gamma == 0.0)),
                             help="Turns off the $\\psi$ RSI feedback. $\\gamma$ above ~0.42 goes "
                                  "super-exponential inside the horizon (spec N4).")
        if freeze:
            d["gamma"] = 0.0
        else:
            _param(sb, "gamma", "$\\gamma$  $\\psi$ compounding (/OOM)", 0.0, 1.0, 0.05)
        _param(sb, "rho0", "$\\rho_0$  AI R&D speedup", 0.0, 0.6, 0.05)
        if LEVEL >= 4:
            _param(sb, "xi", "$\\xi$  slowdown decay (/yr)", 0.05, 1.5, 0.05)
        if LEVEL >= 5:
            _param(sb, "x_mid", "$x_{mid}$  value bend (OOM)", 1.0, 15.0, 0.5)
        if LEVEL >= 6:
            _param(sb, "split", "algo share of $\\Delta_0$", 0.2, 0.9, 0.05)

    # -------------------------------------------------- Level 9: Extensions (value, CES, dials)
    if LEVEL >= 9:
        with sb.expander("Value & demand", expanded=False):
            c = st.container()
            _tparam(c, "t_value_x", "nu",
                    lambda v: float(np.log10(v)),
                    lambda x: f"⇒ $\\nu$ = {x:.2f} value-OOM per OOM",
                    lambda a, b: f"⇒ $\\nu \\in$ [{a:.2f}, {b:.2f}] value-OOM per OOM")
            _tparam(c, "t_profit_B", "W0",
                    lambda v: float(m.invert_targets({"t_profit_B": v}, m.Params(**d))["W0"]),
                    lambda x: f"⇒ $W_0$ = {x:,.0f} \\$B/yr",
                    lambda a, b: f"⇒ $W_0 \\in$ [{a:,.0f}, {b:,.0f}] \\$B/yr")
        with sb.expander("Research model (CES)", expanded=False):
            c = st.container()
            d["A1"] = c.checkbox("A1 benchmark ($\\dot a^L = g_a$, pure exogenous)",
                                 key=_reg("w_A1", bool(P0.A1)),
                                 help="Bypass the CES research model: algo progress is exactly $g_a$.")
            _param(c, "alpha", "$\\alpha$  experiment-compute weight", 0.0, 1.0, 0.05)
            eta_options = ["1 (weighted avg)", "0.61", "0 (Cobb-Douglas)", "-2 (complements)",
                           "min (Leontief)"]
            eta_values = {"1 (weighted avg)": 1.0, "0.61": 0.61, "0 (Cobb-Douglas)": 1e-9,
                          "-2 (complements)": -2.0, "min (Leontief)": -2.0}
            eta_default = next((o for o in eta_options
                                if abs(eta_values[o] - P0.eta) < 1e-6 and not P0.leontief),
                               "min (Leontief)" if P0.leontief else eta_options[0])
            _erow = c.container(key="row_eta")
            _row_head(_erow, "$\\eta$  CES exponent (compute–labor)",
                      str(st.session_state.get("w_eta", eta_default)).split(" ")[0], "eta")
            ec1, ec2 = _erow.columns([5.3, 0.55], vertical_alignment="bottom")
            eta_choice = ec1.selectbox("$\\eta$  CES exponent (compute–labor substitution)",
                                       eta_options, key=_reg("w_eta", eta_default),
                                       help=INTERP.get("eta"), label_visibility="collapsed")
            if MC_ACTIVE:
                _mode_tick(ec2, "eta")   # choice dimension: the tick pins/samples it (no range)
            d["leontief"] = eta_choice.startswith("min")
            d["eta"] = eta_values[eta_choice]
        with sb.expander("Extensions (dials, default off)", expanded=False):
            if st.checkbox("II.2 public-knowledge mix", key=_reg("w_ii2", False)):
                _param(st.container(), "phi_mix", "$\\phi_{mix}$", 0.0, 0.8, 0.05)
            d["theta_gap"] = st.checkbox("II.4 gap-dependent $\\theta$ (provisional)",
                                         key=_reg("w_theta_gap", bool(P0.theta_gap)))
            if d["theta_gap"]:
                _param(st.container(), "theta_Delta_scale", "$\\Delta_\\theta$ scale (OOM)", 0.2, 2.0, 0.1)
            if st.checkbox("II.5 distillation defenses", key=_reg("w_ii5", False)):
                _param(st.container(), "chi",
                       "$\\chi$ defense (cuts $\\delta_{rel}$; costs $0.4\\chi$ revenue)", 0.0, 0.35, 0.05)
            d["own_mode"] = st.checkbox("II.6 ownership / user-cost (provisional)",
                                        key=_reg("w_own_mode", bool(P0.own_mode)))
            if d["own_mode"]:
                _param(st.container(), "delta_K", "$\\delta_K$ GPU depreciation (/yr)", 0.1, 0.6, 0.05)
                _param(st.container(), "r", "$r$  discount rate (/yr, user-cost only)", 0.03, 0.15, 0.01)
            d["labor_line"] = st.checkbox("II.7 decoupled labor line", key=_reg("w_labor", bool(P0.labor_line)))
            if d["labor_line"]:
                _param(st.container(), "L0", "$L_0$ labor today (\\$B/yr)", 0.0, 60.0, 5.0)
                _param(st.container(), "g_w", "$g_w$ wage growth (/yr)", 0.0, 0.12, 0.01)

    return d
