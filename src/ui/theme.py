"""Theme: the two-hue semantic palette (D-040), CSS injections, and plotly helpers.
"""
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---- "two-hue semantic" palette (D-040): leader = BLUE family, follower = ORANGE family, on
#      every chart (deterministic model paths AND the MC panels). Derived/diagnostic series are
#      neutral GREY; red/green appear ONLY as profit-verdict semantics (break-even lines, the
#      profitability block). Where two series share a panel they also differ in lightness and
#      dash pattern (colorblind redundancy). Backgrounds/grids stay with Streamlit's theming.
BLUE, BLUE_MID, BLUE_LIGHT, BLUE_DARK = "#4c8dff", "#6fa8ee", "#a9d0ff", "#2a6fd0"
ORANGE = "#ff8a4c"
GREY, GREY_LIGHT = "#9a9a94", "#c4c4be"
PAL = dict(blue=BLUE, orange=ORANGE, red="#ff6b6b", good="#2ec26b", critical="#ff6b6b")

# Entity roles: each actor keeps ONE colour family on every tab, so the reader learns it once.
C_LEADER = BLUE             # the frontier lab(s) — its internal "developed" frontier model x^L
C_FOLLOWER = ORANGE         # open-source / competitive fringe — capability x^F
C_SERVED = BLUE_LIGHT       # released / served model x^R — leader family, lighter + dashed
C_GAP = GREY                # capability gap Δ — derived series → neutral grey
C_GAP_MED = "#b9b9b3"       # gap MC median — visible grey on the dark surface
C_PSI = GREY_LIGHT          # ψ-share diagnostic — derived, dotted
C_REV, C_PROFIT, C_CUM = BLUE, BLUE, BLUE_MID      # leader-money series → blue family
C_COST = BLUE_LIGHT         # shares the Finance panel with revenue → lighter + dashed
C_SAMPLE = GREY             # MC inspected sample — thin dotted grey
NEUTRAL = GREY              # reference / baseline lines that carry no entity identity
# Release delay τ is an ORDERED quantity → an ordinal single-hue (leader-blue) ramp
# (light→dark = more delay); identity is also carried by the legend labels.
TAU_RAMP = [BLUE_LIGHT, BLUE_MID, BLUE, BLUE_DARK]
WINDOW_COLS = [BLUE_LIGHT, BLUE_MID, BLUE_DARK]    # 3 window start-times — ordinal blue ramp


def _rgba(hex_color, a):
    """'#rrggbb' + alpha → 'rgba(...)' (band fills as shades of the series' own hue)."""
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def inject_base_css():
    # ---- readability: cap the main content on very wide monitors, and cap PROSE to a comfortable
    #      measure (~85ch) while charts keep the full container width. Sidebar is left untouched.
    st.markdown(
        """
        <style>
          [data-testid="stMain"] .block-container { max-width: 1560px; }
          /* readable line length for text blocks (captions, markdown, info/warning); charts stay wide */
          [data-testid="stMain"] [data-testid="stMarkdownContainer"],
          [data-testid="stMain"] [data-testid="stCaptionContainer"],
          [data-testid="stMain"] [data-testid="stAlertContainer"],
          [data-testid="stMain"] [data-testid="stAlertContentInfo"],
          [data-testid="stMain"] [data-testid="stAlertContentWarning"] { max-width: 68rem; }
          /* keep chart/metric containers full width even though they contain markdown */
          [data-testid="stMain"] [data-testid="stPlotlyChart"] [data-testid="stMarkdownContainer"],
          [data-testid="stMain"] [data-testid="stMetric"] [data-testid="stMarkdownContainer"]
            { max-width: none; }
          /* metric cards: let labels wrap instead of ellipsizing away their units, and keep
             long values (e.g. "never") from truncating when four cards share the tile */
          [data-testid="stMetric"] [data-testid="stMetricLabel"] p
            { white-space: normal; overflow: visible; text-overflow: clip; }
          [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.9rem; }
          /* display math (st.latex) scrolls horizontally instead of clipping at the card edge */
          .katex-display { overflow-x: auto; overflow-y: hidden; padding: 3px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_mc_slider_css():
    # Uniform fill language on the Monte-Carlo view (Pavel's design, D-042 follow-up): SOLID
    # always means "the active selection". A SPOT slider's min→thumb fill is de-emphasized
    # (the whole ~4px track div — one element carrying the fill gradient — drops to 35%
    # opacity, leaving the full-opacity thumb as the only salient mark); a RANGE slider keeps
    # the default look, whose between-handles segment is already the solid selection and the
    # outer segments already faint. Ordinary sliders outside this view are untouched.
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] [data-testid="stSlider"]
            div[role="group"] > div > div:first-child { opacity: 0.35; }
          section[data-testid="stSidebar"] [class*="st-key-srng_"] [data-testid="stSlider"]
            div[role="group"] > div > div:first-child { opacity: 1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================= plotly helper
def fig_base(title, xlabel, ylabel, height=230):
    """Theme-neutral figure sized for the narrow right chart panel (D-047): compact margins
    and fonts, title pinned to the container top with the legend on its own row below it.
    No explicit backgrounds/templates so `st.plotly_chart(fig, theme='streamlit')` can
    restyle it to follow the app's light/dark theme; trace colors come from the PAL palette.
    D-051: the figure is RESPONSIVE (autosize, no fixed layout height) — its height is driven by
    the chart container, which the shim sizes via the --chart-h CSS variable so the two stacked
    charts fill most of the panel and re-fit to the panel width (the `height` arg is unused, kept
    for call-site compatibility)."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0, y=1, yanchor="top", yref="container", pad=dict(t=4),
                   font=dict(size=13)),
        autosize=True, margin=dict(l=44, r=10, t=48, b=32),
        font=dict(size=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    font=dict(size=9)),
    )
    fig.update_xaxes(title_text=xlabel, zeroline=True, automargin=True,
                     title_font=dict(size=10))
    # automargin so a long y-axis title grows the margin instead of clipping at the tile edge
    fig.update_yaxes(title_text=ylabel, zeroline=True, automargin=True,
                     title_font=dict(size=10))
    return fig


def line(fig, x, y, name, color, dash=None, width=2):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", showlegend=True,
                             line=dict(color=color, width=width, dash=dash)))


def show(fig, key=None):
    """Render a figure with Streamlit-native theming so it follows the app's light/dark theme.
    A stable `key` lets Streamlit update the chart in place across reruns (no remount flash) —
    used by the live Monte-Carlo view. D-051: `height="stretch"` makes the chart fill its element
    container, whose height the shim drives via the --chart-h CSS variable (see _layout_css)."""
    st.plotly_chart(fig, use_container_width=True, theme="streamlit", key=key, height="stretch")


def tab_intro(what, how):
    """A consistent two-line header at the top of every tab: what it shows + how to read it."""
    st.caption(f"**What this shows.**  {what}")
    st.caption(f"**How to read it.**  {how}")



# ================================================================= D-049/D-050 panel geometry
# The right chart panel MIRRORS the native sidebar: both DEFAULT to the SAME width, derived by
# construction from Streamlit's real sidebar default. That default is 300px — a `re-resizable`
# element, width clamped to [200, 600], remembered in localStorage under `sidebarWidth` — NOT the
# 16rem sometimes assumed (verified in the installed Streamlit 1.59 bundle). Pavel's spec: equal
# defaults, both bumped +10% → PANEL_W_PX = round(1.10 · 300) = 330.
#
# D-050 makes BOTH panels drag-adjustable with snap-to-collapse, entirely CLIENT-side:
#   · the right panel's width lives in ONE root CSS variable (--charts-w on <html>), set by the
#     injected shim and persisted in localStorage; every dependent width (spacer column, header
#     cut-off, top-strip width) is a calc() off that variable, so a drag reflows the whole layout
#     live — no server round-trip, no rerun churn, and reruns/level switches can't reset it
#     (Streamlit never touches <html>'s inline style or extra <body> children);
#   · dragging the divider below PANEL_SNAP_PX collapses the panel into a PANEL_STRIP_PX-wide
#     clickable strip (arrow + vertical label); clicking the strip reopens at the PRE-drag width;
#   · the server always renders the chart column — collapse only CSS-hides it, so the MC finance
#     component stays MOUNTED and its accumulation heartbeat keeps ticking by construction;
#   · the LEFT panel (st.sidebar) gets the IDENTICAL mechanism (Pavel, v2): its native chrome —
#     re-resizable drag handle, « collapse button, collapsed control — is suppressed and the
#     same controller drives --sb-w / display:none + strip; Streamlit-side the sidebar stays
#     "expanded" forever, so its content (and every sv_/rmv_ state pattern) is always mounted.
SIDEBAR_DEFAULT_PX = 300                              # Streamlit's real re-resizable default
PANEL_SCALE = 1.10                                    # "+10%" (Pavel)
PANEL_W_PX = round(SIDEBAR_DEFAULT_PX * PANEL_SCALE)  # 330 — the shared default width
PANEL_GUTTER_PX = 8                                   # spacer column sits this much inside the panel
PANEL_SNAP_PX = 220     # released below this width → the panel snaps closed (both panels)
PANEL_W_MIN_PX = PANEL_SNAP_PX                        # expanded drag clamp: [SNAP, MAX]
PANEL_W_MAX_PX = 560
PANEL_STRIP_PX = 26                                   # collapsed strip width (both panels)

# D-051: fluid-font canvas (Pavel's chosen "D" system). On a wide monitor the whole app UI
# ZOOMS — root font-size = clamp(viewport / FONT_CANVAS_REM, FONT_BASE_PX, FONT_MAX_PX) — so the
# reading measure stays ~constant in CHARACTERS instead of the columns over-widening / lines
# running too long. The divisor is tuned so font is 14px at ≈1500px and reaches 18px by ≈1930px,
# then caps. Panel DEFAULT widths are expressed in rem (PANEL_W_REM = the 330px default at 14px)
# so they scale WITH the font; a manual divider drag still pins an absolute px width (localStorage)
# and the middle pane flexes around it — the drag feeds the MIDDLE, never the font. Because the
# default is a continuous function of the font (no flex↔canvas mode switch), there is no width
# seam as the viewport crosses the 1500px font-floor.
FONT_BASE_PX = 14
FONT_MAX_PX = 18
FONT_CANVAS_REM = 107
PANEL_W_REM = round(PANEL_W_PX / FONT_BASE_PX, 3)    # 23.571 → 330px at the 14px base

# D-051 responsive modes (thresholds from the panel/middle floors, see the v3 prototype report):
#   · WIDE  (≥ NARROW_BP): the full three-column layout (params | middle | charts).
#   · NARROW (PHONE_BP … NARROW_BP): the fixed charts column can't coexist with the middle floor,
#     so it hides and a client-injected "Graphs" tab overlays the SAME always-mounted charts
#     column over the middle (single-mount preserved — the column is only CSS-shown/hidden).
#   · PHONE (< PHONE_BP): single column; the parameter sidebar becomes a ☰ drawer.
# The mode is a client-only decision (viewport width), published as the <html> data-app-mode
# attribute by the shim; all mode CSS branches off it. Streamlit's own <768px auto-collapse /
# column-stacking is overridden deliberately (our shim owns the sidebar and column widths).
NARROW_BP_PX = 1100
PHONE_BP_PX = 620


# The D-050/D-049 shim. Runs INSIDE the component iframe against the same-origin parent DOM.
# It normally executes once per page load (the host iframe stays mounted across reruns because
# inject_frontend_js renders identical markup every run); if the iframe ever reloads, the
# install is idempotent — old elements are replaced and old listeners/observers disconnected
# via the parent-window registries (__d050L, __d050RO), never stacked.
_D050_JS = r"""
<script>(function () { try {
  var w = window.parent || window, d = w.document, ls = w.localStorage, R = d.documentElement;
  var SNAP = __SNAP__, MIN = __MIN__, MAX = __MAX__, STRIP = __STRIP__, DEF = __DEF__;
  var FBASE = __FBASE__, FMAX = __FMAX__, FCANVAS = __FCANVAS__, PWREM = __PWREM__;
  var NARROW = __NARROW__, PHONE = __PHONE__;   /* (D-051) responsive-mode viewport thresholds */

  /* (D-051) fluid-font canvas: the whole UI zooms on wide monitors so the reading measure stays
     ~constant in characters. curFont drives BOTH the root font-size and every rem-derived panel
     default; recomputed on load and on window resize. */
  var curFont = FBASE;
  function calcFont() { return Math.max(FBASE, Math.min(FMAX, w.innerWidth / FCANVAS)); }
  function applyFont() { curFont = calcFont(); R.style.fontSize = curFont + 'px'; }
  function defW() { return Math.round(PWREM * curFont); }   /* rem-derived default panel width */
  applyFont();

  /* (D-051) graph-height system: the two stacked charts should fill most of the panel height.
     Each Plotly chart height = clamp(150, min(width × aspect, per-chart vertical fill), 560) —
     drag the panel narrower and they shorten; widen it and the height caps at the fill budget so
     only the width grows. Driven client-side via the parent page's Plotly (relayout); debounced,
     and a no-op when the height already matches so MC in-place updates don't thrash. Works for
     the right column AND the narrow/phone overlay (same .st-key-chartscol charts). */
  var CHART_ASPECT = 1.0, CHART_CHROME = 210, CHART_MIN = 150, CHART_MAX = 560, _szT = 0;
  function sizeCharts() {
    var col = d.querySelector('.st-key-chartscol');
    if (!col) return;
    var cw = col.getBoundingClientRect().width - 24;   /* minus the panel's horizontal padding */
    if (cw <= 0) return;                                /* hidden (narrow, overlay closed) → skip */
    /* (D-054, revised) the per-chart vertical budget = space from the FIRST chart's top down to
       the viewport bottom, minus the inter-chart gaps and the column's bottom padding (footer
       clearance), split across n charts. Every term is measured from element POSITIONS, not from
       chart heights — the first chart's top is fixed by the chrome above it (tab strip, header,
       any expanders), the gap is the flex row-gap, the bottom padding is a CSS constant — so the
       budget is INDEPENDENT of --chart-h and cannot feed back / thrash (the old scrollHeight −
       chartsH measure was polluted whenever a plot's SVG overflowed its clamped container). */
    var charts = col.querySelectorAll('[data-testid="stPlotlyChart"]');
    var n = Math.max(1, charts.length);
    var fill;
    if (charts.length) {
      var first = charts[0].getBoundingClientRect();
      var gap = 0;
      if (charts.length > 1)                            /* flex row-gap, invariant to chart height */
        gap = Math.max(0, charts[1].getBoundingClientRect().top - charts[0].getBoundingClientRect().bottom);
      var padB = parseFloat(w.getComputedStyle(col).paddingBottom) || 0;
      fill = (w.innerHeight - first.top - (n - 1) * gap - padB - 6) / n;   /* −6 safety */
    } else {
      fill = (w.innerHeight - Math.max(0, col.getBoundingClientRect().top) - CHART_CHROME) / n;
    }
    var h = Math.round(Math.max(CHART_MIN, Math.min(Math.min(cw * CHART_ASPECT, fill), CHART_MAX)));
    if (R.style.getPropertyValue('--chart-h') !== h + 'px') {
      R.style.setProperty('--chart-h', h + 'px');
      poke(); w.setTimeout(poke, 280);                  /* nudge Plotly to re-fit (twice: the
                                                           container height needs a frame to apply
                                                           before the plot settles to it) */
    }
    /* (D-054) while the Graphs OVERLAY is showing, always re-fit explicitly: the CSS-only
       show/hide fires no resize Plotly notices, and a chart-group switch or MC re-render
       remounts plots at the stale narrow width. Debounced by sizeChartsSoon upstream. */
    if (R.getAttribute('data-graphs-open') === '1') plotlyFitSoon();
  }
  function sizeChartsSoon() { if (_szT) w.clearTimeout(_szT); _szT = w.setTimeout(sizeCharts, 90); }

  /* (D-054) explicit per-chart Plotly re-fit. The Graphs overlay opens via a pure CSS attribute
     toggle — no Streamlit re-render and (on some builds) no effective resize reaches Plotly, so
     the chart stayed at its pre-overlay ~285px width. A direct Plotly.Plots.resize on every
     visible chart in the (overlay) column is authoritative; guarded — window.Plotly is a global
     under stlite 1.57 but may be absent locally, where the poke() resize event still covers it. */
  function plotlyFit() {
    try {
      var P = w.Plotly;
      if (!P || !P.Plots || !P.Plots.resize) return;
      d.querySelectorAll('.st-key-chartscol .js-plotly-plot').forEach(function (el) {
        if (el.offsetWidth > 0) { try { P.Plots.resize(el); } catch (e) {} }
      });
    } catch (e) {}
  }
  function plotlyFitSoon() {   /* after layout settles: next frame + a settle-delay pass */
    w.requestAnimationFrame(plotlyFit);
    w.setTimeout(plotlyFit, 320);
  }

  /* (D-051) responsive MODE — a client-only decision (viewport width) published as data-app-mode
     on <html>; all mode CSS branches off it. In narrow/phone the fixed charts column can't sit
     beside the middle, so a client 'Graphs' tab (graphsOpen) overlays the SAME always-mounted
     column over the middle — the column is only CSS-shown/hidden, so the single MC mount holds.
     (Only the FUNCTION declarations live here — hoisted; the executable wiring runs after the
     listener registry __d050L is (re)initialised further down.) */
  var graphsOpen = false, drawerOpen = false, dismissRotate = false;
  /* (D-052) PHONE is decided by the SMALLER viewport dimension, so a phone stays 'phone' in
     LANDSCAPE too (e.g. 844×390): single column + ☰ drawer + full-width Graphs overlay. The old
     width-only rule put a landscape phone in 'narrow', where the 330px static sidebar ate half
     the screen and the charts column was unreachable. Desktop-narrow windows are unaffected
     (their height comfortably exceeds the phone breakpoint). */
  function modeFor() { var W = w.innerWidth;
    if (Math.min(w.innerWidth, w.innerHeight) < PHONE) return 'phone';
    return W >= NARROW ? 'wide' : 'narrow'; }
  function applyMode() {
    var m = modeFor();
    R.setAttribute('data-app-mode', m);
    if (m === 'wide') graphsOpen = false;          /* charts have their own column again */
    R.setAttribute('data-graphs-open', graphsOpen ? '1' : '0');
    var gt = d.getElementById('graphsTab');        /* mirror active state onto the injected tab */
    if (gt) gt.setAttribute('data-selected', graphsOpen ? 'true' : 'false');
    /* (D-051) PHONE: single column, the sidebar becomes a ☰ drawer; a portrait phone gets the
       polite 'rotate to landscape' suggestion until dismissed. Leaving phone resets both. */
    if (m !== 'phone') { drawerOpen = false; dismissRotate = false; }
    R.setAttribute('data-sb-drawer', drawerOpen ? '1' : '0');
    var portrait = w.innerHeight > w.innerWidth;
    R.setAttribute('data-rotate', (m === 'phone' && portrait && !dismissRotate) ? '1' : '0');
    sizeChartsSoon();                              /* re-fit chart heights to the new panel width */
  }
  /* the injected 'Graphs' tab (narrow/phone only, CSS-shown) appended beside the real pane tabs,
     and a × on the overlay. Re-added by the observer if a rerun reconciles them away. */
  function ensureGraphs() {
    /* the Graphs overlay lives with the Intro/Equations tabs; when the calibration panel opens
       the tabbed pane folds away (no .st-key-pane_tab), so close the overlay too — otherwise it
       would cover the calibration panel */
    if (graphsOpen && !d.querySelector('.st-key-pane_tab')) { graphsOpen = false; applyMode(); }
    /* (D-052) fallback selector: the stlite web build bundles Streamlit 1.57, whose segmented
       buttons may not carry data-variant — any pane_tab button then anchors the injected tab */
    var real = d.querySelector('.st-key-pane_tab button[data-variant="segmented_control"]') ||
               d.querySelector('.st-key-pane_tab button');
    if (real && !d.getElementById('graphsTab')) {
      var b = d.createElement('button');
      b.id = 'graphsTab'; b.type = 'button'; b.textContent = 'Graphs';
      b.className = real.className;                 /* copy the live (hashed) segmented-button look */
      b.setAttribute('data-variant', 'segmented_control');
      /* poke twice: charts that first laid out while the column was display:none need a
         resize to paint, and a second delayed poke covers the show transition */
      on(b, 'click', function () { graphsOpen = true; applyMode(); poke();
        w.setTimeout(poke, 260); plotlyFitSoon(); });   /* (D-054) snap charts to overlay width */
      real.parentElement.appendChild(b);
    }
    var col = d.querySelector('.st-key-chartscol');
    if (col && !d.getElementById('graphsClose')) {
      var c = d.createElement('div');
      c.id = 'graphsClose'; c.textContent = '×'; c.title = 'close charts';
      on(c, 'click', function () { graphsOpen = false; applyMode(); });
      d.body.appendChild(c);
    }
    /* (D-051) phone controls — the ☰ that slides the parameter sidebar in as a drawer, and the
       portrait 'rotate' suggestion with a fall-through to the single-column view. Body-level, so
       reruns never remove them; visibility is gated by data-app-mode / data-rotate in CSS. */
    if (!d.getElementById('sbDrawerBtn')) {
      var h = d.createElement('div');
      h.id = 'sbDrawerBtn'; h.textContent = '☰'; h.title = 'parameters';
      on(h, 'click', function () { drawerOpen = !drawerOpen; applyMode(); });
      d.body.appendChild(h);
    }
    if (!d.getElementById('sbScrim')) {
      var sc = d.createElement('div'); sc.id = 'sbScrim';
      on(sc, 'click', function () { drawerOpen = false; applyMode(); });
      d.body.appendChild(sc);
    }
    if (!d.getElementById('rotateOverlay')) {
      var o = d.createElement('div'); o.id = 'rotateOverlay';
      o.innerHTML = '<div class="rot-ic">📱</div>' +
        '<div class="rot-h">This explorer works best in landscape</div>' +
        '<div class="rot-p">Rotate your device, or continue in a simplified single-column view.</div>' +
        '<a id="rotateContinue">Continue anyway →</a>';
      d.body.appendChild(o);
      on(d.getElementById('rotateContinue'), 'click', function () { dismissRotate = true; applyMode(); });
    }
  }

  /* ---- idempotent re-install: the script normally runs ONCE per page load (the host iframe
     stays mounted across reruns); if the iframe ever reloads, listeners/observers from the
     dead realm are replaced via the parent-window registries, never stacked */
  if (w.__d050RO) { try { w.__d050RO.disconnect(); } catch (e) {} }
  (w.__d050L || []).forEach(function (p) { try { p[0].removeEventListener(p[1], p[2], p[3]); } catch (e) {} });
  w.__d050L = [];
  w.__d050Applies = [];   /* (D-051) per-install list of panel apply() closures for resize reflow */
  function on(tgt, type, fn, cap) { tgt.addEventListener(type, fn, cap); w.__d050L.push([tgt, type, fn, cap]); }
  ['chartsDivider', 'chartsStrip', 'chartsClose', 'sbDivider', 'sbStrip', 'sbClose']
    .forEach(function (id) { var e = d.getElementById(id); if (e) e.remove(); });
  function mk(id, html, title) { var e = d.createElement('div'); e.id = id; e.innerHTML = html;
    e.title = title; d.body.appendChild(e); return e; }
  /* the resize poke must use the PARENT realm's constructor (cross-realm events fail the
     host page's instanceof guards — browser-verified) */
  function poke() { try { w.dispatchEvent(new w.Event('resize')); } catch (e) {} } /* plotly */

  /* (D-049) the level selector stays click/arrow-navigable but not typable */
  function lvlFix() {
    d.querySelectorAll('.st-key-topbar [data-testid="stSelectbox"] input').forEach(function (el) {
      el.readOnly = true; el.setAttribute('inputmode', 'none'); el.style.caretColor = 'transparent'; });
  }

  /* ---- ONE controller, BOTH panels (D-050 v2): identical drag / snap / strip behavior.
     Width lives in a root CSS variable (the layout is calc()'d off it), collapse in a root
     attribute; both persist in localStorage, so reruns/level switches/reloads keep them. */
  function panel(side, lsPre, varName, fromLeft, label, openTitle, closeGlyph, closeTitle) {
    var kW = lsPre + 'W', kC = lsPre + 'C';
    var aC = 'data-' + side + '-collapsed', aD = 'data-' + side + '-dragging';
    /* (D-051) a stored width means the user has DRAGGED this panel → pin that px width; otherwise
       the width tracks the rem-derived default defW() and reflows live as the font scales. */
    var stored = ls.getItem(kW), hasW = stored != null;
    var W = hasW ? parseInt(stored, 10) : defW();
    if (hasW && !(W >= MIN && W <= MAX)) { hasW = false; W = defW(); }
    var C = ls.getItem(kC) === '1';
    function apply() { R.style.setProperty(varName, (C ? STRIP : (hasW ? W : defW())) + 'px');
      if (C) R.setAttribute(aC, '1'); else R.removeAttribute(aC); }
    function save() { hasW = true; ls.setItem(kW, String(W)); ls.setItem(kC, C ? '1' : '0'); }
    (w.__d050Applies || (w.__d050Applies = [])).push(apply);   /* resize → re-eval rem default */
    apply();
    var arrOpen = fromLeft ? '\u00bb' : '\u00ab';   /* strip arrow points the OPEN direction */
    var strip = mk(side + 'Strip',
      '<span class="d050-arr">' + arrOpen + '</span><span class="d050-lbl">' + label + '</span>',
      openTitle);
    on(strip, 'click', function () { C = false; apply(); save(); poke(); });
    var closer = mk(side + 'Close', closeGlyph, closeTitle);
    on(closer, 'click', function () { C = true; apply(); save(); });
    var div = mk(side + 'Divider', '', 'drag to resize; release narrow to collapse');
    var drag = null;
    on(div, 'pointerdown', function (e) { e.preventDefault();
      if (!hasW) W = defW();             /* seed from the live rem default before the first drag */
      drag = { w0: W };
      try { div.setPointerCapture(e.pointerId); } catch (err) {}   /* iframes can't eat moves */
      R.setAttribute(aD, '1'); });
    on(div, 'pointermove', function (e) { if (!drag) return;
      var nw = fromLeft ? e.clientX : (w.innerWidth - e.clientX);
      if (nw < SNAP) { C = true; }                                  /* live snap preview */
      else { C = false; W = Math.round(Math.min(MAX, Math.max(MIN, nw))); }
      apply(); });
    function end() { if (!drag) return;
      if (C) W = drag.w0;                /* strip reopens at the PRE-drag width, not the min */
      drag = null; R.removeAttribute(aD); apply(); save(); poke(); }
    on(div, 'pointerup', end); on(div, 'pointercancel', end);
  }
  panel('charts', 'd050Charts', '--charts-w', false, 'CHARTS',
        'open the chart panel', '\u00bb', 'collapse the chart panel');
  panel('sb', 'd050Sb', '--sb-w', true, 'PARAMETERS',
        'open the parameter panel', '\u00ab', 'collapse the parameter panel');

  /* (D-051) window resize \u2192 rescale the font and reflow any non-dragged (rem-default) panels.
     Must NOT call poke() here: poke() dispatches its OWN 'resize' (to nudge Plotly after a
     collapse/drag that fires no native resize), so poking from the resize handler would recurse
     infinitely. A native resize already reaches Plotly; a poke()-driven resize just reflows. */
  on(w, 'resize', function () { applyFont(); applyMode();
    (w.__d050Applies || []).forEach(function (f) { f(); }); sizeChartsSoon(); });

  /* (D-051) executable Graphs wiring — here, AFTER the reset re-initialised __d050L so on()
     can register. Picking a real text tab closes the charts overlay; then publish the mode. */
  on(d, 'click', function (e) {
    var t = e.target; if (!t || !t.closest) return;
    var seg = t.closest('.st-key-pane_tab button');   /* (D-052) variant-attr-free: works on 1.57 too */
    if (seg && seg.id !== 'graphsTab' && graphsOpen) { graphsOpen = false; applyMode(); }
  });
  applyMode();

  /* the native sidebar must stay EXPANDED from Streamlit's point of view — our mechanism owns
     collapse (display:none + strip), so its content stays mounted. If Streamlit auto-collapsed
     it (narrow first paint), click it back open once. */
  function sbForceOpen() {
    var s = d.querySelector('section[data-testid="stSidebar"]');
    if (s && s.getAttribute('aria-expanded') === 'false') {
      var b = d.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
              d.querySelector('[data-testid="stExpandSidebarButton"]');
      if (b) b.click();
    }
  }

  /* hover-reveal for the two collapse controls (the old sidebar's « idiom, mirrored) */
  on(d, 'mouseover', function (e) {
    var t = e.target; if (!t || !t.closest) return;
    var oc = t.closest('.st-key-chartscol') || t.closest('#chartsClose') || t.closest('#chartsDivider');
    var os = t.closest('section[data-testid="stSidebar"]') || t.closest('#sbClose') || t.closest('#sbDivider');
    if (oc) R.setAttribute('data-charts-hover', '1'); else R.removeAttribute('data-charts-hover');
    if (os) R.setAttribute('data-sb-hover', '1'); else R.removeAttribute('data-sb-hover');
  });

  /* ---- ONE observer for everything that must survive reruns ---- */
  lvlFix(); sbForceOpen(); ensureGraphs(); sizeChartsSoon();
  w.__d050RO = new MutationObserver(function () { lvlFix(); sbForceOpen(); ensureGraphs(); sizeChartsSoon(); });
  w.__d050RO.observe(d.body, { childList: true, subtree: true,
                               attributes: true, attributeFilter: ['aria-expanded'] });
} catch (e) {} })();</script>
"""


def _frontend_js_html():
    """The final shim markup with the geometry constants baked in (pure — testable)."""
    return (_D050_JS
            .replace("__SNAP__", str(PANEL_SNAP_PX))
            .replace("__MIN__", str(PANEL_W_MIN_PX))
            .replace("__MAX__", str(PANEL_W_MAX_PX))
            .replace("__STRIP__", str(PANEL_STRIP_PX))
            .replace("__DEF__", str(PANEL_W_PX))
            .replace("__FBASE__", str(FONT_BASE_PX))
            .replace("__FMAX__", str(FONT_MAX_PX))
            .replace("__FCANVAS__", str(FONT_CANVAS_REM))
            .replace("__PWREM__", str(PANEL_W_REM))
            .replace("__NARROW__", str(NARROW_BP_PX))
            .replace("__PHONE__", str(PHONE_BP_PX)))


def inject_frontend_js():
    """The widget's ONE injected-JS pass (D-049 + D-050), in a 0-height iframe that reaches the
    same-origin parent. Rendered EVERY run with IDENTICAL markup: Streamlit then keeps the same
    iframe mounted across reruns (skipping the render on later runs would unmount it and kill
    the shim's observers and event listeners with its realm — the previous round's bug)."""
    with st.container(key="fejs"):
        components.html(_frontend_js_html(), height=0)


# ======================================================================= D-043 layout CSS
def _layout_css(light):
    """The full layout stylesheet (pure — testable). Theme colors resolve from `light`; every
    panel-dependent width is a calc() off the ONE --charts-w root variable (D-050): the shim
    moves that variable during a divider drag and the whole layout (panel, spacer column,
    header cut-off, top strip) reflows live, collapsed state included (--charts-w = strip)."""
    # the sticky bar needs an OPAQUE theme-matched background (content scrolls under it) and
    # the right chart panel the SIDEBAR's background; `--background-color` is not a defined
    # CSS var in Streamlit 1.59, so the app theme is resolved server-side (same best-effort
    # read the MC component uses). Streamlit default secondary backgrounds: light #f0f2f6,
    # dark #262730.
    topbar_bg = "#ffffff" if light else "#0e1117"
    panel_bg = "#f0f2f6" if light else "#262730"
    text_col = "#31333F" if light else "#fafafa"
    css = """
        <style>
          /* D-050: the ONE panel-width variable. The shim overrides it inline on <html>
             (persisted in localStorage); this rule is only the first-paint default. */
          :root { --charts-w: __PANEL_W__; --sb-w: __PANEL_W__; }

          /* D-050 v2 (Pavel: IDENTICAL behavior on both sides): the LEFT panel is st.sidebar
             with its native chrome SUPPRESSED — width imposed by --sb-w (re-resizable's
             inline width loses to !important; its drag handle and « button are hidden and
             replaced by our injected mirror controls), collapse = display:none + the strip.
             aria-expanded stays true forever, so the sidebar content is ALWAYS mounted and
             the sv_/rmv_ hidden-row state patterns are untouched. */
          section[data-testid="stSidebar"] {
            width: var(--sb-w) !important; min-width: var(--sb-w) !important;
            max-width: var(--sb-w) !important; transition: none !important; }
          section[data-testid="stSidebar"] > div:first-child {
            width: var(--sb-w) !important; transition: none !important; }
          section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
          section[data-testid="stSidebar"] div[style*="col-resize"]
            { display: none !important; }
          html[data-sb-collapsed] section[data-testid="stSidebar"] { display: none; }
          /* (D-053) drop the dead air above the "Parameters" title: the sidebar header strip
             collapses to its buttons and the user content loses its ~6rem top padding */
          section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]
            { padding-top: 0.8rem; padding-bottom: 2.4rem; /* clears the full-width footer */ }
          section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]
            { padding-top: 0.4rem; padding-bottom: 0; min-height: 0; }
          section[data-testid="stSidebar"] h1 { padding-top: 0; }

          /* frozen top bar. Streamlit wraps every element in a stLayoutWrapper exactly its
             own height, so `sticky` must go on the WRAPPER (a sticky child of a same-height
             parent has no room to stick — verified in-browser). `top` must clear Streamlit's
             own fixed header (3.75rem, opaque, z-index ~1e6), which would otherwise cover a
             bar stuck at top: 0 once the page scrolls. */
          [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {
            position: sticky; top: 3.75rem; z-index: 92;
            background-color: __TOPBAR_BG__;
            border-bottom: 1px solid rgba(128,128,128,0.25); }
          .st-key-topbar { padding: 0.15rem 0 0.35rem 0; }
          /* the main block sits closer to the header so the bar reads as part of the chrome;
             the small bottom padding keeps the fixed-height columns row inside one viewport;
             the horizontal padding drops Streamlit's 5rem default (an 80px void between the
             sidebar and the content) to a normal gutter */
          [data-testid="stMainBlockContainer"]
            { padding: 4.2rem 1.5rem 1rem 1.5rem; }

          /* ---- D-047 main region: [cal panel + fold strip (when open) | middle pane (flex)]
             | right chart panel (fixed, drag-collapsible). Rows must NOT wrap — a percentage
             or fixed flex-basis ignores its siblings' widths and would push columns onto a
             second row (verified in-browser). */
          [data-testid="stHorizontalBlock"]:has(.st-key-mainpane),
          [data-testid="stHorizontalBlock"]:has(.st-key-calpanel)
            { flex-wrap: nowrap !important; }
          /* the injected-JS iframe (inject_frontend_js) carries no visual — collapse it */
          .st-key-fejs { height: 0 !important; min-height: 0 !important;
            margin: 0 !important; overflow: hidden; }
          /* D-048/050: the right chart panel MIRRORS the sidebar — full height from the very
             top of the window, the sidebar's background, and the live width --charts-w
             (default 330px = 300px sidebar +10%; drag the divider to resize, D-050). The
             fixed panel sits out of flow; the spacer column below reserves its footprint
             inside the block, so the middle pane ends a gutter before it. The top padding
             clears the injected » collapse control. Collapsed, the panel is display:none
             (STILL MOUNTED — the MC component keeps ticking) and only the strip shows. */
          .st-key-chartscol {
            position: fixed; top: 0; right: 0; width: var(--charts-w);
            height: 100vh; overflow-y: auto; overflow-x: hidden; z-index: 99;
            background-color: __PANEL_BG__;
            border-left: 1px solid rgba(128,128,128,0.2);
            padding: 2.3rem 0.7rem 2.4rem; /* bottom clears the full-width footer (D-053) */ }
          html[data-charts-collapsed] .st-key-chartscol { display: none; }
          [data-testid="stColumn"]:has(.st-key-chartscol) {
            flex: 0 0 calc(var(--charts-w) - __GUTTER__) !important;
            min-width: calc(var(--charts-w) - __GUTTER__) !important; }
          /* the Streamlit header stops where the right panel (or its strip) begins — the
             ⋮ menu stays reachable left of it */
          header[data-testid="stHeader"] { right: var(--charts-w) !important;
            width: auto !important; }
          /* the top strip belongs to the MIDDLE area only (width calc, not margin — the
             wrapper is width:100%, so a margin would overflow instead of shrinking it) */
          [data-testid="stLayoutWrapper"]:has(> .st-key-topbar)
            { width: calc(100% - (var(--charts-w) - __GUTTER__)) !important; }
          /* top strip anchoring: Streamlit columns are PROPORTIONAL, so on a very wide panel
             fractional ratios leave slack and the left group drifts inward. Override the flex
             basis per column instead: label + ⓘ columns shrink to CONTENT width, the picker
             column is a fixed 11.5rem, and the mode-switch column absorbs ALL remaining width —
             so "Model:" pins to the true left edge and the switch to the true right edge at any
             panel width. (The switch's container is shrink-wrapped under testid stButtonGroup —
             not stSegmentedControl — hence the auto margin; same DOM on 1.57 and 1.59.) */
          .st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
          /* …except on phone, where the row may be narrower than the two groups combined */
          html[data-app-mode="phone"] .st-key-topbar [data-testid="stHorizontalBlock"]
            { flex-wrap: wrap !important; }
          .st-key-topbar [data-testid="stColumn"] { min-width: 0 !important; }
          .st-key-topbar [data-testid="stColumn"]:has(.model-label)
            { flex: 0 0 auto !important; width: auto !important; }
          .st-key-topbar [data-testid="stColumn"]:has([data-testid="stSelectbox"])
            { flex: 0 0 11.5rem !important; width: 11.5rem !important; }
          .st-key-topbar [data-testid="stColumn"]:has(.st-key-levelinfo)
            { flex: 0 0 auto !important; width: auto !important; }
          .st-key-topbar [data-testid="stColumn"]:has([data-testid="stButtonGroup"])
            { flex: 1 1 0 !important; }
          .st-key-topbar [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"])
            { margin-left: auto; }
          .st-key-topbar [data-testid="stButtonGroup"] { margin-left: auto; }
          /* the "Model:" label sits flush against the level selector, vertically centred */
          .st-key-topbar .model-label { text-align: right; font-weight: 600; opacity: 0.8;
            white-space: nowrap; font-size: 0.95rem; }
          /* the ⓘ level-explainer popover: hidden on wide (the selectbox tooltip covers it),
             shown on narrow/phone where hover doesn't exist */
          .st-key-levelinfo { display: none; }
          html[data-app-mode="narrow"] .st-key-levelinfo,
          html[data-app-mode="phone"]  .st-key-levelinfo { display: block; }
          .st-key-levelinfo [data-testid="stPopover"] button { min-height: 1.6rem;
            padding: 0 0.4rem; opacity: 0.7; } .st-key-levelinfo button:hover { opacity: 1; }

          /* ---- D-051 page heading (spans the middle pane, above the sticky top strip) ---- */
          [data-testid="stLayoutWrapper"]:has(> .st-key-apptitle)
            { width: calc(100% - (var(--charts-w) - __GUTTER__)) !important; }
          /* (D-053) the title reads as a BAR: centered, panel-grey like the side panels.
             Streamlit's markdown/element wrappers carry stray margins (-18px bottom on 1.57's
             markdown, same class of bug as the footer) that cropped the subtitle out of the
             bar — neutralize ALL inner margins so the title+subtitle sit as one centered
             block with balanced padding on 1.59 (local) and 1.57 (stlite) alike. */
          .st-key-apptitle { padding: 0.6rem 1rem 0.7rem; text-align: center;
            background: __PANEL_BG__; border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.15); }
          .st-key-apptitle div { margin: 0 !important; }
          /* the prose max-width cap (base CSS) would pin the title/footer text left — lift it */
          .st-key-apptitle [data-testid="stMarkdownContainer"],
          .st-key-appfooter [data-testid="stMarkdownContainer"]
            { max-width: none !important; width: 100%; text-align: center; }
          .st-key-apptitle .apptitle-main { font-weight: 700; font-size: 1.85rem;
            line-height: 1.12; letter-spacing: -0.01em; }
          /* padding, not margin — the margin-neutralizer above would eat a margin-top */
          .st-key-apptitle .apptitle-sub { font-size: 0.98rem; opacity: 0.6; padding-top: 0.15rem; }

          /* ---- D-051/D-053 author footer: quiet, FULL page width (spans over both side
             panels — z-index above the fixed charts column and the native sidebar) ---- */
          .st-key-appfooter { position: fixed; bottom: 0; left: 0; z-index: 999995;
            width: 100%;
            background: __TOPBAR_BG__; border-top: 1px solid rgba(128,128,128,0.18);
            padding: 0.25rem 0 0.35rem; text-align: center;
            min-height: 1.7rem; align-items: center; justify-content: center; }
          /* Streamlit's markdown wrapper carries a -18px bottom margin that collapses the
             fixed bar to ~12px and pushes the text below the viewport — neutralize inside */
          .st-key-appfooter div { margin: 0 !important; }
          .st-key-appfooter .appfooter-inner { font-size: 0.72rem; opacity: 0.55; }
          .st-key-appfooter .appfooter-sep { margin: 0 0.55rem; }
          .st-key-appfooter a { color: inherit; text-decoration: underline; }
          .st-key-appfooter a:hover { color: #4c8dff; opacity: 1; }
          /* the middle tabbed pane = the flexible remainder */
          [data-testid="stColumn"]:has(.st-key-mainpane) {
            flex: 1 1 0 !important; min-width: 320px !important; }
          /* cal panel open: the pane folds into the strip and the panel inherits its flex
             space — the charts column is FIXED, so the swap is width-neutral for it by
             construction. The cap keeps ultrawide windows from stretching the panel's cards;
             the auto margin then pins the charts to the right edge. */
          [data-testid="stColumn"]:has(.st-key-calpanel) {
            flex: 1 1 0 !important; min-width: 320px !important; max-width: 560px !important; }
          [data-testid="stHorizontalBlock"]:has(.st-key-calpanel)
            [data-testid="stColumn"]:has(.st-key-chartscol)
            { margin-left: auto; }
          .st-key-calpanel { border-right: 1px solid rgba(128,128,128,0.25);
            padding-right: 10px; }
          /* independent scroll containers: the pane and the cal panel scroll on their own
             (the chart panel scrolls itself — it is fixed full-height above). Height =
             viewport minus the chrome above the row (header 3.75rem + block padding 4.2rem
             + top bar ~3rem + gap 1rem + breathing ≈ 12.5rem); the max() keeps very short
             windows usable (the page then scrolls normally) */
          [data-testid="stColumn"]:has(.st-key-mainpane),
          [data-testid="stColumn"]:has(.st-key-calpanel) {
            height: max(calc(100vh - 12.5rem), 320px);
            overflow-y: auto; overflow-x: hidden; }

          /* the folded tabbed-pane strip: a thin vertical click target */
          [data-testid="stColumn"]:has(.st-key-eqstrip) {
            flex: 0 0 52px !important; min-width: 52px !important; }
          .st-key-eqstrip button
            { writing-mode: vertical-rl; height: 340px; width: 38px;
              padding: 10px 2px; font-size: 12px; color: inherit; opacity: 0.75; }
          .st-key-eqstrip button:hover { opacity: 1; }

          /* ================= D-050 injected panel controls =================
             the #*Divider / #*Strip / #*Close pairs are plain divs the shim
             appends to <body> — OUTSIDE Streamlit's element tree, so reruns never remove
             them; visibility is gated by the html-root attributes the shim maintains. */
          /* the drag dividers straddle each panel's inner edge; a slim accent line appears
             on hover / while dragging (VS Code idiom) — identical on both sides */
          #chartsDivider, #sbDivider { position: fixed; top: 0; height: 100vh; width: 9px;
            z-index: 999996; cursor: col-resize; }
          #chartsDivider { right: calc(var(--charts-w) - 4px); }
          #sbDivider { left: calc(var(--sb-w) - 4px); }
          #chartsDivider::after, #sbDivider::after { content: ""; position: absolute;
            left: 3px; top: 0; height: 100%; width: 3px; background: #4c8dff; opacity: 0;
            transition: opacity 0.1s; }
          #chartsDivider:hover::after,
          html[data-charts-dragging] #chartsDivider::after,
          #sbDivider:hover::after,
          html[data-sb-dragging] #sbDivider::after { opacity: 0.55; }
          html[data-charts-collapsed]:not([data-charts-dragging]) #chartsDivider
            { display: none; }
          html[data-sb-collapsed]:not([data-sb-dragging]) #sbDivider { display: none; }
          html[data-charts-dragging] *, html[data-sb-dragging] *
            { user-select: none !important; }

          /* the collapsed strips (both panels, identical build): arrow on top + vertical
             letter label, sidebar-colored, whole strip clickable with a hover tint */
          #chartsStrip, #sbStrip { display: none; position: fixed; top: 0; height: 100vh;
            width: __STRIP__; z-index: 999997; background: __PANEL_BG__; cursor: pointer;
            flex-direction: column; align-items: center; gap: 0.45rem;
            padding-top: 0.45rem; user-select: none; color: __TX__; }
          #chartsStrip { right: 0; border-left: 1px solid rgba(128,128,128,0.25); }
          #sbStrip { left: 0; border-right: 1px solid rgba(128,128,128,0.25); }
          html[data-charts-collapsed] #chartsStrip { display: flex; }
          html[data-sb-collapsed] #sbStrip { display: flex; }
          #chartsStrip:hover, #sbStrip:hover { background: rgba(76,141,255,0.14); }
          .d050-arr { font-size: 1.5rem; line-height: 1.1; opacity: 0.6; }
          .d050-lbl { writing-mode: vertical-rl; text-orientation: upright;
            font-size: 0.68rem; letter-spacing: 0.28em; opacity: 0.55; }
          #chartsStrip:hover .d050-arr, #chartsStrip:hover .d050-lbl,
          #sbStrip:hover .d050-arr, #sbStrip:hover .d050-lbl
            { opacity: 1; color: #4c8dff; }

          /* the collapse controls: » at the chart panel's top-left, « at the parameter
             panel's top-right — same 1.5rem glyph, same hit area, hover-revealed, mirror
             images of each other (D-050) */
          #chartsClose, #sbClose { position: fixed; top: 0.35rem; z-index: 999996;
            font-size: 1.5rem; line-height: 1; min-height: 2rem;
            padding: 0.2rem 0.4rem; cursor: pointer; opacity: 0; color: __TX__;
            transition: opacity 0.15s; }
          #chartsClose { right: calc(var(--charts-w) - 2.7rem); }
          #sbClose { left: calc(var(--sb-w) - 2.7rem); }
          html[data-charts-hover] #chartsClose { opacity: 0.6; }
          html[data-sb-hover] #sbClose { opacity: 0.6; }
          #chartsClose:hover, #sbClose:hover { opacity: 1 !important; color: #4c8dff; }
          html[data-charts-collapsed] #chartsClose { display: none; }
          html[data-sb-collapsed] #sbClose { display: none; }

          /* the native collapsed-sidebar control never shows (our strip replaces it); pad
             the main area so content never slides under the strip */
          [data-testid="stSidebarCollapsedControl"],
          [data-testid="stExpandSidebarButton"] { display: none !important; }
          html[data-sb-collapsed] [data-testid="stMain"] { padding-left: __STRIP__; }

          /* the ⓘ on the equation cards: same glyph idiom as the sidebar rows */
          [class*="st-key-ieq_"] button
            { border: none; background: transparent; padding: 0 2px; min-height: 1.2rem;
              line-height: 1.2; font-size: 0.85rem; opacity: 0.65; }
          [class*="st-key-ieq_"] button:hover { opacity: 1; color: #4c8dff; }
          /* the ⌖ click-to-highlight glyph in each equations subsection (D-048) */
          [class*="st-key-eqhl_"] { display: flex; justify-content: flex-end;
            margin-top: -0.4rem; }
          [class*="st-key-eqhl_"] button
            { border: none; background: transparent; padding: 0 2px; min-height: 1.2rem;
              line-height: 1.2; font-size: 0.95rem; opacity: 0.6; }
          [class*="st-key-eqhl_"] button:hover { opacity: 1; color: #4c8dff; }

          /* compact sidebar rows: tighter vertical rhythm + flush label lines */
          section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
          section[data-testid="stSidebar"] [class*="st-key-row_"]
            { padding: 2px 6px 4px; border-radius: 10px; }
          section[data-testid="stSidebar"] [class*="st-key-row_"] p { margin-bottom: 0; }
          /* no thumb-value labels inside the compact rows: the current value already sits
             right-aligned on the label line and the implied interval in the caption, so the
             floating numbers only collided with the row text above (QA S7) */
          section[data-testid="stSidebar"] [class*="st-key-row_"]
            [data-testid="stSliderThumbValue"] { display: none; }
          /* slightly smaller row labels so each stays on ONE line in the ~215px label cell;
             captions get their own (smaller) size back since the label rule would inflate them */
          section[data-testid="stSidebar"] [class*="st-key-row_"]
            [data-testid="stMarkdownContainer"] p { font-size: 0.92rem; }
          section[data-testid="stSidebar"] [class*="st-key-row_"]
            [data-testid="stCaptionContainer"] p { font-size: 0.78rem; }
          /* the tiny ⓘ (and per-row ↺): strip the button chrome down to a glyph */
          section[data-testid="stSidebar"] [class*="st-key-row_"] button,
          .st-key-calpanel .st-key-calclose button
            { border: none; background: transparent; padding: 0 2px; min-height: 1.2rem;
              line-height: 1.2; font-size: 0.85rem; opacity: 0.65; }
          section[data-testid="stSidebar"] [class*="st-key-row_"] button:hover
            { opacity: 1; color: #4c8dff; }

          /* ---- D-051 graph-height: the shim sets --chart-h (clamp of panel-width × aspect and
             the vertical fill); the responsive (autosize, height="stretch") plotly charts fill a
             container pinned to it, so the two stacked charts occupy most of the panel and re-fit
             to the panel width. CSS-driven (not per-chart relayout) → survives the background
             MC-accumulation reruns.
             (D-054) A plotly chart's height is pinned to whatever --chart-h reads when Streamlit
             RENDERS the chart element; Streamlit then REVERTS any later client change (a pure CSS
             --chart-h change or a Plotly.relayout never re-fits an already-mounted plot — verified
             on BOTH local 1.59 and deployed stlite 1.57). So the mounted plots are effectively
             this default value, uniform across tabs. Make it a viewport-fitted clamp (not a flat
             300px) so a two-chart stack fits a short laptop on load, and size the chrome term to
             the LARGEST-chrome 2-chart group so NO 2-chart group overflows. After the D-054 round-2
             declutter (section headers + the how-to-read expander removed) BOTH groups (Graphs and
             Capability) have ≈9.7rem of chrome (measured at the 14px font floor of wide mode; a
             touch less at 18px since the fixed tab-strip px shrink in rem terms), so a single
             constant fits both AND the group-switch carry-over can't overflow (equal budgets).
             10.5rem sits just above 9.7 with a small cushion. The chrome is in rem so it tracks the
             fluid font. The shim then refines the CONTAINER via --chart-h (correct per group &
             width); the plot re-fits to it on the next rerun (any interaction). */
          :root { --chart-h: clamp(150px, calc((100vh - 10.5rem) / 2), 560px); }
          .st-key-chartscol [data-testid="stPlotlyChart"] { height: var(--chart-h) !important; }

          /* ================= D-051 responsive modes (data-app-mode = wide|narrow|phone) =========
             WIDE is the untouched three-column layout. NARROW: the fixed charts column can't dock
             beside the middle, so it hides and the injected 'Graphs' tab overlays the SAME
             always-mounted column over the middle (single MC mount preserved — only CSS toggles). */
          /* the injected Graphs tab: hidden in wide, a segmented-style pill in narrow/phone */
          #graphsTab { display: none !important; }
          html[data-app-mode="narrow"] #graphsTab,
          html[data-app-mode="phone"]  #graphsTab { display: inline-flex !important;
            align-items: center; margin-left: 6px; cursor: pointer; }
          html[data-graphs-open="1"] #graphsTab { color: #4c8dff; }
          /* the overlay's close × (body-level), only while the charts overlay is showing */
          #graphsClose { display: none; position: fixed; top: 4.2rem; right: 0.6rem; z-index: 121;
            font-size: 1.7rem; line-height: 1; cursor: pointer; color: __TX__;
            background: __PANEL_BG__; border-radius: 8px; padding: 0 0.55rem; }
          html[data-app-mode="narrow"][data-graphs-open="1"] #graphsClose,
          html[data-app-mode="phone"][data-graphs-open="1"]  #graphsClose { display: block; }
          #graphsClose:hover { color: #4c8dff; }
          /* (D-054) overlay charts: stlite 1.57 charts re-fit to the full overlay width via the
             shim's Plotly.Plots.resize; a build whose chart width is server-computed (1.59 dev)
             ignores that, so any narrower-than-overlay plot at least sits CENTERED, not pinned
             left in empty panel background. */
          html[data-graphs-open="1"] .st-key-chartscol .js-plotly-plot .svg-container
            { margin-left: auto !important; margin-right: auto !important; }

          /* narrow/phone: suppress the wide charts drag controls; the middle, its spacer column
             and the top strip take the full width (the fixed column is out of flow) */
          html[data-app-mode="narrow"] #chartsDivider, html[data-app-mode="phone"] #chartsDivider,
          html[data-app-mode="narrow"] #chartsStrip,  html[data-app-mode="phone"] #chartsStrip,
          html[data-app-mode="narrow"] #chartsClose,  html[data-app-mode="phone"] #chartsClose
            { display: none !important; }
          html[data-app-mode="narrow"] .st-key-chartscol,
          html[data-app-mode="phone"]  .st-key-chartscol { display: none; }
          html[data-app-mode="narrow"] [data-testid="stColumn"]:has(.st-key-chartscol),
          html[data-app-mode="phone"]  [data-testid="stColumn"]:has(.st-key-chartscol)
            { flex: 0 0 0 !important; min-width: 0 !important; }
          html[data-app-mode="narrow"] [data-testid="stLayoutWrapper"]:has(> .st-key-topbar),
          html[data-app-mode="phone"]  [data-testid="stLayoutWrapper"]:has(> .st-key-topbar),
          html[data-app-mode="narrow"] [data-testid="stLayoutWrapper"]:has(> .st-key-apptitle),
          html[data-app-mode="phone"]  [data-testid="stLayoutWrapper"]:has(> .st-key-apptitle)
            { width: 100% !important; }
          /* page heading: compact in phone; footer spans the full width when the charts column
             is out of flow (narrow) or the sidebar is a drawer (phone) */
          html[data-app-mode="phone"] .st-key-apptitle .apptitle-main { font-size: 1.3rem; }
          html[data-app-mode="phone"] .st-key-apptitle .apptitle-sub { font-size: 0.85rem; }
          html[data-app-mode="narrow"] header[data-testid="stHeader"],
          html[data-app-mode="phone"]  header[data-testid="stHeader"] { right: 0 !important; }

          /* charts OVERLAY (Graphs tab active): the same fixed column, spanning from the sidebar's
             right edge to the window edge, below the top bar, above the middle content. */
          html[data-app-mode="narrow"][data-graphs-open="1"] .st-key-chartscol {
            display: block; left: var(--sb-w); right: 0; width: auto; top: 6.5rem;
            height: calc(100vh - 6.5rem); z-index: 120; padding-top: 1rem; }

          /* ---- PHONE: single column. The parameter sidebar becomes an off-canvas drawer slid in
             by the injected ☰ button; the charts overlay (Graphs tab) spans the full width. ---- */
          #sbDrawerBtn { display: none; position: fixed; top: 0.5rem; left: 0.5rem; z-index: 135;
            font-size: 1.5rem; line-height: 1; cursor: pointer; color: __TX__;
            background: __PANEL_BG__; border-radius: 8px; padding: 0.1rem 0.55rem; }
          html[data-app-mode="phone"] #sbDrawerBtn { display: block; }
          html[data-app-mode="phone"] section[data-testid="stSidebar"] {
            position: fixed !important; top: 0; left: 0; height: 100vh !important; z-index: 130;
            width: min(310px, 86vw) !important; min-width: 0 !important; max-width: 86vw !important;
            transform: translateX(-102%); transition: transform 0.2s;
            background: __PANEL_BG__; overflow: hidden; }
          /* the inner content div is imposed to --sb-w by D-050; in the drawer it must match the
             (narrower) drawer width so no row spills past the panel edge */
          html[data-app-mode="phone"] section[data-testid="stSidebar"] > div:first-child
            { width: 100% !important; }
          html[data-app-mode="phone"][data-sb-drawer="1"] section[data-testid="stSidebar"] {
            transform: none; box-shadow: 8px 0 30px rgba(0,0,0,0.5); }
          /* tap-outside scrim (also lets a tap close the drawer) */
          #sbScrim { display: none; position: fixed; inset: 0; z-index: 129;
            background: rgba(0,0,0,0.45); }
          html[data-app-mode="phone"][data-sb-drawer="1"] #sbScrim { display: block; }
          html[data-app-mode="phone"] [data-testid="stMainBlockContainer"]
            { padding-left: 1rem !important; padding-top: 3rem !important; }
          /* the D-050 sidebar drag/collapse controls are replaced by the ☰ drawer on phone */
          html[data-app-mode="phone"] #sbDivider,
          html[data-app-mode="phone"] #sbClose,
          html[data-app-mode="phone"] #sbStrip { display: none !important; }
          html[data-app-mode="phone"][data-graphs-open="1"] .st-key-chartscol {
            display: block; left: 0; right: 0; width: auto; top: 6.5rem;
            height: calc(100vh - 6.5rem); z-index: 120; padding-top: 1rem; }

          /* the polite portrait 'rotate to landscape' suggestion (body-level, gated by data-rotate) */
          #rotateOverlay { display: none; position: fixed; inset: 0; z-index: 200;
            background: __TOPBAR_BG__; flex-direction: column; align-items: center;
            justify-content: center; text-align: center; gap: 1rem; padding: 2rem; }
          html[data-rotate="1"] #rotateOverlay { display: flex; }
          #rotateOverlay .rot-ic { font-size: 2.6rem; }
          #rotateOverlay .rot-h { font-size: 1.05rem; font-weight: 600; color: __TX__; }
          #rotateOverlay .rot-p { font-size: 0.9rem; color: __TX__; opacity: 0.7; max-width: 26ch; }
          #rotateOverlay a { color: #4c8dff; cursor: pointer; text-decoration: underline;
            font-size: 0.9rem; }
        </style>
        """
    return (css.replace("__TOPBAR_BG__", topbar_bg)
               .replace("__PANEL_BG__", panel_bg)
               .replace("__TX__", text_col)
               .replace("__PANEL_W__", f"{PANEL_W_PX}px")
               .replace("__GUTTER__", f"{PANEL_GUTTER_PX}px")
               .replace("__STRIP__", f"{PANEL_STRIP_PX}px"))


def inject_layout_css():
    """Phase-2 layout (D-043, variant A2 + D-050 drag-collapse panels): the sticky top bar,
    the fixed-width docked calibration panel and folded-pane strip columns (Streamlit columns
    are proportional, so the fixed widths are imposed on the column that CONTAINS the keyed
    container), the compact sidebar row rhythm, and the D-050 injected panel controls."""
    try:
        _light = getattr(st.context.theme, "type", None) == "light"
    except Exception:
        _light = False
    st.markdown(_layout_css(_light), unsafe_allow_html=True)


def inject_cal_emphasis_css(row_keys):
    """Dim every sidebar row (~40%) except the emphasized one(s), which get a tint + outline.
    Used by the calibration panel (one row, D-043) and by the equations click-to-highlight
    (a subsection's whole parameter set, D-048). Accepts one row key or an iterable."""
    if isinstance(row_keys, str):
        row_keys = [row_keys]
    sel = ", ".join(f'section[data-testid="stSidebar"] .st-key-{rk}' for rk in row_keys)
    st.markdown(
        f"""
        <style>
          section[data-testid="stSidebar"] [class*="st-key-row_"]
            {{ opacity: 0.4; transition: opacity 0.15s; }}
          {sel}
            {{ opacity: 1 !important; background: rgba(76,141,255,0.10);
               outline: 2px solid rgba(76,141,255,0.40); }}
        </style>
        """,
        unsafe_allow_html=True,
    )
