"""Theme: the two-hue semantic palette (D-040), CSS injections, and plotly helpers.
"""
import plotly.graph_objects as go
import streamlit as st

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
    restyle it to follow the app's light/dark theme; trace colors come from the PAL palette."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0, y=1, yanchor="top", yref="container", pad=dict(t=4),
                   font=dict(size=13)),
        height=height, margin=dict(l=44, r=10, t=48, b=32),
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
    used by the live Monte-Carlo view."""
    st.plotly_chart(fig, use_container_width=True, theme="streamlit", key=key)


def tab_intro(what, how):
    """A consistent two-line header at the top of every tab: what it shows + how to read it."""
    st.caption(f"**What this shows.**  {what}")
    st.caption(f"**How to read it.**  {how}")



# ======================================================================= D-043 layout CSS
def inject_layout_css():
    """Phase-2 layout (D-043, variant A2): the sticky top bar, the fixed-width docked
    calibration panel and folded-pane strip columns (Streamlit columns are proportional, so
    the fixed widths are imposed on the column that CONTAINS the keyed container), and the
    compact sidebar row rhythm."""
    # the sticky bar needs an OPAQUE theme-matched background (content scrolls under it) and
    # the right chart panel the SIDEBAR's background; `--background-color` is not a defined
    # CSS var in Streamlit 1.59, so resolve the app theme server-side (same best-effort read
    # the MC component uses). Streamlit default secondary backgrounds: light #f0f2f6,
    # dark #262730.
    try:
        _light = getattr(st.context.theme, "type", None) == "light"
    except Exception:
        _light = False
    topbar_bg = "#ffffff" if _light else "#0e1117"
    panel_bg = "#f0f2f6" if _light else "#262730"
    # D-048: when the right panel is open, the header and the top strip stop at its edge
    charts_on = bool(st.session_state.get("_charts_open", True))
    header_right = "16rem" if charts_on else "0px"
    topbar_mr = "15.5rem" if charts_on else "0.5rem"
    css = """
        <style>
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
             | right chart panel (fixed, collapsible). Rows must NOT wrap — a percentage or
             fixed flex-basis ignores its siblings' widths and would push columns onto a
             second row (verified in-browser). */
          [data-testid="stHorizontalBlock"]:has(.st-key-mainpane),
          [data-testid="stHorizontalBlock"]:has(.st-key-calpanel)
            { flex-wrap: nowrap !important; }
          /* D-048: the right chart panel MIRRORS the sidebar — full height from the very top
             of the window, the sidebar's background, the sidebar's default width (16rem).
             The fixed panel sits out of flow; the spacer column below reserves its footprint
             inside the block, so the middle pane ends a gutter before it. Collapsed, only a
             floating « control remains (mirror of the sidebar's ») and the freed width flows
             to the flexible middle pane. */
          .st-key-chartscol {
            position: fixed; top: 0; right: 0; width: 16rem;
            height: 100vh; overflow-y: auto; overflow-x: hidden; z-index: 99;
            background-color: __PANEL_BG__;
            border-left: 1px solid rgba(128,128,128,0.2);
            padding: 0.4rem 0.7rem 1.2rem; }
          [data-testid="stColumn"]:has(.st-key-chartscol) {
            flex: 0 0 15.5rem !important; min-width: 15.5rem !important; }
          [data-testid="stColumn"]:has(.st-key-chartsstrip) {
            flex: 0 0 0px !important; min-width: 0px !important; }
          .st-key-chartsstrip {
            position: fixed; top: 0.55rem; right: 3.4rem; z-index: 999995; }
          .st-key-chartsstrip button
            { border: none; background: transparent; padding: 0.15rem 0.4rem;
              font-size: 1.05rem; line-height: 1.2; opacity: 0.65; }
          .st-key-chartsstrip button:hover { opacity: 1; color: #4c8dff; }
          /* the Streamlit header stops where the right panel begins (it already starts after
             the sidebar) — the ⋮ menu stays reachable left of the panel */
          header[data-testid="stHeader"] { right: __HEADER_RIGHT__ !important;
            width: auto !important; }
          /* the top strip belongs to the MIDDLE area only (width calc, not margin — the
             wrapper is width:100%, so a margin would overflow instead of shrinking it) */
          [data-testid="stLayoutWrapper"]:has(> .st-key-topbar)
            { width: calc(100% - __TOPBAR_MR__) !important; }
          /* top strip: level selector left, mode switch flush right */
          .st-key-topbar [data-testid="stSegmentedControl"]
            { display: flex; justify-content: flex-end; }
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
            [data-testid="stColumn"]:has(.st-key-chartscol),
          [data-testid="stHorizontalBlock"]:has(.st-key-calpanel)
            [data-testid="stColumn"]:has(.st-key-chartsstrip)
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
          /* the chart panel's collapse chevron: top-LEFT of the panel (mirror image of the
             sidebar's « at its top-right) */
          .st-key-chartshide { display: flex; justify-content: flex-start; }
          .st-key-chartshide button
            { border: none; background: transparent; padding: 0 4px; min-height: 1.2rem;
              line-height: 1.2; font-size: 0.9rem; opacity: 0.6; }
          .st-key-chartshide button:hover { opacity: 1; color: #4c8dff; }

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
        </style>
        """
    st.markdown(css.replace("__TOPBAR_BG__", topbar_bg)
                   .replace("__PANEL_BG__", panel_bg)
                   .replace("__HEADER_RIGHT__", header_right)
                   .replace("__TOPBAR_MR__", topbar_mr), unsafe_allow_html=True)


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
