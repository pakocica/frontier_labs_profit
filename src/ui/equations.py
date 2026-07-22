"""The equations-&-calibration panel (one collapsible subsection per model block, with
the changed-at-this-level tag riding in the expander label for targeted remounts) and the
base-model profitability race (levels 1–4).
"""
import streamlit as st

from .calibration import _cal_cards, _cal_delta_merged
from .state import _gc_sym, _reg, toggle_eq_hl


def _eqcols():
    try:
        return st.columns([2, 1], vertical_alignment="top")
    except TypeError:
        return st.columns([2, 1])


def _profit_condition(level, dd):
    """The base-model profitability race (levels 1–4, where value is exponential and the gap is
    constant): revenue ×/yr vs cost ×/yr, both expressed through targets the user already
    controls, with a live verdict and the break-even value-multiplier pivot. Hidden from L5 —
    saturation invalidates the asymptotic reading (same hide-don't-annotate ruling as the
    horizon metrics)."""
    # every number below recomputes each rerun from the EFFECTIVE dict the simulation uses
    # (post-inversion, post-pin) — no default fallbacks.
    gc = dd["g_C0"]
    ga = dd["g_a"]
    nu_ = dd["nu"]
    gp = dd["g_p"]
    speed = gc + ga
    vm = float(10.0**nu_)                   # value multiplier per OOM (the Basics target)
    rev_ooms = nu_ * speed                  # revenue growth in value-OOMs per year
    cost_ooms = gc - gp                     # cost growth in OOMs per year
    rev_x = float(10.0**rev_ooms)           # revenue growth, ×/yr — vm^(g_c+g_a)
    cost_x = float(10.0**cost_ooms)         # cost growth, ×/yr — the bill-growth observable
    be_vm = float(cost_x ** (1.0 / speed))  # break-even value multiplier per OOM
    sym = _gc_sym()
    rel = ">" if rev_x > cost_x else "<"    # borderline equality reads as the red case
    with st.container(border=True):
        st.markdown("**Will the leader *ever* be profitable?** With the gap constant and value "
                    "exponential, one inequality decides it — a race between two growth rates "
                    "you already control:")
        # two stacked lines, not one long row: the single-line form clipped its right operand
        # at the card edge in the narrow tabbed pane (QA S4; .katex-display also scrolls now)
        st.latex(
            rf"\begin{{aligned}}"
            rf"&\small\underbrace{{\;10^{{\,\nu\,({sym}+g_a)}}\;}}"
            rf"_{{\text{{revenue}}\;\times{rev_x:.2f}\text{{/yr}}}}"
            rf"\;\;{rel}\;\;"
            rf"\underbrace{{\;10^{{\,{sym}-g_p}}\;}}"
            rf"_{{\text{{cost}}\;\times{cost_x:.2f}\text{{/yr}}}}"
            rf"\\[10pt]"
            rf"&\small\Longleftrightarrow\quad"
            rf"\nu\,({sym}+g_a)\;{rel}\;{sym} - g_p"
            rf"\end{{aligned}}"
        )
        st.markdown(
            f"Each year the leader gains **{speed:.2f} OOM** of capability, each worth "
            f"**×{vm:.2f}** in value, so revenue grows **×{rev_x:.2f}/yr** (the gap is constant, "
            f"so the rent $\\theta[W(x^L)-W(x^F)]$ inherits $W$'s growth rate). The training "
            f"bill grows **×{cost_x:.2f}/yr**. In OOM terms: revenue climbs "
            f"**+{rev_ooms:.2f} OOM/yr** vs the bill's **+{cost_ooms:.2f} OOM/yr**.")
        if rev_x > cost_x:
            st.markdown(f":green[**✓ ×{rev_x:.2f} > ×{cost_x:.2f} — revenue outruns cost ⇒ "
                        "profitable sooner or later, and it stays profitable.**]")
        else:
            st.markdown(f":red[**✗ ×{rev_x:.2f} < ×{cost_x:.2f} — cost outruns revenue ⇒ never "
                        "profitable at these settings.**]")
        st.caption(f"Break-even pivot: a value multiplier of **×{be_vm:.2f}/OOM** (now "
                   f"×{vm:.2f}) balances the race — slide *Value per OOM* past it "
                   "and the verdict flips.")
        if level >= 3:
            st.caption("Computed at *today's* rates: from Level 3 the ψ feedback accelerates "
                       "the revenue side over time, and the Level-4 slowdown pulls both sides "
                       "toward their long-run floors.")


# Per-subsection equation panel. Each level changes exactly ONE subsection (or adds Extensions at
# L8); the change-tag rides in the EXPANDER LABEL so Streamlit remounts just that subsection at the
# level it changes (applying the expanded= default), while every other subsection keeps the user's
# manual open/closed state across level switches (stable label -> stable widget identity).
_SUB_ORDER = ["leader_compute", "leader_algo", "follower", "value", "revenue", "cost",
              "profit", "extensions"]
_SUB_LABEL = {"leader_compute": "Leader compute  ċᴸ",
              "leader_algo": "Leader algorithmic progress  ȧᴸ",
              "follower": "Follower", "value": "Value  W(x)",
              "revenue": "Revenue & release", "cost": "Cost", "profit": "Profit",
              "extensions": "Extensions"}
# level -> the single subsection that is changed / new at that level (L1: nothing; cost is tagged
# at BOTH 2 (ℓ unpins — training in advance) and 8 (the R&D overhead φ_RD)).
_CHANGED_AT = {2: "cost", 3: "leader_algo", 4: "leader_compute", 5: "value", 6: "follower",
               7: "revenue", 8: "cost", 9: "extensions"}


# ---- D-048: the ONE authoritative subsection → parameters map ------------------------------
# (param_key, pinned) pairs whose calibration cards ride in each subsection at each level.
# The render code below consumes it AND the sidebar's equation-driven filter reads it — keep
# it as explicit data here, never as scattered conditionals.
def subsection_param_entries(sub_id, level):
    if sub_id == "leader_compute":
        cards = [("g_C0", False)]
        if level >= 4:
            cards += [("g_C_inf", False), ("xi", False)]
        return cards
    if sub_id == "leader_algo":
        cards = [("g_a", level < 3)]
        if level >= 3:
            cards += [("rho0", False), ("gamma", False)]
        if level >= 9:
            cards += [("alpha", False), ("eta", False)]
        return cards
    if sub_id == "follower":
        if level <= 5:
            return [("delta_total", False)]
        return [("delta_dev", False), ("delta_rel", False), ("Delta0", False),
                ("split", False), ("g_a_F", False), ("g_CF0", False), ("g_CF_inf", False),
                ("xi_F", False)]
    if sub_id == "value":
        cards = [("nu", False)]
        if level >= 5:
            cards.append(("x_mid", False))
        cards.append(("W0", level < 9))
        return cards
    if sub_id == "revenue":
        cards = [("theta", False)]
        if level >= 7:
            cards.append(("tau", False))
        return cards
    if sub_id == "cost":
        cards = []
        if level >= 8:
            cards.append(("phi_RD", False))
        cards.append(("S0", False))
        if level >= 2:
            cards.append(("ell", False))
        cards.append(("g_p", level < 8))
        return cards
    return []   # profit, extensions: no parameter cards


def eq_show_all(level):
    """Effective show-all state: at L1 everything is new, so ALL subsections show and the
    toggle is hidden (D-048); above L1 the 'show all equations' checkbox decides."""
    return True if level == 1 else bool(st.session_state.get("w_eq_all", False))


def visible_subsections(level):
    """The subsections the Equations tab currently renders (concise → only the changed one)."""
    changed = _CHANGED_AT.get(level)
    existing = [s for s in _SUB_ORDER if not (s == "extensions" and level < 9)]
    if not eq_show_all(level) and changed is not None:
        return [changed]
    return existing


def sidebar_filter_keys(level):
    """The parameter keys whose sidebar rows should show under the equation-driven filter
    (D-048): the parameters of the currently visible subsections. None = show ALL (the
    middle tab is Introduction — no equations on screen to filter by)."""
    if st.session_state.get("_pane_tab_mem", "Introduction") != "Equations":
        return None
    keys = set()
    for sub_id in visible_subsections(level):
        keys |= {k for k, _ in subsection_param_entries(sub_id, level)}
    return keys


def equations_panel(level, dd, p):
    """Equations + merged calibration, one collapsible subsection at a time. Left [2/3] = glosses +
    st.latex (with new/changed tags); right [1/3] = calibration cards for that subsection's params
    (dial/range, or pinned-shown, at ≤ level). View modes control which subsections show / open."""
    st.markdown("**Equations & calibration at this level**")
    st.caption("Grades: **A** solid data anchor · **B** reasonable anchor · **C** judgment / weakly "
               "identified · **F** free choice or decision variable. Each right-hand card shows "
               "value, MC range and calibration target; **details** has the full note and any "
               "alternatives. Defaults are provisional → calibration session.")
    if level == 1:
        st.caption("Pinned-but-shown (no dial yet): $\\ell$ (Level 2), $g_a$ (Level 3), and "
                   "$g_p$ (pinned by the bill-growth anchor). The Level-1 controls are "
                   "observables — compute scaling (⇒ $g_c$), "
                   "follower lag (⇒ $\\Delta_0$, $\\delta$), value multiplier (⇒ $\\nu$) — plus "
                   "$\\theta$ and $S_0$. Capability $x = a + c$, OOM above the "
                   "2026 frontier.")
    # CONCISE by default (Pavel's ruling): only the subsection changed at this level shows;
    # one checkbox expands to the full model. At L1 EVERYTHING is new, so all subsections
    # show and the checkbox is hidden (D-048). The old display-mode tabs' stale session key
    # is dropped so nothing rebinds to it.
    st.session_state.pop("eq_view", None)
    if level >= 2:
        st.checkbox("show all equations", key=_reg("w_eq_all", False),
                    help="Unticked: only the subsection that is **new or changed at this "
                         "level** is shown (and the left panel narrows to its parameters). "
                         "Ticked: every subsection of the model so far, expanded.")

    def eq(gloss, latex, tag=None):
        if tag == "new":
            gloss += "  :green[● new at this level]"
        elif tag == "changed":
            gloss += "  :orange[● changed at this level]"
        st.caption(gloss)
        st.latex(latex)

    def render(sub_id):
        left, right = _eqcols()
        if sub_id == "leader_compute":
            with left:
                if level < 4:
                    # constant growth: plain g_c — the g_c0/g_c∞ split is part of what L4 teaches
                    eq("Leader compute grows at the constant rate $g_c$"
                       + ("." if level == 1 else " (the slowdown arrives at Level 4)."),
                       r"\dot c^L = g_c")
                else:
                    eq("Compute growth decays from **today's** rate $g_{c0}$ (the constant $g_c$ "
                       "of the levels below) toward a long-run floor $g_{c\\infty}$.",
                       r"\dot c^L = g_c(t) = g_{c\infty} + (g_{c0}-g_{c\infty})e^{-\xi t}",
                       tag="changed" if level == 4 else None)
            _cal_cards(right, subsection_param_entries("leader_compute", level), dd, p)
        elif sub_id == "leader_algo":
            with left:
                if level <= 2:
                    eq("Algorithmic progress at the constant rate $g_a$.", r"\dot a^L = g_a")
                elif level >= 9:
                    eq("Base rate $g_a$ times a CES mix of the AI-assistance feedback $\\psi$ and "
                       "experiment compute (curvature $\\eta$).",
                       r"\dot a^L = g_a\left[(1-\alpha)\Big(\tfrac{\psi(x^L)}{\psi(0)}\Big)^{\eta} "
                       r"+ \alpha\Big(\tfrac{g_c(t)}{g_{c0}}\Big)^{\eta}\right]^{1/\eta},"
                       r"\quad \psi(x)=1+\rho_0 e^{\gamma x}")
                elif level == 3:
                    # compute growth is still constant at L3, so the experiment-compute ratio is
                    # identically 1 — show the collapsed form (and plain g_c convention holds)
                    eq("The base model's constant $g_a$ now responds to research inputs: a "
                       "weighted average ($\\eta=1$) of the AI-assistance feedback $\\psi$ and "
                       "experiment compute — whose ratio is 1 while compute growth is constant "
                       "(it starts moving at Level 4).",
                       r"\dot a^L = g_a\left[(1-\alpha)\tfrac{\psi(x^L)}{\psi(0)} "
                       r"+ \alpha\right],\quad \psi(x)=1+\rho_0 e^{\gamma x}",
                       tag="changed")
                else:
                    eq("The base model's constant $g_a$ now responds to research inputs: a weighted "
                       "average ($\\eta=1$) of the AI-assistance feedback $\\psi$ and experiment "
                       "compute.",
                       r"\dot a^L = g_a\left[(1-\alpha)\tfrac{\psi(x^L)}{\psi(0)} "
                       r"+ \alpha\tfrac{g_c(t)}{g_{c0}}\right],\quad \psi(x)=1+\rho_0 e^{\gamma x}")
                if level >= 3:
                    eq("ψ-share: the fraction of algo progress from the $\\psi$ feedback (past ~25% "
                       "it is no longer a small correction).",
                       r"\psi\text{-share} = 1 - \dot a^L\big|_{\psi\ \text{frozen}} \big/ \dot a^L",
                       tag="new" if level == 3 else None)
            _cal_cards(right, subsection_param_entries("leader_algo", level), dd, p)
        elif sub_id == "follower":
            with left:
                if level <= 5:
                    eq("The follower has no engine of its own — pure catch-up at the single rate "
                       "$\\delta$, where $\\Delta = x^L - x^F$ is the **capability gap** (the "
                       "follower's own compute and algorithmic progress arrive at Level 6).",
                       r"\dot x^F = \delta\,(x^L - x^F) = \delta\,\Delta")
                else:
                    srv = "x^R" if level >= 7 else "x^L"
                    eq("The follower gets its own algo rate $g_a^F$, and $\\delta$ unpacks into two "
                       "channels: ambient diffusion $\\delta_{dev}$ on the algo gap, plus "
                       "distillation $\\delta_{rel}$ from the served model (off where the follower "
                       "is already ahead); the **capability gap** is $\\Delta = x^L - x^F$."
                       + ("" if level >= 7 else " Here the leader serves its newest model."),
                       rf"\dot a^F = g_a^F + \delta_{{dev}}(a^L - a^F) + \delta_{{rel}}\max({srv} - x^F,\,0)",
                       tag="changed" if level == 6 else None)
                    eq("Follower compute growth — its own decaying family (introduced here).",
                       r"\dot c^F = g_c^F(t) = g_{c\infty}^F + (g_{c0}^F-g_{c\infty}^F)e^{-\xi^F t}",
                       tag="new" if level == 6 else None)
                    eq("The initial gap $\\Delta_0$ splits between algo and compute.",
                       r"a^F(0) = -\text{split}\cdot\Delta_0, \qquad c^F(0) = -(1-\text{split})\Delta_0",
                       tag="new" if level == 6 else None)
            if level <= 5:
                _cal_delta_merged(right, dd, p)
            else:
                _cal_cards(right, subsection_param_entries("follower", level), dd, p)
        elif sub_id == "value":
            with left:
                if level <= 4:
                    eq("Dollar value of capability — pure exponential here: each OOM multiplies "
                       "value by $10^{\\nu}$. (The saturation bend $x_{mid}$ arrives at Level 5.)",
                       r"W(x) = W_0\, 10^{\,\nu x}")
                else:
                    eq("Dollar value — logistic: exponential, then saturating past $x_{mid}$; "
                       "anchored so $W(0) = W_0$.",
                       r"W(x) = \frac{W_{\max}}{1 + 10^{-\nu (x - x_{mid})}}, \qquad W(0) = W_0",
                       tag="changed" if level == 5 else None)
            _cal_cards(right, subsection_param_entries("value", level), dd, p)
        elif sub_id == "revenue":
            with left:
                if level >= 7:
                    eq("The leader serves the model it had $\\tau$ ago (time-varying $\\tau(t)$ "
                       "gives the window scenarios); $\\tau = 0$ means the served model equals the "
                       "developed one.",
                       r"x^R_t = x^L_{t-\tau}",
                       tag="new" if level == 7 else None)
                served = "x^R" if level >= 7 else "x^L"
                _on = "the *served* model" if level >= 7 else "the leader's model"
                eq(f"Revenue = margin $\\theta$ times the value gap it can charge for — earned on "
                   f"{_on}.",
                   rf"\text{{revenue}} = \theta\,\big[\,W({served}) - W(x^F)\,\big]",
                   tag="changed" if level == 7 else None)
            _cal_cards(right, subsection_param_entries("revenue", level), dd, p)
        elif sub_id == "cost":
            with left:
                if level >= 8:
                    eq("The training bill = compute path marked up by the R&D overhead "
                       "$\\phi_{RD}$: pay now for the model shipping $\\ell$ ahead, at prices "
                       "falling at $g_p$.",
                       r"\text{cost}(t) = (1+\phi_{RD})\,S_0\,10^{\,c^L(t+\ell) - c^L(0)}\,10^{-g_p t}",
                       tag="changed" if level == 8 else None)
                elif level >= 2:
                    eq("Training in advance: the firm pays now for the compute of the model "
                       "shipping $\\ell$ ahead, at prices falling at $g_p$ — today's bill jumps to "
                       f"$10^{{\\,{_gc_sym()}\\ell}}\\times$ the current model's "
                       f"(≈ {10.0 ** (dd['g_C0'] * dd['ell']):.2f}× at $\\ell = $ "
                       f"{dd['ell']:.2f}).",
                       r"\text{cost}(t) = S_0\,10^{\,c^L(t+\ell) - c^L(0)}\,10^{-g_p t}",
                       tag="changed" if level == 2 else None)
                else:
                    eq("The training bill: the compute of the model the leader is **running right "
                       "now** ($\\ell = 0$), at prices falling at $g_p$ — so cost today is exactly "
                       "$S_0$ and net cost growth ≈ 2.4×/yr at the constant-compute defaults.",
                       r"\text{cost}(t) = S_0\,10^{\,c^L(t) - c^L(0)}\,10^{-g_p t}")
            _cal_cards(right, subsection_param_entries("cost", level), dd, p)
            if level == 1:
                right.markdown("$\\ell$ **=** 0 · *(pinned — training in advance arrives at "
                               "Level 2)*")
        elif sub_id == "profit":
            with left:
                eq("Profit — undiscounted yearly flows (no NPV).",
                   r"\Pi = \text{revenue} - \text{cost}")
                if level <= 4:
                    _profit_condition(level, dd)
            right.caption("(no free parameter)")
        elif sub_id == "extensions":
            active = []
            if dd.get("theta_gap"):
                active.append(("II.4 gap-dependent margin (on): $\\theta$ rises with the gap.",
                               r"\theta(\Delta) = \bar\theta\,"
                               r"\frac{1 - e^{-\Delta/\Delta_\theta}}{1 - e^{-\Delta_0/\Delta_\theta}}"))
            if dd.get("chi", 0.0):
                active.append(("II.5 distillation defenses (on): throttle $\\delta_{rel}$ at a "
                               "revenue cost.",
                               r"\delta_{rel} \to (1-\chi)\,\delta_{rel}, \qquad "
                               r"\text{revenue} \to (1 - 0.4\,\chi)\,\text{revenue}"))
            if dd.get("own_mode"):
                active.append(("II.6 ownership user-cost (on): a depreciation load on the cost flow.",
                               r"\text{cost}(t) \to \text{cost}(t)\,\frac{r + \delta_K + g_p}{r + g_p}"))
            if dd.get("phi_mix", 0.0):
                active.append(("II.2 public-knowledge mix (on): the diffusion channel sees a blend "
                               "of leader and follower.",
                               r"\tilde a^L = (1-\phi_{mix})\,a^L + \phi_{mix}\,a^F"))
            if dd.get("labor_line"):
                active.append(("II.7 decoupled labor line (on): a growing wage bill subtracted from "
                               "profit.", r"\Pi \to \Pi - L_0\,e^{g_w t}"))
            if active:
                for g, l in active:
                    eq(g, l, tag="new")
            else:
                st.caption("All extension dials (II.2, II.4–II.7) are off — turn one on in the "
                           "sidebar and its equation appears here.")

    changed_here = _CHANGED_AT.get(level)
    existing = [s for s in _SUB_ORDER if not (s == "extensions" and level < 9)]
    shown = visible_subsections(level)
    if len(shown) < len(existing):
        st.caption(f"**{len(existing) - len(shown)} unchanged subsections hidden** — they "
                   "carry over from the levels below (tick **show all equations** for the "
                   "full model).")
    # D-048: the changed subsection is marked VISUALLY (accent left border + subtle header
    # tint, both themes) instead of the old "· ● changed at this level" label text
    if changed_here in shown:
        st.markdown(
            f"<style>.st-key-eqsub_{changed_here} [data-testid='stExpander'] summary "
            "{ border-left: 3px solid #4c8dff; background: rgba(76,141,255,0.10); }"
            "</style>", unsafe_allow_html=True)
    for sub_id in shown:
        with st.container(key=f"eqsub_{sub_id}"):
            with st.expander(_SUB_LABEL[sub_id], expanded=True):
                # click-to-highlight (D-048): the ⌖ in the subsection's header area lights up
                # this subsection's parameters in the left panel; clicking again clears
                if subsection_param_entries(sub_id, level):
                    on = st.session_state.get("_eq_hl") == sub_id
                    with st.container(key=f"eqhl_{sub_id}"):
                        st.button("⌖ ✓" if on else "⌖", key=f"hl_{sub_id}",
                                  on_click=toggle_eq_hl, args=(sub_id,),
                                  help="highlight this subsection's parameters in the left "
                                       "panel" + ("; click again to clear" if on else ""))
                render(sub_id)
