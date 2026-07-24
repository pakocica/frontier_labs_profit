"""Content & parameter metadata — the ONE app-side home for prose and labels.

Per-parameter metadata lives in a single registry `PARAMS: dict[str, ParamMeta]` (D-061); the
historical dict names — interpretation texts (INTERP), the three label maps (_MATH_LABEL /
_SHORT_NAME / _UNI_LABEL: LaTeX / plain words / unicode), calibration captions (_CAL_TARGET /
_CAL_ALT) and grades (GRADES) — are thin DERIVED VIEWS over it, so consumers are unchanged.
Target-specific content (INTERP_T, TSPEC), level cards and notation stay as their own maps.
Envelope+tight ranges, CAL_SOURCES and inversions stay in the NOTEBOOK (single source of truth);
this module only holds app-side presentation.
"""
from dataclasses import dataclass

import numpy as np
import streamlit as st

from .model_access import m, P0


# ======================================================================= parameter registry
# ONE registry keyed by parameter — the single home for a parameter's label/prose metadata,
# replacing seven parallel param-keyed dicts that used to drift out of sync (D-061). Every field
# is OPTIONAL; a parameter carries only the metadata it has. The seven historical dict names live
# below as thin DERIVED VIEWS over PARAMS, so every consumer (sidebar / calpanel / equations /
# calibration / views) is unchanged.
#
# Key PRESENCE is load-bearing: the `.get(k, k)` / `.get(k, '—')` fallbacks at the call sites
# encode behaviour (e.g. `tau` is deliberately absent from _MATH_LABEL so it renders the literal
# "tau"). A derived view therefore omits EXACTLY the keys whose field is None — never "helpfully"
# fill a missing field. (Envelope/tight ranges, CAL_SOURCES and inversions stay in the NOTEBOOK,
# the single source of truth; this module only holds app-side presentation.)
#
# Field meanings (each was one dict before D-061):
#   interp     — INTERP: concise per-parameter interpretation (units, plain-language meaning,
#                reference anchor). Rendered in markdown / captions / popovers AND as the slider
#                help= tooltip. Math in inline $...$; literal dollars escaped \$. Grounding:
#                Notes/calibration_master.md.
#   math_label — _MATH_LABEL: LaTeX symbol (no $ delimiters).
#   uni_label  — _UNI_LABEL: unicode symbol for contexts that can't render LaTeX (st.dataframe).
#   short_name — _SHORT_NAME: plain-word name shown beside the symbol wherever raw code names would
#                otherwise leak into the UI. Code names appear only in "Under the hood".
#   cal_target — _CAL_TARGET: one-line observable FACT the parameter is calibrated to, with its
#                number (Pavel's ruling), merged into the equations panel's right-hand cards.
#   cal_alt    — _CAL_ALT: alternative calibration / documented tension, surfaced in the details
#                popover.
#   grade      — GRADES: grounding grade (A solid data anchor · B reasonable · C judgment / weakly
#                identified · F free choice or decision variable). From Notes/calibration_master.md.
@dataclass
class ParamMeta:
    interp: str | None = None
    math_label: str | None = None
    uni_label: str | None = None
    short_name: str | None = None
    cal_target: str | None = None
    cal_alt: str | None = None
    grade: str | None = None


PARAMS: dict[str, ParamMeta] = {
    'gamma': ParamMeta(interp='**$\\gamma$ — $\\psi$ compounding, per OOM.** Strength of the recursive-self-improvement (RSI) feedback: each OOM of capability multiplies AI-R&D speed by $e^{\\gamma x}$. $\\gamma = 0$ disables it entirely (freeze). $\\gamma \\gtrsim 0.42$ goes super-exponential — finite-time blow-up inside the 10-yr horizon (spec N4). Default 0.2, tentative (grade C).', math_label='\\gamma', uni_label='γ', short_name='RSI compounding', cal_target='how strongly AI-for-AI-R&D compounds (no observable yet)', grade='C'),
    'x_mid': ParamMeta(interp='**$x_{mid}$ — value-curve bend, OOM above the 2026 frontier.** Where $W(x)$ turns from exponential to saturating. Reference values: **2** = value saturates early (commoditization); **5** = mid; **10** = harvest continues across the horizon. Pivotal knob (grade C).', math_label='x_{mid}', uni_label='x_mid', short_name='value-curve bend', cal_target="value stops compounding ~10 OOM above today's frontier", cal_alt='**Refs:** 2 = early commoditization · 5 = mid · 10 = harvest continues to the horizon.', grade='C'),
    'theta': ParamMeta(interp="**$\\theta$ — conduct / operating margin ($\\theta = \\rho\\eta$).** Leaders' share of the capability-gap rent they actually extract. Range [0.15, 0.6], triangular mode 0.35 from Cournot triangulation ($n \\approx 3$–5 → $\\rho \\approx 1/(n{+}1)$), softened by differentiation/lock-in. NOT the 60–80% gross margin (Lerner-index trap). Grade C.", math_label='\\theta', uni_label='θ', short_name='operating margin', cal_target='3–5 competing leaders each keep ≈ 1/(n+1) of the rent', cal_alt='**Alt:** NOT the 60–80% gross margin — that is the Lerner-index trap.', grade='C'),
    'delta_dev': ParamMeta(interp='**$\\delta_{dev}$ — developed-model diffusion, per yr.** Release-invariant (ambient) catch-up: the follower closes the *algorithmic* gap via talent, published methods and ambient know-how even with nothing released. Default 0.20; U[0.08, 0.40]/yr. Jointly calibrated with $\\delta_{rel}$ so the observed ~8-month lag is stationary (see $\\delta$); weakly identified (grade C).', math_label='\\delta_{dev}', uni_label='δ_dev', short_name='ambient diffusion', cal_target='keeps the ~8-mo lag constant — the ambient share of the wedge', grade='C'),
    'delta_rel': ParamMeta(interp='**$\\delta_{rel}$ — released-model distillation, per yr.** Release-controlled channel: the follower distills from the model the leader serves ($x^R$, which equals $x^L$ until a release delay is set). This is the lever release delay acts on. Default 0.26; U[0.12, 0.75]/yr; jointly calibrated with $\\delta_{dev}$ for lag stationarity (grade C). **$\\delta_{rel} = 0$ = distillation disabled** — released channel off; the follower catches up through $\\delta_{dev}$ only.', math_label='\\delta_{rel}', uni_label='δ_rel', short_name='distillation', cal_target='keeps the ~8-mo lag constant — the distillation share of the wedge', grade='C'),
    'g_C_inf': ParamMeta(interp='**$g_{c\\infty}$ — compute-growth floor, OOM/yr.** Long-run compute growth once scaling hits limits (power ~2030). 0.13 OOM/yr ≈ 1.35×/yr (hardware-only). The floor *level* is our extrapolation (grade C); widget-critical for the slowdown scenario.', math_label='g_{c\\infty}', uni_label='g_c∞', short_name='compute-growth floor', cal_target='long-run compute grows only ~×1.35/yr (hardware-only)', cal_alt='**Note:** the hardware-only floor is our extrapolation, not measured.', grade='C'),
    'tau': ParamMeta(interp='**$\\tau$ — release / withholding delay, months.** Policy lever: $x^R_t = x^L_{t-\\tau}$, the leader serves the model it had $\\tau$ ago. $\\tau = 0$ = release immediately (baseline). Capped at the 3-month policy-relevant range. Grade F by design (a decision variable, not calibrated).', uni_label='τ', short_name='release delay', cal_target='the policy lever itself — chosen, not calibrated'),
    'g_C0': ParamMeta(interp='**$g_{c0}$ — physical compute growth today, OOM/yr.** 0.623 = log10(4.2), i.e. 4.2×/yr (Epoch; multiple series agree — grade A).', math_label='g_{c0}', uni_label='g_c0', short_name='compute growth today', cal_target='frontier training compute grows ~×4.2/yr', grade='A'),
    'xi': ParamMeta(interp='**$\\xi$ — slowdown decay rate, per yr.** How fast $g_c(t)$ decays from $g_{c0}$ toward the floor: $g_c(t) = g_{c\\infty} + (g_{c0} - g_{c\\infty})e^{-\\xi t}$. Pure scenario knob (grade F) — sweep it.', math_label='\\xi', uni_label='ξ', short_name='compute slowdown speed', cal_target='how fast the compute slowdown bites (scenario dial)', grade='F'),
    'g_a': ParamMeta(interp='**$g_a$ — algorithmic progress today, OOM/yr.** 0.447 = log10(2.8), i.e. 2.8×/yr pretraining-efficiency gains (Ho et al. and Whitfill et al., two independent anchors → grade A).', math_label='g_a', uni_label='g_a', short_name='algo progress today', cal_target='algorithms add ~×2.8/yr of effective compute', grade='A'),
    'alpha': ParamMeta(interp='**$\\alpha$ — experiment-compute weight in the CES research bracket.** Share of algo progress driven by experiment compute vs researchers. 0.5 placeholder (grade F → calibration).', math_label='\\alpha', uni_label='α', short_name='experiment-compute weight', cal_target='experiments-vs-researchers weight in R&D (no observable)', grade='F'),
    'eta': ParamMeta(interp='**$\\eta$ — CES exponent for compute–labor substitution in research.** $\\eta = 1$ weighted avg; $\\eta \\to 0$ Cobb-Douglas; $\\eta < 0$ complements; min = Leontief. From Whitfill & Wu 2025 ($\\sigma = 2.58$ substitutes / ≈ 0 complements — sign flips on controls, grade B).', math_label='\\eta', uni_label='η', short_name='research elasticity', cal_target='how substitutable compute and researchers are in R&D', grade='B'),
    'rho0': ParamMeta(interp='**$\\rho_0$ — AI-R&D speedup today.** $\\psi(0) = 1 + \\rho_0$ is the current AI-assistance multiplier on research throughput. 0.3 tentative (grade C).', math_label='\\rho_0', uni_label='ρ₀', short_name='AI R&D speedup today', cal_target='AI makes AI R&D ~30% faster today', grade='C'),
    'Delta0': ParamMeta(interp='**$\\Delta_0$ — initial capability gap, OOM.** The follower starts $\\Delta_0$ OOM behind. Central 0.7 OOM (~8-month lag: METR agentic, private benches), MC [0.35, 1.4]. Benchmark lags are *lower bounds* (benchmaxxing). Grade A/B.', math_label='\\Delta_0', uni_label='Δ₀', short_name='initial gap', cal_target='the follower is ~8 months behind the frontier today', cal_alt='**Note:** benchmark lags are *lower bounds* (benchmaxxing).', grade='A/B'),
    'split': ParamMeta(interp='**split — algo share of $\\Delta_0$.** How much of the initial gap is algorithmic vs compute. 0.5 placeholder (grade F, open question).', math_label='\\text{split}', uni_label='split', short_name='algo share of the gap', cal_target='about half the initial gap is algorithmic, half compute', grade='F'),
    'g_a_F': ParamMeta(interp='**$g_a^F$ — follower algo progress, OOM/yr.** ≈ 0.7·$g_a$: progress is scale-biased (frontier improves faster than small scale — Gundlach), $\\theta^F/\\theta^L \\approx 0.6$–0.8 → 0.31 OOM/yr. Grade B.', math_label='g_a^F', uni_label='g_a,F', short_name='follower algo progress', cal_target="follower algo progress ≈ 70% of the leader's", grade='B'),
    'g_CF0': ParamMeta(interp='**$g_{c0}^F$ — follower compute growth today, OOM/yr.** Chinese/fringe compute build-out → calibration. 0.5 placeholder (grade C).', math_label='g_{c0}^F', uni_label='g_c0ᶠ', short_name='follower compute growth', cal_target='follower compute grows ~×3.2/yr today'),
    'g_CF_inf': ParamMeta(interp='**$g_{c\\infty}^F$ — follower compute-growth floor, OOM/yr.** Long-run follower compute growth after slowdown. 0.10 placeholder (grade C/F).', math_label='g_{c\\infty}^F', uni_label='g_c∞ᶠ', short_name='follower compute floor', cal_target='follower long-run compute floor ~×1.26/yr'),
    'xi_F': ParamMeta(interp='**$\\xi^F$ — follower slowdown decay, per yr.** Decay of $g_c^F(t)$ toward its floor. Scenario knob (grade F).', math_label='\\xi^F', uni_label='ξ_F', short_name='follower slowdown speed', cal_target="how fast the follower's slowdown bites (scenario dial)"),
    'W0': ParamMeta(interp="**$W_0$ — value at today's frontier, \\$B/yr.** Absolute dollar scale of $W(0)$, reverse-engineered so $\\theta\\,\\Delta W \\approx$ \\$50B matches observed frontier gross profit. NOT independently pinned (grade F, calibration gap #1).", math_label='W_0', uni_label='W₀', short_name='value scale', cal_target='frontier operating profit today ≈ \\$39B/yr', grade='F'),
    'nu': ParamMeta(interp='**$\\nu$ — value curvature, value-OOMs per capability-OOM.** Base-10 slope of log-value in the exponential regime: each capability OOM multiplies value by $10^{\\nu}$. (0, $\\log_{10} 3$] from per-task token blow-up / cost-intensity; 0.239 = 0.55/ln 10 (grade C; log10 units).', math_label='\\nu', uni_label='ν', short_name='value slope', cal_target='each OOM of capability is worth ~×1.73 more', grade='C'),
    'phi_RD': ParamMeta(interp='**$\\phi_{RD}$ — R&D markup on compute.** Cost = $(1 + \\phi_{RD})\\cdot$ train-compute flow: staff + experiments on top of the final run. 1.0 ≈ experiments+staff double the final run (grade C).', math_label='\\phi_{RD}', uni_label='φ_RD', short_name='R&D overhead', cal_target='staff + experiments ≈ double the final-run bill', grade='C'),
    'ell': ParamMeta(interp="**$\\ell$ — training lead time, yr.** How much in advance the next model's compute is bought/trained: the firm pays at $t$ for the compute of the model shipping at $t+\\ell$, so today's bill is $10^{\\,g_{c0}\\ell}\\times$ the current model's (≈ ⟪JUMP⟫× at $\\ell = $ ⟪ELL⟫) — this is what turns the roughly break-even per-model economics of Level 1 into a flow loss. Default 0.5 yr, U[0.25, 1.0] (grade B).", math_label='\\ell', uni_label='ℓ', short_name='training lead time', cal_target="the next model's compute is bought ~6 months ahead", grade='B'),
    'S0': ParamMeta(interp="**$S_0$ — today's training spend, \\$B/yr.** Combined frontier annual training-compute spend; anchors the cost scale (N3). 40 \\$B/yr (grade C).", math_label='S_0', uni_label='S₀', short_name='training cost today', cal_target='frontier labs spend ~\\$40B/yr on training compute today', grade='C'),
    'g_p': ParamMeta(interp="**$g_p$ — effective compute-price decline, OOM/yr.** 0.243 = log₁₀(4.2/2.4): chosen so cost growth reproduces Cottier's 2.4×/yr given compute growth 4.2×/yr — prices fall to $10^{-g_p} \\approx 57\\%$ of the year before. Hardware-only price-performance is ~1.35×/yr ($g_p \\approx 0.13$) — the three anchors can't all hold; we keep the two better-measured. Grade B/C; log10 units.", math_label='g_p', uni_label='g_p', short_name='price decline', cal_target='the training bill grows ~×2.4/yr despite ×4.2 compute (prices fall ~43%/yr)', cal_alt='**Tension:** the three cost anchors (Cottier 2.4×/yr, hardware 1.35×/yr, compute 4.2×/yr) are mutually inconsistent — a standing agenda item.', grade='B'),
    'r': ParamMeta(interp='**$r$ — discount rate, per yr.** Only used by the ownership user-cost extension (II.6); the widget reports **undiscounted** profit flows, not NPV, so $r$ is hidden unless II.6 is on. 0.08 (grade C).', math_label='r', uni_label='r', short_name='discount rate', cal_target='\\$1 next year ≈ \\$0.92 today', grade='C'),
    'T': ParamMeta(interp='**$T$ — horizon, yr.** The time window every graph uses. Switch between **5 yr** and **10 yr** with the toggle at the top of the sidebar; default 10 yr (the walkthrough horizon). The N5 harvest condition is asymptotic and reported analytically, not plotted beyond 10 yr.', uni_label='T', short_name='horizon'),
    'delta_total': ParamMeta(math_label='\\delta', uni_label='δ', short_name='catch-up rate', cal_alt='**Alt:** the transient / DeepSeek fast-catch-up reading lives in the MC upper tail (δ ≈ 1.0).'),
    't_compute_x': ParamMeta(uni_label='compute ×/yr', short_name='compute scaling today'),
    't_algo_x': ParamMeta(uni_label='algo ×/yr', short_name='algorithmic progress today'),
    't_lag_mo': ParamMeta(uni_label='lag (mo)', short_name='follower lag'),
    't_bill_x': ParamMeta(uni_label='bill ×/yr', short_name='training-bill growth today'),
    't_profit_B': ParamMeta(uni_label='profit₀ $B', short_name='frontier profit today'),
    't_value_x': ParamMeta(uni_label='value ×/OOM', short_name='value multiplier per OOM'),
    't_floor_x': ParamMeta(uni_label='floor ×/yr', short_name='long-run compute floor'),
}

# ---- derived views: the historical dict names, rebuilt from PARAMS. Each omits EXACTLY the keys
# whose field is None, preserving the original key presence the .get() fallbacks at call sites
# depend on. INTERP is used as BOTH the slider help= tooltip and the "what does this value mean?"
# popover; _MATH_LABEL/_UNI_LABEL/_SHORT_NAME are the three label maps; _CAL_TARGET/_CAL_ALT are
# the calibration captions; GRADES the grounding grades.
INTERP = {k: v.interp for k, v in PARAMS.items() if v.interp is not None}
_MATH_LABEL = {k: v.math_label for k, v in PARAMS.items() if v.math_label is not None}
_UNI_LABEL = {k: v.uni_label for k, v in PARAMS.items() if v.uni_label is not None}
_SHORT_NAME = {k: v.short_name for k, v in PARAMS.items() if v.short_name is not None}
_CAL_TARGET = {k: v.cal_target for k, v in PARAMS.items() if v.cal_target is not None}
_CAL_ALT = {k: v.cal_alt for k, v in PARAMS.items() if v.cal_alt is not None}
GRADES = {k: v.grade for k, v in PARAMS.items() if v.grade is not None}

# ---- target sliders (D-037): the control IS the observable; the implied parameter renders as a
#      live caption underneath. One source of truth: bounds/defaults/MC all come from the
#      notebook's TARGET_RANGES / target_defaults.
INTERP_T = {
    "t_compute_x": "**Compute scaling today (×/yr).** How fast frontier training compute grows — "
                   "Epoch: **4.2×/yr**, several series agree (grade A). Sets the log-compute "
                   "growth rate: $g_{c0} = \\log_{10}(\\cdot)$.",
    "t_algo_x": "**Algorithmic progress today (×/yr).** The effective-compute gain from better "
                "algorithms per year — Ho et al. and Whitfill et al.: **2.8×/yr** (grade A). Sets "
                "$g_a = \\log_{10}(\\cdot)$.",
    "t_lag_mo": "**Follower lag (months).** How far the open-source / fringe follower trails the "
                "frontier today — METR agentic ≈ **8 months**; benchmark lags are *lower bounds* "
                "(benchmaxxing). ONE fact, applied per level: it sets the initial "
                "gap $\\Delta_0$ = lag × leader speed AND the catch-up rate(s) that keep the lag "
                "constant — the merged $\\delta = 12/\\text{lag}$ at the pure-catch-up levels, the "
                "two channels $\\delta_{dev}, \\delta_{rel}$ (rescaled from the joint calibration) "
                "once the follower has its own engine (Level 6). Moving *other* targets (compute, "
                "algo) changes what the same lag means in OOMs — the caption updates live.",
    "t_bill_x": "**Training-bill growth today (×/yr).** How fast the frontier training bill grows "
                "— Cottier et al.: **2.4×/yr** (grade B). Given the compute-scaling target, sets "
                "the effective price decline $g_p = g_{c0} - \\log_{10}(\\cdot)$ (OOM/yr).",
    "t_profit_B": "**Frontier profit today (\\$B/yr).** The combined frontier labs' operating "
                  "profit on their capability lead, $\\theta\\,\\Delta W$ — observed ≈ "
                  "**\\$40–60B/yr**. Pins the value scale $W_0$ ($\\theta$ keeps its independent "
                  "Cournot anchor).",
    "t_value_x": "**Value multiplier per OOM (×).** How much more a model one OOM more capable is "
                 "worth — (1, 3]× per OOM from per-task token blow-up / cost-intensity (grade C). "
                 "Sets the value curvature $\\nu = \\log_{10}(\\cdot)$ (value-OOMs per "
                 "capability-OOM).",
    "t_floor_x": "**Long-run compute floor (×/yr).** Compute scaling once power/fab/capital limits "
                 "bind (~2030) — hardware-only price-performance ≈ **1.35×/yr** (grade C, our "
                 "extrapolation). Sets $g_{c\\infty} = \\log_{10}(\\cdot)$.",
}

# label, slider step, display format for each target slider (bounds come from TARGET_RANGES).
# Labels stay ONE line in the ~200px compact-row label cell (QA S7) — the "today"/long-form
# wording lives in INTERP_T (the hover help) and the calibration panel.
TSPEC = {
    "t_compute_x": ("Compute scaling (×/yr)", 0.01, "%.2f"),
    "t_algo_x":    ("Algo progress (×/yr)", 0.01, "%.2f"),
    "t_lag_mo":    ("Follower lag (months)", 0.05, "%.1f"),
    "t_bill_x":    ("Bill growth (×/yr)", 0.01, "%.2f"),
    "t_profit_B":  ("Frontier profit (\\$B/yr)", 0.5, "%.0f"),
    "t_value_x":   ("Value per OOM (×)", 0.01, "%.2f"),
    "t_floor_x":   ("Compute floor (×/yr)", 0.01, "%.2f"),
}

# "What's new at this level" cards: (heading, plain-language mechanism, equation, small print)
LEVEL_CARDS = {
    1: ("Basics — the race and the money",
        "A **leader** pushes the capability frontier $x^L$ at a constant speed; a **follower** "
        "trails at $x^F$ and catches up at rate $\\delta$. The leader earns the **rent on its "
        "lead** — margin $\\theta$ times the extra value it can offer over the follower — and "
        "**pays for the compute of the model it is running right now**.",
        "\\text{revenue} = \\theta\\,[\\,W(x^L) - W(x^F)\\,], \\qquad "
        "\\dot\\Delta = \\dot x^L - \\delta\\,\\Delta",
        "This is the complete base model — five controls, nothing hidden: compute scaling ⇒ "
        "$g_c$, follower lag ⇒ $\\delta$, value per OOM ⇒ $\\nu$ (each extra order of magnitude "
        "of capability translates into $\\nu$ orders of magnitude of value), plus the margin "
        "$\\theta$ and today's training bill $S_0$. The follower has **no engine of its own** — "
        "pure catch-up — so the observed lag pins $\\delta \\approx$ ⟪DELTA⟫/yr and the gap "
        "**holds steady at $\\Delta_0$**."),
    2: ("Training in advance",
        "**Adds the training lead time $\\ell$.** A frontier model isn't free when it ships — its "
        "compute is **bought $\\ell$ years in advance**. So today's bill is for the *next*, bigger "
        "model: it jumps by $10^{\\,g_c\\ell}$ (≈ **⟪JUMP⟫×** at $\\ell = $ ⟪ELL⟫).",
        "\\text{cost}(t) = S_0\\,10^{\\,c^L(t+\\ell)}\\,10^{-g_p t}",
        "Funding the *next* model while the current one only breaks even turns **today's flow "
        "profit negative** — the reported **Anthropic-vs-OpenAI** contrast (profit per model vs "
        "loss while scaling), depending on calibration. While compute growth is constant $\\ell$ "
        "doesn't change the *shape* of the path; it starts to bite once the slowdown (Level 4) "
        "bends the compute curve."),
    3: ("Where growth comes from",
        "**Adds the growth engine.** The base model's constant algorithmic rate $g_a$ becomes a "
        "*produced* quantity — driven by research inputs, and crucially by AI speeding up its "
        "*own* R&D: the **$\\psi$ feedback** (strength $\\gamma$) makes progress compound.",
        "\\dot a^L \\;\\propto\\; \\psi(x),\\qquad \\psi(x) = 1 + \\rho_0\\,e^{\\gamma x}",
        "The new feedback is live at its default $\\gamma = 0.2$, so paths steepen vs the base "
        "model. Above $\\gamma \\approx 0.42$ it goes super-exponential (blow-up) inside the "
        "horizon. The dashed 25% line on the gap chart marks where the feedback stops being "
        "a small correction. (Compute growth is still constant — the slowdown arrives at Level 4.)"),
    4: ("The compute slowdown",
        "**Adds the compute slowdown.** Today's ~⟪GC_X⟫×/yr compute scaling **can't persist** — "
        "power, fabs and capital all bind. Compute growth now **decays** from today's rate "
        "$g_{c0}$ toward a long-run floor $g_{c\\infty}$ at rate $\\xi$.",
        "g_c(t) = g_{c\\infty} + (g_{c0} - g_{c\\infty})\\,e^{-\\xi t}",
        "Teachable consequence: the leader **slows** while $\\delta$ is fixed, so the base model's "
        "stationary gap now starts **closing** — the follower's catch-up outruns the decelerating "
        "leader. It becomes stationary again at Level 6, where the follower gets its own engine "
        "and $\\delta$ is recalibrated down. (This is also where $\\ell$ starts to matter for cost.)"),
    5: ("Value saturation",
        "**Adds the bend in the value curve.** Until now the dollar value of capability grows "
        "**exponentially forever**. Now the value curve gains its **saturation bend $x_{mid}$**: "
        "past that point extra capability is worth steeply less — **commoditization and demand "
        "limits** mean value cannot compound without end.",
        "W(x) = \\frac{W_{\\max}}{1 + 10^{-\\nu (x - x_{mid})}}",
        "Where the bend sits can **flip the profitability verdict at the horizon**: an early bend "
        "(small $x_{mid}$) caps the rent the leader can ever earn, a late bend lets the harvest "
        "continue across the whole horizon."),
    6: ("How the follower catches up",
        "**The follower gets its own engine.** Until now it was pure catch-up; now it has its own "
        "**compute path** and its own **algorithmic progress $g_a^F$**, and the single $\\delta$ "
        "unpacks into **two channels**: ambient **developed-model diffusion $\\delta_{dev}$** "
        "(happens regardless of release, but only carries algorithmic knowledge) and "
        "**released-model distillation $\\delta_{rel}$** (the follower learns from what the leader "
        "*serves*). The follower's own compute and starting gap $\\Delta_0$ are now yours to set.",
        "\\dot a^F = g_a^F + \\delta_{dev}\\,(a^L - a^F) + \\delta_{rel}\\,\\max(x^R - x^F,\\,0)",
        "Because the follower now supplies most of its own speed, the catch-up channels only have "
        "to close the remaining ~⟪WEDGE⟫ OOM/yr wedge: $\\delta_{dev} = $ ⟪DDEV⟫ and "
        "$\\delta_{rel} = $ ⟪DREL⟫ (effective $\\approx \\delta_{rel} + "
        "\\tfrac{1}{2}\\delta_{dev} \\approx$ ⟪DEFF⟫) "
        "replace the base model's single $\\delta \\approx$ ⟪DELTA⟫ — and the gap is stationary "
        "again. "
        "$\\delta_{rel}$ is the channel a release delay could throttle; "
        "$\\delta_{rel} = 0$ disables distillation."),
    7: ("The release-delay question",
        "The leader can **serve an older model** $x^R$ than it has (delay $\\tau$): this throttles "
        "distillation $\\delta_{rel}$ but forgoes revenue now, since revenue is earned on the served "
        "model. A new **Release delay** view weighs the trade-off.",
        "x^R_t = x^L_{t-\\tau}, \\qquad \\text{revenue} = \\theta\\,[\\,W(x^R) - W(x^F)\\,]",
        "The capability chart now shows the served $x^R$ (dashed) whenever $\\tau > 0$."),
    8: ("The R&D overhead on cost",
        "**Adds the R&D overhead.** The base cost already tracks the leader's actual compute path "
        "(pay now for the model shipping $\\ell$ ahead, at prices falling at $g_p$). Level 8 marks "
        "up that compute bill by the **R&D overhead $\\phi_{RD}$** — staff and experiments on top "
        "of the final training run — and frees the price-decline dial $g_p$.",
        "\\text{cost}(t) = (1+\\phi_{RD})\\,S_0\\,10^{\\,c^L(t+\\ell)}\\,10^{-g_p t}",
        "At $\\phi_{RD} = 1$ the bill roughly doubles (experiments + staff ≈ the final run). The "
        "compute-path shape is unchanged — this only lifts the level and lets you retune the price "
        "decline."),
    9: ("Extensions — the full model",
        "The **complete widget**: the value scale $W_0$, the research-CES knobs ($\\alpha$, "
        "$\\eta$), and the extension dials (II.4 gap-dependent $\\theta$, II.5 distillation "
        "defenses, II.6 ownership user-cost). **Under the hood** exposes the live model code.",
        "",
        "The base model plus every mechanism the levels added, and the fine-tuning dials on top — "
        "nothing further is hidden."),
}

# Notation sections revealed cumulatively as levels unlock.
NOTATION_SECTIONS = {
    1: "**Who is who.** *Leader* = the frontier lab(s), one player; *Follower* = open-source / "
       "competitive fringe. Leader capability $x^L$, follower $x^F$; the gap is "
       "$\\Delta = x^L - x^F$. The follower closes it at a single catch-up rate $\\delta$.\n\n"
       "**Units.** Capability is in **OOM** = orders of magnitude (factors of 10) of effective "
       "compute, measured *above the early-2026 frontier* (today = 0). Rates are OOM/yr; dollars "
       "are \\$B/yr.\n\n"
       "**Money.** $W(x)$ = the dollar value of capability $x$. $\\theta$ = the leader's operating "
       "margin (share of the value gap it keeps). Cost today is $S_0$ = the compute bill of the "
       "**current** frontier model. Profit is shown as **undiscounted yearly flows** (no NPV).",
    2: "**Training in advance.** $\\ell$ = training lead time: the firm pays now for the compute of "
       "the model shipping at $t+\\ell$, so today's bill is $S_0\\,10^{c^L(\\ell)}$ (bigger than "
       "$S_0$). At $\\ell = 0$ (Level 1) it pays only for the current model.",
    3: "**Growth engine.** Capability = compute + algorithmic progress. $\\psi$ = the AI-R&D "
       "self-improvement multiplier (RSI = recursive self-improvement), $\\psi = 1+\\rho_0 e^{\\gamma x}$; "
       "*$\\psi$-share* is how much of algorithmic progress comes from that feedback (the dashed "
       "25% reference line marks where it stops being small). $\\gamma$ = its compounding strength.",
    4: "**Compute slowdown.** Compute growth is no longer constant, and the notation splits with "
       "it: the plain $g_c$ of Levels 1–3 becomes **today's rate $g_{c0}$**, and $g_c(t)$ decays "
       "from it toward a long-run floor $g_{c\\infty}$ at rate $\\xi$ (power/fab/capital limits). "
       "With $\\delta$ fixed, the slowing leader lets the gap start closing.",
    5: "**Value saturation.** $W(x)$ bends from exponential to saturating at $x_{mid}$ (OOM above "
       "today's frontier): past the bend, extra capability is worth steeply less — commoditization "
       "and demand limits. $\\nu$ sets the slope before the bend.",
    6: "**Two catch-up channels.** **developed-model diffusion $\\delta_{dev}$** — ambient, happens "
       "even with nothing released — and **released-model distillation $\\delta_{rel}$** — learning "
       "from the served model. The base model's single $\\delta$ unpacks into these two here, and "
       "the follower gains its own engine ($g_a^F$, own compute). $\\Delta_0$ = the initial gap.",
    7: "**Two versions of the leader's model.** **Developed** $x^L$ — the best model the leader "
       "*has*; **released / served** $x^R$ — what it actually deploys. With a release delay $\\tau$, "
       "$x^R$ is $x^L$ from $\\tau$ months ago. Revenue and distillation both run on $x^R$.",
    8: "**R&D overhead.** The base cost already tracks the compute path (lead time $\\ell$, price "
       "decline $g_p$); Level 8 adds the R&D markup $\\phi_{RD}$ on top of the final training run "
       "and frees the $g_p$ dial.",
    9: "**Extensions.** value scale $W_0$, research CES ($\\alpha$, $\\eta$), and dials II.4 "
       "(gap-dependent $\\theta$), II.5 (distillation defenses), II.6 (ownership user-cost, which "
       "also exposes the discount rate $r$).",
}

# ---- live numbers in static text: derived figures (jump factors, implied δ, ×/yr multipliers)
# quoted inside INTERP / LEVEL_CARDS prose must track the CURRENT effective parameters, never
# freeze at the defaults. Templates carry ⟪TOKEN⟫ placeholders (plain ⟪⟫ so LaTeX braces are
# untouched) substituted at render time from the same dict the simulation uses.
def _live_vals(d):
    """Current effective values for the ⟪TOKEN⟫ placeholders (fallbacks = the pinned defaults)."""
    gc = d.get("g_C0", P0.g_C0)
    ga = d.get("g_a", P0.g_a)
    # L1 pins ℓ=0 → hypothetical default; mid-sidebar the slider state is already live
    ell = d.get("ell") or st.session_state.get("w_ell") or P0.ell
    Delta0 = d.get("Delta0", P0.Delta0)
    speed = gc + ga
    gaf = d.get("g_a_F") or P0.g_a_F        # below L6 the engine is pinned to 0 → hypothetical
    gcf = d.get("g_CF0") or P0.g_CF0
    ddev, drel = m.channels_from_lag(Delta0 / speed, speed, gaf + gcf)
    return {
        "JUMP": f"{10.0 ** (gc * ell):.2f}", "ELL": f"{ell:.2f}",
        "GC_X": f"{10.0 ** gc:.1f}", "DELTA": f"{speed / Delta0:.2f}",
        "WEDGE": f"{max(speed - (gaf + gcf), 0.0):.2f}",
        "DDEV": f"{ddev:.2f}", "DREL": f"{drel:.2f}", "DEFF": f"{drel + 0.5 * ddev:.2f}",
    }


def _sub_live(txt, d):
    """Substitute ⟪TOKEN⟫ placeholders with current values (no-op for token-free text)."""
    if txt and "⟪" in txt:
        for k, v in _live_vals(d).items():
            txt = txt.replace("⟪" + k + "⟫", v)
    return txt

# (_CAL_TARGET — the one-line "observable FACT it is calibrated to" caption per parameter — and
# _CAL_ALT — alternative calibrations / documented tensions for the details popover — are now
# derived views over PARAMS (see the registry above), not standalone dicts.)

# ---- calibration sources: the modal's source-picker table now lives in the NOTEBOOK
# (cell E8b, D-042) as `CAL_SOURCES`, because it also derives the tight default simulation
# ranges (`SIM_DEFAULT`) — one source of truth for calibration data. The app only renders it.

_DELTA_MERGED_DOC = ("**$\\delta$ — merged catch-up rate (/yr).** At the base-model levels the "
                     "follower has no engine of its own, so $\\delta$ supplies its whole motion. "
                     "The lag target sets $\\delta = 12/\\text{lag}$, i.e. "
                     "$\\delta\\,\\Delta_0$ = the leader's speed exactly — the gap holds at "
                     "$\\Delta_0$ for any lag while compute growth is constant. At Level 4 the "
                     "compute slowdown decelerates the leader and the gap starts closing; from "
                     "Level 6 the follower's own engine covers most of its speed and the two "
                     "channels only close the ~⟪WEDGE⟫ OOM/yr wedge (effective "
                     "$\\delta \\approx$ ⟪DEFF⟫).")


def _fmt_range(rng):
    kind = rng[0]
    if kind == "uniform":
        return f"U[{rng[1]:g}, {rng[2]:g}]"
    if kind == "lognormal":
        med = float(np.exp(rng[1]))
        lo, hi = med * float(np.exp(-1.645 * rng[2])), med * float(np.exp(1.645 * rng[2]))
        return f"lognormal, median {med:.2f} (90% CI [{lo:.2f}, {hi:.2f}])"
    if kind == "triangular":
        return f"triangular [{rng[1]:g}, {rng[3]:g}], mode {rng[2]:g}"
    if kind == "scale_of":
        return f"U[{rng[2]:g}, {rng[3]:g}] × {_UNI_LABEL.get(rng[1], rng[1])}"
    if kind == "choice":
        return "choice {" + ", ".join(f"{v:g}" for v in rng[1]) + "}"
    return str(rng)


def _param_word_label(k):
    """'δ_rel — distillation' — unicode symbol plus plain words; never the raw code name."""
    sym = _UNI_LABEL.get(k, k)
    words = _SHORT_NAME.get(k)
    return f"{sym} — {words}" if words else sym
