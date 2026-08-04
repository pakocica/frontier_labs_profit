"""Content & parameter metadata — the ONE app-side home for prose and labels.

Per-parameter metadata lives in a single registry `PARAMS: dict[str, ParamMeta]` (D-061); the
historical dict names — interpretation texts (INTERP), the three label maps (_MATH_LABEL /
_SHORT_NAME / _UNI_LABEL: LaTeX / plain words / unicode), calibration captions (_CAL_TARGET /
_CAL_ALT) and grades (GRADES) — are thin DERIVED VIEWS over it, so consumers are unchanged.
Target-specific content (INTERP_T, TSPEC) and level intros stay as their own maps.
(The notation map is retired — D-096.)
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
#   short_tip  — SHORT_TIP (D-078 follow-up, Pavel): the ≤~90-char one-liner the sidebar
#                row-title tooltip shows — what it is + units, NO calibration rationale (that
#                stays in the » panel via `interp`). A missing entry means NO tooltip, never a
#                fallback to the long text. New field, so the frozen content snapshot (which
#                pins the seven pre-existing views) is untouched by construction.
@dataclass
class ParamMeta:
    interp: str | None = None
    math_label: str | None = None
    uni_label: str | None = None
    short_name: str | None = None
    cal_target: str | None = None
    cal_alt: str | None = None
    grade: str | None = None
    short_tip: str | None = None


PARAMS: dict[str, ParamMeta] = {
    'gamma': ParamMeta(interp='**$\\gamma$ — how fast the AI-R&D speedup grows, per OOM.** The GROWTH rate of the **recursive self-improvement** (RSI) feedback — $\\beta_0$ is how much faster AI makes AI R&D *today*, $\\gamma$ is how fast that advantage compounds as capability rises: each OOM of capability multiplies AI-R&D speed by $10^{\\gamma x}$ — so $\\gamma$ is **decades of R&D speed per OOM of capability**, the same units as every other slope in the model (base 10 throughout). $\\gamma = 0$ disables it entirely (freeze). $\\gamma \\gtrsim 0.182$ goes super-exponential — finite-time blow-up inside the 10-yr horizon. Default 0.087, tentative (grade C). *(Both numbers were stated in nats under an earlier unit convention, where the same thresholds read 0.42 and 0.2 — an older note quoting those is not describing a different model.)*', math_label='\\gamma', uni_label='γ', short_name='RSI growth (how fast the speedup compounds)', cal_target='how strongly AI-for-AI-R&D compounds (no observable yet)', grade='C'),
    # D-132 (Pavel, 2026-08-03): spot 10 → 6, envelope [2, 20] → [2, 12], an evidenced menu, and
    # the grade earns its way F → C. ⟪VAL_M_IMPLIED⟫ / ⟪VAL_M_COHERENT⟫ are the XM5 cross-check.
    'x_mid': ParamMeta(interp='**$x_{mid}$ — value-slope transition midpoint, OOM above the 2026 frontier.** Not a half-saturation point: the value slope $w\'(x)$ rides the universal transition curve from $\\nu$ down to the floor $\\nu_\\infty$, and $x_{mid}$ is where that transition is HALF-DONE ($q^w_0$% of it at today\'s frontier, $(100-q^w_0)$% at $2x_{mid}$). It is a **pure scale on the capability axis** — the curve depends on $x$ and $x_{mid}$ only through $x/x_{mid}$ — so a source pins it only by giving a *location* and a *completion fraction* together. Most published statements are full-automation dates, i.e. ~99% complete, which this dial reads at **half** their face value. Default **6.0 OOM**: Epoch\'s GATE ramp read at *half of tasks automated*, and corroborated to 0.12 OOM by a chain sharing no input with it — the ruler-free level identity $x_{mid} = \\lambda(q^w_0)(1-q^w_0/100)/\\nu$. **This is the gate.** It is the only dial in the value block whose envelope contains a sign flip of the Level-2 verdict: below $x_{mid} \\approx 4.0$ the leader fails to cover its model-building bill *even granting the optimist his own 10 %/yr floor*. Grade C — earned, not asserted, and B is unreachable until somebody measures the total remaining addressable-value multiple, because no source fits a dollar-valued logistic in capability OOMs.', math_label='x_{mid}', uni_label='x_mid', short_name='value-slope transition midpoint', cal_target="half of tasks automated ~6 OOM above today's frontier (Epoch GATE)", cal_alt='**Coherence check (the value block\'s only one):** read as a bounded market, $(x_{mid}, q^w_0)$ over-determine the total value multiple $W(\\infty)/W(0)$. This pair implies **×⟪VAL_M_IMPLIED⟫** while $q^w_0$ itself asserts **×⟪VAL_M_ASSERTED⟫**; they agree exactly at $x_{mid}$ = ⟪VAL_M_COHERENT⟫.⟪VAL_M_WARN⟫ It binds only under that reading — with $\\nu_\\infty > 0$ there is no ceiling — so a large gap is a *warning*, not an error. The retired default (10.0) sat 18.5× out. **Read the pair, not the number:** $\\nu_\\infty$ bites only through this dial.', grade='C'),
    # D-083: the ASYMPTOTIC value slope — an app target (t_value_growth_inf), like ν's
    # t_value_growth. Calibrated by D-107, re-keyed to a growth rate by D-118, to ×/yr by D-133.
    'nu_inf': ParamMeta(interp='**$\\nu_\\infty$ — asymptotic value slope, value-OOMs per capability-OOM.** Where the value slope $w\'$ lands once the transition is done: far past $x_{mid}$, each further OOM of capability is worth $10^{\\nu_\\infty}$× more — the hard ceiling $\\bar W$ is retired, value grows without bound at this floor slope. **You do not dial this number; you dial the growth rate it implies.** Long-run AI-attributable value growth is $\\nu_\\infty \\times \\dot x^L$, so the control is that rate in ×/yr and the slope is derived at the frontier speed the horizon actually reaches (0.409 OOM/yr at the defaults). Default **×1.10/yr** (10 %/yr) ⇒ $\\nu_\\infty \\approx 0.101$ (×1.26/OOM, against ≈×2.10 today): the floor of Amodei\'s own 10–20% forecast, granted deliberately — if the leader cannot cover its bill even on the most favourable published view of what capability is worth, the verdict does not rest on our pessimism. Envelope **[×1.00, ×1.30]/yr** — 0 to 30 %/yr — runs from stagnation to the explosive-growth threshold, and the » menu lets you pick a published worldview instead. $\\nu_\\infty = \\nu$ switches the transition off (the Level-1 pin). Grade C: nothing measures an asymptotic value elasticity, but the model\'s own identity turns it into a growth rate the macro literature bounds directly.', math_label='\\nu_\\infty', uni_label='ν_inf', short_name='asymptotic value slope', cal_target="long-run AI-attributable value grows ~10%/yr — the optimist's floor, granted", cal_alt='**Conditional, not pivotal:** $\\nu_\\infty$ bites only through $x_{mid}$. At the shipped midpoint the frontier never reaches the transition, so the whole envelope moves year-5 coverage by ~4 pp and flips no verdict; bring the midpoint within reach and the same sweep spans harvest to commoditization. Sweep the two together.', grade='C'),
    'delta_dev': ParamMeta(interp='**$\\delta_{dev}$ — developed-model diffusion, per yr.** Release-invariant (ambient) catch-up: the follower closes the *algorithmic* gap via talent, published methods and ambient know-how even with nothing released. Default 0.20/yr, a point value — it is *derived* from the Fringe-lag dial, never sampled on its own. Jointly calibrated with $\\delta_{rel}$ so the observed ~7-month lag is stationary (see $\\delta$); weakly identified (grade C).', math_label='\\delta_{dev}', uni_label='δ_dev', short_name='ambient diffusion', cal_target='keeps the ~7-mo lag constant — the ambient share of the wedge', grade='C'),
    'delta_rel': ParamMeta(interp='**$\\delta_{rel}$ — released-model distillation, per yr.** Release-controlled channel: the follower distills from the model the leader serves — the developed frontier $x^L$ (the release-delay extension that would let the served model lag it is parked). This is the lever a release delay would act on. Default 0.26/yr, a point value — it is *derived* from the Fringe-lag dial, never sampled on its own; jointly calibrated with $\\delta_{dev}$ for lag stationarity (grade C). **$\\delta_{rel} = 0$ = distillation disabled** — released channel off; the follower catches up through $\\delta_{dev}$ only.', math_label='\\delta_{rel}', uni_label='δ_rel', short_name='distillation', cal_target='keeps the ~7-mo lag constant — the distillation share of the wedge', grade='C'),
    'g_C_inf': ParamMeta(interp='**$g_{c\\infty}$ — compute-growth floor, OOM/yr.** Long-run compute growth once scaling hits limits (power ~2030). 0.15 OOM/yr ≈ 1.41×/yr, and it is a SUM OF TWO LEGS in these units: hardware price-performance (0.14 ≈ ×1.38/yr, the same price of compute $g_p$ the cost side uses) plus the real growth of the training budget (0.01 ≈ +2.3 %/yr, long-run economic growth). That decomposition is an identity here rather than an analogy — compute bought = dollars spent × FLOP per dollar, and the model carries exactly one price series — so this dial has an exact read-out: implied long-run budget growth $10^{g_{c\\infty}-g_p}$ = **×⟪FLOOR_BILL_X⟫/yr**. Read it: below ×1.00 the leader\'s real budget is shrinking forever, above it the budget outgrows the economy. The floor *level* is still an extrapolation (grade C) — no series measures post-2030 frontier compute growth — and it is widget-critical for the slowdown scenario.', math_label='g_{c\\infty}', uni_label='g_c∞', short_name='compute-growth floor', cal_target='long-run compute grows ~×1.41/yr — hardware ×1.38 × budget +2.3%/yr', cal_alt='**Read the pair, not the number:** every setting of this dial names an implied long-run budget growth. The alternative readings are the flat-budget world (×1.38/yr, hardware alone) and the AI-boosted one (×1.52/yr, budget at 10%/yr). **Note:** the floor *level* is an extrapolation past 2030, not a measurement.', grade='C'),
    'tau': ParamMeta(interp='**$\\tau$ — release / withholding delay, months.** PARKED. Policy lever: $x^R_t = x^L_{t-\\tau}$, the leader serves the model it had $\\tau$ ago. $\\tau = 0$ = release immediately (baseline). Capped at the 3-month policy-relevant range. Grade F by design (a decision variable, not calibrated).', uni_label='τ', short_name='release delay', cal_target='the policy lever itself — chosen, not calibrated'),
    'g_C0': ParamMeta(interp='**$g_c$ — compute growth at the CAPABILITY frontier, OOM/yr.** 0.511 = log10(3.24), i.e. **3.24×/yr**. This is the growth of the training compute behind the *most capable* model — not of the largest training run (Epoch’s 4.2×/yr series), which is a different object: the two diverged in 2023–25 as labs shifted spend into post-training and inference. Two independent routes meet at the spot: the dollar identity 2.4 (bill growth) × 1.35 (the *cluster-midpoint* hardware reading, not the measured price leg $g_p$ = 1.38 the cost side runs on), and Epoch’s capability-frontier reading of 3–4×/yr. This dial means **today** at every level: the transition curve\'s upper plateau $g_c^{pre}$ is *derived* inside $\\Gamma$, so moving the position dial $q^c_0$, the midtime or the floor changes where the slowdown started, never what the rate is now. Grade A.', math_label='g_c', uni_label='g_c', short_name='compute growth today', cal_target='the compute behind the most capable model grows ~×3.24/yr', cal_alt='**Different object:** the compute-frontier series (largest run) reads 4.2–5.3×/yr — kept in the menu, labelled, but it is not what $x^L$ tracks.', grade='A'),
    't_mid': ParamMeta(interp='**$t_{mid}$ — slowdown midpoint, yr.** The compute-growth path is the universal transition curve $g_c(t) = \\Gamma(t;\\, g_c,\\, g_{c\\infty},\\, t_{mid},\\, q^c_0)$: half the slowdown has played out by $t_{mid}$ — $q^c_0$% of it today and $(100-q^c_0)$% by $2t_{mid}$, so the transition occupies roughly the window $[0, 2t_{mid}]$. Replaces the retired decay rate $\\xi$ (default 2.3 yr ≈ the old $\\xi = 0.3$ half-decay time $\\ln 2/0.3$). Pure scenario knob (grade F) — sweep it.', math_label='t_{mid}', uni_label='t_mid', short_name='slowdown midpoint', cal_target='when the compute slowdown is half-done (scenario dial)', grade='F'),
    'g_a': ParamMeta(interp='**$g_a$ — algorithmic progress, OOM/yr — a RESIDUAL, not an estimate.** 0.544 = log10(3.5), i.e. 3.5×/yr. The *observable* is *effective*-compute growth — **11.34×/yr**, everything that is not physical FLOP: architecture, data, and post-training know-how — and $g_a = g_{eff} - g_c$. Defining it this way makes the RL-compute double-count structurally impossible: whatever counts as compute is by construction not counted again as algorithms. Identity: $11.34 = 3.24 \\times 3.50$. Reference-dependent (the same model sequence shows 63%/yr against an LSTM reference and 0%/yr against a dense-Transformer one) — the reference here is 2026 frontier practice at frontier scale. Grade B.', math_label='g_a', uni_label='g_a', short_name='algo progress (residual)', cal_target='effective compute grows ×11.3/yr, of which ×3.24 is physical compute', cal_alt='**Bounds:** pretraining-only measurements (2.2–3.2×/yr) are *lower* bounds — they exclude post-training know-how, which this model puts in $a$. Index-derived readings (5.9–9×/yr) are *upper* bounds — they include capability bought at inference time, which is in neither $a$ nor $c$.', grade='B'),
    'alpha': ParamMeta(interp='**$\\alpha$ — experiment-compute weight in the CES research bracket.** Share of algorithmic progress carried by experiment-compute growth rather than by AI-assisted researchers. Calibrated at **0.70**, range [0.45, 0.90], dialled through the observable "% of progress lost if compute growth halves" — $1 - 2^{-\\alpha}$ at the base $\\eta = 0$, so **38.44% ⇒ 0.70**. Inert at Level 1. The **delivered** weight moves with $\\eta$: the shipped 38.44% drag implies $\\alpha = 0.70$ at $\\eta = 0$, $0.74$ at $0.61$, $0.77$ at $1$ and $0.61$ at $-1.2$, which is what counts the bottleneck evidence once rather than twice. The menu\'s rows are each read at their OWN construct\'s $\\eta$: a compute cost SHARE is a Cobb-Douglas object (under CD the output elasticity equals the cost share), while the growth reading is natively an $\\eta = 1$ one.', math_label='\\alpha', uni_label='α', short_name='experiment-compute weight', cal_target='% of algo progress lost if experiment-compute growth halved', grade='C'),
    'eta': ParamMeta(interp='**$\\eta$ — CES exponent for compute–labor substitution in research.** A continuous dial over **[−1.20, +1.00]**, read through $\\sigma = 1/(1-\\eta)$: **$\\eta = 0$ is the base** — Cobb-Douglas, $\\sigma = 1$, the recursive-self-improvement literature\'s own convention (Davidson et al. 2026) and the only setting at which $\\alpha$ is identified by the compute cost share; $\\eta = 0.61$ is Whitfill & Wu 2025\'s estimated $\\sigma = 2.58$ (grade B, the menu\'s one measurement); $\\eta < 0$ makes the two inputs complements, so the scarcer one rules. $\\eta = 1$ — perfect substitution, a weighted average — was the default until it was superseded: a tractability convention with no literature behind it, and it is the CES family\'s mathematical ceiling, since $\\sigma$ is negative above it. *(The left end stops at −1.20, the most complementary value any cited publication states as a number. Neither the $\\eta = -2$ stand-in nor the $\\eta \\to -\\infty$ Leontief limit is offered: the complements spec behind them estimates $\\sigma = -0.10$ (SE 0.176), i.e. no finite negative $\\eta$ at all. Leontief is still in the model, where $\\alpha$ is provably unread.)*', math_label='\\eta', uni_label='η', short_name='research elasticity', cal_target='how substitutable compute and researchers are in R&D', grade='B'),
    # D-084: RENAMED rho0 -> beta0. rho was doing double duty — the coverage ratio rho_t and
    # its t = 0 dial rho_0 = m/k (D-080) — and coverage keeps the letter (it is the reported
    # outcome, and c/C are compute while k and m are taken), so the RSI feedback scale moves.
    # D-132 (Pavel, 2026-08-03, B1): the INPUT reading is ruled, and `cal_target` / `short_name`
    # move with it — they carried the OUTPUT reading while `interp` and the code carried the
    # input one, and at the retired η = 1 the two differed by 10× on the same card.
    'beta0': ParamMeta(interp='**$\\beta_0$ — today\'s AI-assistance level in AI R&D.** $\\psi(0) = 1 + \\beta_0$ is the counterfactual multiplier on frontier-lab research **throughput**: $\\sigma_0 = \\beta_0/(1+\\beta_0)$ is the share of today\'s throughput that AI assistance is responsible for — what would be lost if assistance were withdrawn with the human workforce held fixed. It is an **input level, not an output speedup**, and that is the ruled construct: a with-versus-without trial reads it off with no conversion, while an output speedup has to be inverted through the CES and so moves whenever $\\alpha$ or $\\eta$ moves. **$\\beta_0$ never changes today\'s rate** — $\\psi$ is normalised by $\\psi(0)$, so $\\dot a^L_0 = g_a$ bitwise at every $\\beta_0$; what it fixes is the *shape* of the assistance path. The object is **headroom, not speed**. 0.3 is an explicit **placeholder**, not a calibration — held rather than moved because the construct was ambiguous and the conversion changed underneath it. Envelope **[0.04, 3.00]**, both ends witnessed. Grade **F**, and it stays F even with the menu: the evidence is plentiful and none of it measures this object — every grade-A study is general developers on coding tasks, every frontier-lab reading is self-report or code-volume telemetry, and the two disagree by 15× on roughly the same population months apart. *(Renamed from $\\rho_0$: $\\rho$ is the coverage ratio $E_t/B_t$ — the reported outcome — and one letter cannot be both.)*', math_label='\\beta_0', uni_label='β₀', short_name='AI-assistance level in AI R&D today', cal_target='AI assistance supplies ~23% of frontier-lab research throughput today (placeholder)', cal_alt='**Only identified jointly with $\\gamma$.** Over ten years the model sees the scalar $R_\\psi(X)$, so a 75× move in $\\beta_0$ changes the answer by ~9% if $\\gamma$ moves with it and flips the Level-3 profit sign if $\\gamma$ is held fixed. A ruling here at fixed $\\gamma$ is a ruling about the strength of the $\\psi$ engine, not about today\'s assistance level — and $\\gamma$ has no adoptable source row at all, so the pair cannot be closed yet.', grade='F'),
    'Delta0': ParamMeta(interp='**$\\Delta_0$ — initial capability gap, OOM.** 0.615 OOM = the **7.0-month** fringe lag × the leader’s current speed. Months are the master: the *lag* is the observable and $\\Delta_0$ is derived from it, so moving the compute or effective-compute dial changes what the same lag is worth in OOMs. The 7.0 reading is on a **strict** catch-up rule and an agentic / long-horizon basis (UK AISI autonomous ranges; Epoch’s ECI on the strict rule reads 6.0). Grade A/B.', math_label='\\Delta_0', uni_label='Δ₀', short_name='initial gap', cal_target='the competitive fringe is ~7 months behind the frontier today', cal_alt='**Rule artefact:** public leaderboard headlines (~4 mo) use a *lenient* catch-up rule; the same data on the strict rule gives 6. Benchmark lags are also *lower bounds* (benchmaxxing).', grade='A/B'),
    'split': ParamMeta(interp='**split — algo share of $\\Delta_0$.** How much of the initial gap is algorithmic vs compute. 0.5 placeholder (grade F, open question).', math_label='\\text{split}', uni_label='split', short_name='algo share of the gap', cal_target='about half the initial gap is algorithmic, half compute', grade='F'),
    'g_a_F': ParamMeta(interp='**$g_a^F$ — follower algo progress, as a SHARE of the leader\'s $g_a$.** The dial is the follower/leader ratio itself: progress is scale-biased (the frontier improves faster than small scale — Gundlach; ratio ≈ 0.6–0.8, central 0.7), so $g_a^F = \\text{share} \\cdot g_a$ ≈ 0.38 OOM/yr at the defaults — and it TRACKS the effective-compute dial instead of silently detaching from it. The MC prior draws the same share band. Grade B.', math_label='g_a^F', uni_label='g_a,F', short_name='follower algo share', cal_target="follower algo progress ≈ 70% of the leader's", grade='B'),
    'g_CF0': ParamMeta(interp='**$g_c^F$ — fringe compute growth TODAY, OOM/yr** ($g^F_c(0)$, not a pre-slowdown plateau)**.** SCENARIO KNOB, grade F: no calibration yet — the 0.5 default is a placeholder, not a source statistic, and it (with the floor and midpoint) drives the Level-3 lag drift, so treat the level\'s forward path as a scenario until the follower-compute calibration pass (projected Chinese/fringe build-out) lands.', math_label='g_c^F', uni_label='g_cᶠ', short_name='follower compute growth', cal_target='scenario dial — follower-compute calibration pass to come', grade='F'),
    'g_CF_inf': ParamMeta(interp='**$g_{c\\infty}^F$ — follower compute-growth floor, OOM/yr.** SCENARIO KNOB, grade F (Q-5 ruling, extensions-sync round): the 0.10 default is a placeholder, not a source statistic — same standing as $g_c^F$; calibration pass to come.', math_label='g_{c\\infty}^F', uni_label='g_c∞ᶠ', short_name='follower compute floor', cal_target='scenario dial — follower-compute calibration pass to come', grade='F'),
    't_mid_F': ParamMeta(interp='**$t_{mid}^F$ — follower slowdown midpoint, yr.** The follower compute path rides the same universal transition curve as the leader: $g^F_c(t) = \\Gamma(t;\\, g^F_c,\\, g^F_{c\\infty},\\, t_{mid}^F,\\, q^F_0)$, half-done at $t_{mid}^F$ and $q^F_0$% along today. SCENARIO KNOB, grade F (Q-5 ruling) — like the leader\'s $t_{mid}$, chosen not calibrated.', math_label='t_{mid}^F', uni_label='t_mid_F', short_name='follower slowdown midpoint', cal_target="when the follower's slowdown is half-done (scenario dial)", grade='F'),
    'nu': ParamMeta(interp='**$\\nu$ — value curvature TODAY, value-OOMs per capability-OOM.** 0.323 = $\\log_{10}(2.19)/1.055$: each order of magnitude of capability commands **≈×2.10** more value at today\'s frontier. **You do not dial this number; you dial the growth rate it implies** — value grows at $\\nu\\,\\dot x^L$, so the control states that rate in ×/yr and the slope is derived at whatever frontier speed the other dials set. The calibrated anchor is the RATE, **×2.19/yr** (119%/yr) — a round number, deliberately; ×2.10/OOM is what it comes to at the calibrated speeds (an exact ×2.1/OOM would read 118.68%/yr, 0.14% away, and the evidence is nowhere near that sharp). Literally $w\'(0)$, at every $q^w_0$ — under an earlier convention it was the *pre-easing* slope, so dialling 2.1 delivered 2.089 today. The pooled central value across three independent constructions — Davidson’s value datum (1.86), a GATE-style automation ramp against the wage bill (2.30), and the revenue decomposition at this calibration’s frontier speed (2.24): an equally-weighted Monte Carlo over the three routes’ own input bands, $p50 = 2.08$/OOM, with a 90% band [1.7, 2.65]. *Not* the median of the three numbers quoted, which are each restated on this calibration’s ruler. **The most consequential number in the model:** the whole profitability race is $\\nu\\,\\dot x^L_t$ against $g_c - g_p$, and the break-even pivot sits at ≈⟪BE_VM⟫×/OOM. Single-benchmark slopes (SWE-bench 10.7, RLI 16) are *not* this number — a dollar-weighted mixture of channels with midpoints spread over ±2–3 OOM aggregates to ~2×/OOM even when individual channels ramp at 16×. Grade B.', math_label='\\nu', uni_label='ν', short_name='value slope', cal_target='each OOM of capability is worth ~×2.1 more', grade='B'),
    'g_p': ParamMeta(interp="**$g_p$ — effective compute-price decline, OOM/yr.** 0.14 = log₁₀(1.38): a dollar buys ~38% more training compute each year. This is now the **measured hardware price-performance leg** (Hobbhahn, Heim & Aydos 2023: 1.39×/yr, CI [1.27, 1.54]) taken as a trusted anchor — *not*, as before, a residual fitted so the bill matched. The consequence is a read-out, not a free parameter: implied bill growth $10^{g_c-g_p} = $ **2.35×/yr** against Cottier's observed 2.4×/yr — a 2% miss we document rather than fit away. Grade B.", math_label='g_p', uni_label='g_p', short_name='price decline', cal_target='training compute gets ~38% cheaper each year (measured hardware leg)', cal_alt='**Retired:** the old residual reading (1.75×/yr) reconciled a 4.2×/yr compute trend with the 2.4×/yr bill. With the capability-frontier definition of $g_c$ the three anchors are consistent to 2%, so the residual is no longer needed.', grade='B'),
    'r': ParamMeta(interp='**$r$ — discount rate, per yr.** Only used by the ownership user-cost extension (II.6); the widget reports **undiscounted** profit flows, not NPV, so $r$ is hidden unless II.6 is on. 0.08 (grade C).', math_label='r', uni_label='r', short_name='discount rate', cal_target='\\$1 next year ≈ \\$0.92 today', grade='C'),
    'T': ParamMeta(interp='**$T$ — horizon, yr.** The time window every graph uses. Switch between **5 yr** and **10 yr** with the toggle at the top of the sidebar; default 10 yr (the walkthrough horizon). The N5 harvest condition is asymptotic and reported analytically, not plotted beyond 10 yr.', uni_label='T', short_name='horizon'),
    'delta_total': ParamMeta(math_label='\\delta', uni_label='δ', short_name='catch-up rate', cal_alt='**Alt:** the transient / DeepSeek fast-catch-up reading lives in the MC upper tail (δ ≈ 1.0).'),
    # D-093: κ, B₀, R₀, m and k are GONE from this registry with the parameters themselves.
    # Their calibration evidence is not lost — it is the derivation of ρ, and it lives in the
    # coverage source menu (model._COVERAGE_SOURCES) and Notes/calibration/. A registry entry
    # for a parameter the model no longer has would be documentation of nothing.
    #
    # D-080: the coverage dial — an APP-side dimension, because it is dialled in PERCENT while
    # its Params field `rho` is the fraction (state.APP_RANGES holds its envelope).
    'cov0': ParamMeta(interp="**$\\rho$ — coverage at $t = 0$, percent.** Earnings ÷ "
                             "model-building cost today. $\\rho_t = E_t/B_t$ is the model's "
                             "REPORTED financial outcome, and $\\rho$ is the **only finance "
                             "parameter there is**: both legs — earnings and "
                             "model-building cost — are normalised at "
                             "$t = 0$, so $E_0 = \\rho$ and $B_0 = 1$ identically and the whole "
                             "block is in multiples of today's build bill. Break-even is "
                             "**100%**: $\\Pi_t > 0 \\iff \\rho_t > 1$. It is CALIBRATED from "
                             "reported dollars as $m/k$ — margin before model-building over "
                             "model-building share of revenue — which is why the source menu "
                             "is still stated in dollars even though the model is not. "
                             "Default **33.5%** = \\$25.1B ÷ \\$75B, both legs dated mid-2026. The "
                             "rule that fixes it: every leg must estimate the **instantaneous "
                             "flow at the same anchor date**, and a calendar-year total already "
                             "*is* a mid-year flow reading, so nothing needs annualizing. The "
                             "retired 53.3% was not overruled but **re-dated** — it divided a "
                             "mid-2026 cost by a spiked May-2026 earnings run-rate, and on one "
                             "date the same evidence is 33.5%. Envelope [26, 46.3], both ends "
                             "read off a construction on the menu. Grade C.",
                      math_label='\\rho', uni_label='ρ', short_name='coverage at t = 0',
                      cal_target='labs earn ~33 cents per dollar of model-building spend today',
                      cal_alt='**Population, not basis:** striking Meta from both sides '
                              'gives 38.6%, striking Google 26.2% — worth ~5 and ~7 pp, and the '
                              'only discretionary spread left. Push the Google earnings leg to '
                              'the top of its judgment span at the same time and the '
                              'labs-favourable corner reads 46.3%, which is where the range '
                              'ends. An audited Anthropic '
                              'calendar-2026 figure above \\$26B would move the dial back up.',
                      grade='C'),
    # D-084 — the POSITION dials, one per use of the universal transition curve Γ. Pavel: "in the
    # transition function you should let the function be function of parameter s as well …
    # whenever S is used to define the transition another parameter representing s should be
    # introduced. For example, how far on the s-curve we already are." Each is stated in PERCENT
    # (his idiom: "we are in the bottom 10% of the s-curve"), and each resolves ITS OWN curve's
    # slope — sharing one dial across curves would tie three separate empirical claims together.
    'p0_c': ParamMeta(interp='**$q^c_0$ — how far into the compute slowdown we already are, in %.** The universal curve $\\Gamma$ needs a slope, and this is how the slowdown sets it: the transition is $q^c_0$% complete **today**, half-done at $t_{mid}$ and $(100-q^c_0)$% done at $2t_{mid}$. What it does NOT touch: today\'s compute growth, which is the dial $g_c$ at every position — moving this slider raises where the slowdown *started* ($g_c^{pre}$, above today), never what the rate is now. Default **1%** = the convention baked in before the dial existed (which is why every earlier path is reproduced exactly). Envelope [1, 25]% is a PROPOSAL flagged for the calibration round; 50% is impossible by construction (it would put the midpoint in the past). Scenario knob, grade F.', math_label='q^c_0', uni_label='q₀ᶜ', short_name='position on the slowdown curve', cal_target='how far into the compute slowdown we already are today — a scenario choice; the default assumes it has barely started', cal_alt='**Flagged:** no evidence pass yet — the envelope [1, 25]% is a proposal spanning "barely started" to "visibly under way" → calibration round.', grade='F'),
    'p0_w': ParamMeta(interp='**$q^w_0$ — how far into the value-slope easing we already are, in %.** The same construction as $q^c_0$, for the $\\nu \\to \\nu_\\infty$ transition in *capability*: the easing is $q^w_0$% complete at today\'s frontier, half-done at $x_{mid}$, $(100-q^w_0)$% done at $2x_{mid}$. $\\nu$ is literally today\'s slope $w\'(0)$ at every $q^w_0$ — it used to be the slope *before* the easing, so this slider silently moved a calibrated observable. A separate dial from $q^c_0$ on purpose: "how far into commoditization are we" and "how far into the compute slowdown are we" are different empirical questions. Envelope [1, 25]% PROPOSED and flagged → calibration round. Scenario knob, grade F.', math_label='q^w_0', uni_label='q₀ʷ', short_name='position on the value-easing curve', cal_target="how far into the value-slope easing we already are at today's frontier — a scenario choice; the default assumes it has barely started", cal_alt='**Flagged:** a 10% reading is the "commoditization is already biting" scenario; no evidence pass yet → calibration round.', grade='F'),
    'p0_F': ParamMeta(interp='**$q^F_0$ — how far into the *fringe\'s* compute slowdown we already are, in %.** The follower\'s curve carries its own position, exactly as it carries its own plateau, floor and midpoint — so the fringe\'s slowdown timing is never silently tied to the leader\'s. Same reading: $q^F_0$% today, half-done at $t^F_{mid}$, $(100-q^F_0)$% at $2t^F_{mid}$. SCENARIO KNOB, grade F, alongside $g_c^F$/$g_{c\\infty}^F$/$t^F_{mid}$ (Q-5 ruling) — the follower-compute calibration pass covers all four.', math_label='q^F_0', uni_label='q₀ᶠ', short_name='position on the fringe slowdown curve', cal_target='scenario dial — follower-compute calibration pass to come', grade='F'),
    't_compute_x': ParamMeta(uni_label='compute ×/yr', short_name='compute scaling today'),
    't_eff_x': ParamMeta(uni_label='eff. compute ×/yr', short_name='effective-compute growth today'),
    't_lag_mo': ParamMeta(uni_label='lag (mo)', short_name='fringe lag'),
    't_price_x': ParamMeta(uni_label='price-perf ×/yr', short_name='compute price-performance'),
    't_value_growth': ParamMeta(uni_label='value growth ×/yr', short_name='value growth today'),
    't_value_growth_inf': ParamMeta(uni_label='asympt. value growth ×/yr',
                                    short_name='long-run value growth'),
    't_floor_x': ParamMeta(uni_label='floor ×/yr', short_name='long-run compute floor'),
}

# ---- sidebar row-title tooltips (D-078 follow-up, Pavel: "just a short context info … the
# details should be in the calibration when one clicks on »"). One block, not inline in the
# registry, so each line stays visibly short. What it is + units — no rationale, no defaults.
_SHORT_TIPS = {
    'cov0': "Earnings ÷ model-building cost today (%) — break-even at 100%.",
    'gamma': "How fast the AI-R&D speedup grows with capability — recursive self-improvement. 0 = off.",
    'beta0': "AI's share of frontier-lab research throughput today — an input level, not a speedup.",
    't_mid': "The year by which half the compute slowdown has played out.",
    'p0_c': "How much of the compute slowdown has already happened, today (%).",
    'x_mid': "Where the value-slope transition is half-done (OOM above the 2026 frontier).",
    'p0_w': "How much of the value-slope easing has already happened, today (%).",
    'p0_F': "How much of the fringe's compute slowdown has already happened, today (%).",
    'g_a_F': "The follower's own algorithmic progress, as a share of the leader's rate.",
    'g_CF0': "The fringe's own compute growth today (OOM/yr).",
    'g_CF_inf': "The fringe's long-run compute-growth floor (OOM/yr).",
    't_mid_F': "The year by which half the follower's compute slowdown has played out.",
    'split': "Share of the initial gap that is algorithmic rather than compute.",
    'tau': "How many months the leader withholds its newest model.",
    'eta': "How substitutable compute and researchers are in R&D.",
    'alpha': "Share of algorithmic progress carried by experiment-compute growth.",
}
for _k, _v in _SHORT_TIPS.items():
    PARAMS[_k].short_tip = _v

# ---- derived views: the historical dict names, rebuilt from PARAMS. Each omits EXACTLY the keys
# whose field is None, preserving the original key presence the .get() fallbacks at call sites
# depend on. INTERP is used as BOTH the slider help= tooltip and the "what does this value mean?"
# popover; _MATH_LABEL/_UNI_LABEL/_SHORT_NAME are the three label maps; _CAL_TARGET/_CAL_ALT are
# the calibration captions; GRADES the grounding grades.
# ---------------------------------------------------------------- percent-valued dials (D-092)
# Pavel, on seeing the card read "p_0^w = 1": "It is not =1 but =1%, which is =0.01, currently it
# is confusing." He is right, and it is worse than cosmetic. p₀ is a FRACTION in the identities
# that define the transition curve, and p₀ = 1 is the DEGENERATE END of its domain — where the
# plateau diverges and the transition is already complete. A reader taking "= 1" literally sees
# the one value the parameter can never take, when the shipped default is 1% = 0.01, at the
# opposite end.
#
# ONE list, consulted by every display path, because the bug was that two call sites each
# hard-coded their own answer (both knew cov0 was a percent; neither knew the position dials
# were). The three position dials and the coverage ratio are the percent-valued quantities.
PCT_KEYS = {"cov0", "p0_c", "p0_w", "p0_F"}


def fmt_dial_value(key, val, places=None):
    """A dial's EFFECTIVE value as displayed, carrying its unit when it has one.

    Percent-valued dials get the sign, so "1" can never be read as the fraction 1. Everything
    else keeps %g, which is what the cards have always shown."""
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return str(val)
    if key in PCT_KEYS:
        return f"{val:.1f}%" if (places is None and key == "cov0") else f"{val:g}%"
    return f"{val:g}"


INTERP = {k: v.interp for k, v in PARAMS.items() if v.interp is not None}
_MATH_LABEL = {k: v.math_label for k, v in PARAMS.items() if v.math_label is not None}
_UNI_LABEL = {k: v.uni_label for k, v in PARAMS.items() if v.uni_label is not None}
_SHORT_NAME = {k: v.short_name for k, v in PARAMS.items() if v.short_name is not None}
_CAL_TARGET = {k: v.cal_target for k, v in PARAMS.items() if v.cal_target is not None}
_CAL_ALT = {k: v.cal_alt for k, v in PARAMS.items() if v.cal_alt is not None}
GRADES = {k: v.grade for k, v in PARAMS.items() if v.grade is not None}
SHORT_TIP = {k: v.short_tip for k, v in PARAMS.items() if v.short_tip is not None}

# ---- target sliders (D-037): the control IS the observable; the implied parameter renders as a
#      live caption underneath. One source of truth: bounds/defaults/MC all come from the
#      notebook's TARGET_RANGES / target_defaults.
INTERP_T = {
    "loss_half_gC": "**If experiment-compute growth halved, how much would algorithmic progress "
                    "slow? (%)** This is $\\alpha$, asked as a question a lab insider could "
                    "actually answer. Because both research channels are normalised to 1 today, "
                    "$\\alpha$ *is* the elasticity of algorithmic progress to compute growth — at "
                    "the base $\\eta = 0$ the answer is exactly $1 - 2^{-\\alpha}$, so **38.44% ⇒ "
                    "$\\alpha = 0.70$** (range 27–47% ⇒ 0.45–0.92, grade C+). The evidence splits "
                    "on *which* share this is: the **level** reading (compute's share of R&D "
                    "spend, Epoch's $\\epsilon_K \\approx 0.67$) and the **growth** reading "
                    "(compute's share of research-effort *growth*, ≈0.84, which is what this "
                    "model's $g_c(t)/g_c(0)$ literally says). The default sits between them — "
                    "the » menu lets you pick a side. Dialling the drag rather than the weight "
                    "is deliberate: hold the drag fixed and a more complementary $\\eta$ implies "
                    "a *lower* $\\alpha$, so the bottleneck evidence is counted once, not twice.",
    "t_compute_x": "**Compute scaling today (×/yr).** How fast the training compute behind the "
                   "**most capable** model grows — **3.24×/yr**, where two independent routes "
                   "meet: the dollar identity (bill 2.4× × hardware price-performance 1.35×) and "
                   "Epoch's capability-frontier reading of 3–4×/yr (grade A). Note this is *not* "
                   "the largest-training-run series (4.2×/yr) — a different object, kept in the "
                   "menu and labelled. Sets $g_c = \\log_{10}(\\cdot)$.",
    "t_eff_x": "**Effective-compute growth today (×/yr).** The frontier's *total* speed — physical "
               "compute **times everything else**: architecture, data, and post-training know-how. "
               "**11.34×/yr**, the only band both families of evidence can occupy (pretraining-only "
               "measurements are lower bounds at 2.2–3.2×/yr algorithmic; index-derived readings "
               "are upper bounds at 5.9–9×, because they include capability bought at inference "
               "time). Algorithmic progress is its **residual**: $g_a = \\log_{10}(\\cdot) - "
               "g_c$, so 11.34 = 3.24 × 3.50 and nothing can be counted twice.",
    "t_lag_mo": "**Fringe lag (months).** How many months ago the leading closed labs first served "
                "a model as capable — on hard, long-horizon, real-economic tasks — as today's best "
                "model available at near-cost prices. The follower is the **competitive fringe**: "
                "open-weight models are the *measurement proxy*, and API-first competitively-priced "
                "models count from their API date. **7.0 months** on a strict rule (UK AISI "
                "autonomous ranges). *Public leaderboard headlines (~4 mo) use a lenient catch-up "
                "rule; the same data on the strict rule gives 6.* ONE fact, applied per level: it "
                "sets the initial gap $\\Delta_0$ = lag × leader speed AND the catch-up rate(s) "
                "that keep the lag constant — the merged $\\delta = 12/\\text{lag}$ at the "
                "pure-catch-up levels, the two channels $\\delta_{dev}, \\delta_{rel}$ once the "
                "follower has its own engine (Level 3). Moving the speed dials changes what the "
                "same lag means in OOMs — the caption updates live.",
    "t_price_x": "**Compute price-performance (×/yr).** How much more training compute a dollar "
                 "buys each year — **1.38×/yr**, the measured hardware trend (Hobbhahn, Heim & "
                 "Aydos 2023: 1.39×/yr, CI [1.27, 1.54]), trusted directly rather than fitted as a "
                 "residual. Sets $g_p = \\log_{10}(\\cdot)$; the training **bill** growth "
                 "$10^{g_c-g_p}$ = 2.35×/yr is then a read-out (observed: 2.4×/yr).",
    "t_value_growth": "**Value growth today (×/yr).** How fast AI-attributable value is growing "
                      "right now — **×2.19/yr**, i.e. it a little more than doubles each year "
                      "(**119%/yr**, the same number stated as a percentage). "
                      "That round rate is the calibrated anchor; at the frontier's own speed it "
                      "is **≈×2.10 per OOM**. The frontier gains ×11.34/yr of effective "
                      "compute, and $\\nu\\,\\dot x^L$ turns the slope into a growth rate — read "
                      "backwards, ×2.19/yr fixes $\\nu$. The ×/OOM reading is "
                      "the pooled central value across three independent constructions "
                      "(Davidson value datum 1.86 · GATE ramp vs the wage bill 2.30 · revenue "
                      "decomposition 2.24) — an equally-weighted Monte Carlo over their input "
                      "bands, p50 = 2.08, not the median of those three numbers — with a "
                      "90% band [1.7, 2.65] ⇒ [×1.75, ×2.79]/yr here. The break-even pivot "
                      "is ≈⟪BE_VM⟫×/OOM, so the verdict turns on this dial more than on any "
                      "other. "
                      "Sets $\\nu = \\log_{10}(m)/\\dot x^L_0$ — move the speed dials "
                      "and the same rate implies a different slope, which is the point: the "
                      "**rate** is what is observed, the slope is what the model uses.",
    "t_value_growth_inf": "**Long-run value growth (×/yr).** How fast AI-attributable value "
                      "grows once the value slope has finished easing — far past $x_{mid}$ each "
                      "further OOM is worth ×$10^{\\nu_\\infty}$ more, with no hard ceiling, so "
                      "value keeps growing at that floor slope. Default **×1.10/yr** (10%/yr): "
                      "the floor of "
                      "Amodei's own 10–20% forecast, granted on purpose — the adversarial "
                      "reading, where the leader's bill has to be covered on the optimist's own "
                      "premise. Envelope **[×1.00, ×1.30]/yr** — no growth to 30%/yr — spans the "
                      "published worldviews, from "
                      "stagnation to the explosive-growth threshold; the » menu carries them as "
                      "clickable rows (Jones's 2% trend · mainstream-plus-AI 2.5–3.5% · "
                      "Korinek–Suh's 18% · Davidson's 30%). Sets $\\nu_\\infty = "
                      "\\log_{10}(m)/\\dot x^L_T$, at the frontier speed the **horizon** reaches "
                      "(0.409 OOM/yr at the defaults, down from 1.055 today) — past the horizon "
                      "the $\\psi$ feedback diverges, so there is no true asymptote to read.",
    "t_floor_x": "**Long-run compute floor (×/yr).** Compute scaling once power/fab/capital limits "
                 "bind (~2030). TWO LEGS, multiplied: what a dollar buys (hardware "
                 "price-performance, **≈1.38×/yr** measured) × how fast the real training budget "
                 "grows (**≈+2.3 %/yr**, long-run economic growth). Together **≈1.41×/yr** — "
                 "grade C, because the pairing is extrapolated past 2030 and no series measures "
                 "frontier compute growth there. The model has exactly one price of compute, so "
                 "wherever you set this dial it names an implied long-run budget growth: "
                 "**×⟪FLOOR_BILL_X⟫/yr** at the current setting. Sets "
                 "$g_{c\\infty} = \\log_{10}(\\cdot)$.",
}

# The targets' row-title tooltips (D-078 follow-up) — same contract as SHORT_TIP: what it is +
# units in ≤~90 chars; the full story stays in INTERP_T (calibration panel) and the » sources.
SHORT_TIP_T = {
    "t_compute_x": "Growth of the training compute behind the most capable model (×/yr).",
    "t_eff_x": "Total frontier speed: physical compute × algorithms, data, know-how (×/yr).",
    "t_lag_mo": "How many months the competitive fringe trails the frontier today.",
    "t_price_x": "How much more training compute a dollar buys each year (×/yr).",
    "t_value_growth": "How fast AI-attributable value is growing today (×/yr).",
    "t_value_growth_inf": "How fast that value grows once the slope has finished easing (×/yr).",
    "t_floor_x": "Long-run compute scaling: hardware price-performance × budget growth (×/yr).",
    "loss_half_gC": "If experiment-compute growth halved, how much slower algorithms get (%).",
}

# label, slider step, display format for each target slider (bounds come from TARGET_RANGES).
# Labels stay ONE line in the ~200px compact-row label cell (QA S7) — the "today"/long-form
# wording lives in INTERP_T (the hover help) and the calibration panel.
TSPEC = {
    "t_compute_x": ("Compute scaling (×/yr)", 0.01, "%.2f"),
    "t_eff_x":     ("Effective compute (×/yr)", 0.01, "%.2f"),
    "t_lag_mo":    ("Fringe lag (months)", 0.05, "%.1f"),
    "t_price_x":   ("Price-performance (×/yr)", 0.01, "%.2f"),
    # D-120: both value dials are growth rates. D-118 rider (Pavel, 2026-08-02, "just round it
    # to integer"): both are dialled on a WHOLE-PERCENT grid. D-133 keys them in ×/yr like every
    # other rate dial here, and the grid carries across exactly — 1 %/yr IS 0.01 ×/yr, so the
    # step and format become the 0.01 / "%.2f" the four ×/yr dials above already use, the anchors
    # (×2.19, ×1.10) and the envelope ends land on it exactly, and one click still moves ν by only
    # ≈0.004–0.007 value-OOM per OOM. The %/yr reading rides on the calibration cards.
    "t_value_growth": ("Value growth today (×/yr)", 0.01, "%.2f"),
    "t_value_growth_inf": ("Long-run value growth (×/yr)", 0.01, "%.2f"),
    "t_floor_x":   ("Compute floor (×/yr)", 0.01, "%.2f"),
    # D-098. A PERCENT, hence no t_ prefix (the t_…_x / t_…_mo convention names a multiplier).
    "loss_half_gC": ("Progress lost if compute growth halves (%)", 0.5, "%.1f"),
}

# Short per-level intros (Pavel, round 2: the Introduction tab is gone — "each level can have
# short introduction just not to complicate it. It should be minimal."). ONE short paragraph
# above the equations, distilled from the retired LEVEL_CARDS copy (git history has the full
# cards); ⟪TOKEN⟫s stay live.
LEVEL_INTRO = {
    1: "A **leader** (the frontier lab(s)) pushes the capability frontier at constant speed; "
       "the **competitive fringe** trails and catches up at rate $\\delta$ — the observed "
       "~7-month lag pins $\\delta$, so the gap holds at $\\Delta_0$. The leader earns the "
       "**rent on its lead** and pays this year's model-building bill; the one financial "
       "outcome is the **coverage** $\\rho_t = E_t/B_t$ (now ⟪COV0⟫%, break-even 100%). "
       "**This explorer is layered:** raise the level (top bar) to add the next block of "
       "mechanisms.",
    # D-081 merged Level 2 — Pavel's two-opposing-forces story. (It used to carry a live ℓ jump
    # number in the compute-slowdown clause; D-127 removed ℓ from the model, and with it the
    # ⟪JUMP⟫/⟪ELL⟫ tokens and their producers in _live_vals.)
    2: "**Dynamics — two opposing forces.** Level 1 held every growth rate constant; now the "
       "dynamics arrive. **(1) Compute growth slows down:** ~⟪GC_X⟫×/yr scaling "
       "cannot persist — it rides a transition curve down to the floor $g_{c\\infty}$, "
       "half-done at $t_{mid}$, with $q^c_0$ of it already behind us. **(2) Algorithmic "
       "progress speeds up:** AI accelerates its *own* R&D (the $\\psi$ feedback, strength "
       "$\\gamma$). The **net effect can go either way** — the closing *speed race* block "
       "reads off which force wins. Value's slope also starts easing: $\\nu$ before the "
       "transition → the floor $\\nu_\\infty$, half-done at $x_{mid}$ (no hard "
       "ceiling), $q^w_0$ of it already done at today's frontier.",
    3: "**Catch-up channels.** The follower gets its own engine (its own compute path and algo "
       "rate $g_a^F$), and the single $\\delta$ unpacks into ambient diffusion $\\delta_{dev}$ "
       "and released-model distillation $\\delta_{rel}$ — the channel a release delay could "
       "throttle.",
    # (entries 4-6 — release delay / cost mechanism / extensions — are GONE with the retired
    # tail: Pavel's D-081 ladder amendment; their content is parked in the spec)
}

# D-096: NOTATION_SECTIONS is RETIRED (Pavel: "I think that the 'Notation & conventions —
# grows with the level' section is not necessary, is there any information crucial for
# understanding that has not been delivered before?"). Audited section by section: the "who is
# who" paragraph repeated t_lag_mo's panel text almost verbatim, the Level-2 block was superseded
# by D-092's per-variable transition graphs plus each dial's own card, the Level-3 channel
# definitions live in the follower equation caption and the two channel cards, and D-093 had
# already rewritten the finance paragraph out from under it. TWO facts appeared nowhere else on
# the reachable path and were relocated before the delete, each to where its claim is made:
#   * the OOM unit convention and its origin -> the capability chart caption (ui/views.py), which
#     renders at EVERY level, since the retired expander was cumulative and this fact was not
#     Level-1-only; the Level-1 equations caption states it too, where units are first met;
#   * "flows are undiscounted, not an NPV" -> the Coverage subsection's profit note
#     (ui/equations.py). It was reachable only through the extensions-only `r` card and the MC
#     headline tooltip, so on the point-forecast path it was effectively unavailable — which is
#     the referee's first question about a profit path.

# ---- live numbers in static text: derived figures (jump factors, implied δ, ×/yr multipliers)
# quoted inside INTERP / LEVEL_INTRO prose must track the CURRENT effective parameters, never
# freeze at the defaults. Templates carry ⟪TOKEN⟫ placeholders (plain ⟪⟫ so LaTeX braces are
# untouched) substituted at render time from the same dict the simulation uses.
def _live_vals(d):
    """Current effective values for the ⟪TOKEN⟫ placeholders (fallbacks = the pinned defaults)."""
    gc = d.get("g_C0", P0.g_C0)
    ga = d.get("g_a", P0.g_a)
    Delta0 = d.get("Delta0", P0.Delta0)
    nu = d.get("nu", P0.nu)
    gp = d.get("g_p", P0.g_p)
    rho = d.get("rho", P0.rho)
    speed = gc + ga
    gaf = d.get("g_a_F") or P0.g_a_F        # below L3 the engine is pinned to 0 → hypothetical
    gcf = d.get("g_CF0") or P0.g_CF0
    split = d.get("split") or P0.split      # same hypothetical fallback below L3
    # The channel LENGTHS the model runs come from `stationary_catchup` — D-081's re-anchor rule,
    # which re-solves them against the exact t = 0 transfer identity. `channels_from_lag` supplies
    # only the DIRECTION inside it, and quoting it here made the merged-δ methodology assert
    # δ_eff ≈ 0.36 where the model runs 0.28 (audit A finding 4): 27% high, and contradicting the
    # ⟪WEDGE⟫ in the very same sentence, since δ_eff·Δ₀ must equal the wedge (0.2825 × 0.6152 =
    # 0.1738 ✓, 0.3600 × 0.6152 = 0.2215 ✗). Read at Level 1-2 the sentence describes what
    # Level 3 does, so the three hypothetical follower values above are substituted back in —
    # below L3 the engine is pinned off and `d` carries zeros for them.
    ddev, drel = m.stationary_catchup(
        m.Params(**{**d, "g_a_F": gaf, "g_CF0": gcf, "split": split}), merged=False)
    return {
        "GC_X": f"{10.0 ** gc:.1f}", "DELTA": f"{speed / Delta0:.2f}",
        "WEDGE": f"{max(speed - (gaf + gcf), 0.0):.2f}",
        # D-034: the effective single rate is δ_rel + split·δ_dev — the LIVE split, not the
        # 0.5 the doc used to hardcode (audit X-23)
        "DDEV": f"{ddev:.2f}", "DREL": f"{drel:.2f}", "DEFF": f"{drel + split * ddev:.2f}",
        # D-093: the four dollar tokens (REV0, COST0, RATIO, K_PCT) went with the dollars —
        # none had a consumer left once the finance prose stopped quoting \$40B and \$75B.
        "COV0": f"{100.0 * rho:.0f}",
        "REV_X": f"{10.0 ** (nu * speed):.2f}", "COST_X": f"{10.0 ** (gc - gp):.2f}",
        # The break-even value multiplier — the ν that balances 10^{ν(g_c+g_a)} against
        # 10^{g_c-g_p}, i.e. cost_x^(1/speed). It is the SAME quantity the Level-1 race card
        # prints (ui/equations.py `be_vm`), so it must be computed here rather than quoted:
        # the prose carried a frozen 1.64 from a retired calibration while the card rendered
        # 2.25, and the two screens gave opposite verdicts on the headline question.
        "BE_VM": f"{10.0 ** ((gc - gp) / speed):.2f}" if speed > 0 else "—",
        "GEFF_X": f"{10.0 ** speed:.2f}", "GA_X": f"{10.0 ** ga:.2f}",
        # D-129: the floor's own read-out. The compute-growth floor is calibrated AS two legs —
        # hardware price-performance plus real budget growth — and since the model has exactly
        # one price series, whatever the dial is set to NAMES an implied long-run budget growth.
        # Surfacing it beside the dial is what makes the two-leg construct legible: the shipped
        # floor reads ×1.02/yr (+2.3 %/yr real), and the retired 0.13 read ×0.98 — a budget
        # shrinking forever, which was the defect the recalibration fixed.
        "FLOOR_BILL_X": f"{10.0 ** (d.get('g_C_inf', P0.g_C_inf) - gp):.2f}",
        # D-132 (XM5): the value block's one internal cross-check, on the card that owns it.
        # Read as a bounded market, (x_mid, p0_w) OVER-DETERMINE the total value multiple, and
        # until now nothing noticed when they over-determined it inconsistently — the retired
        # (10.0, 1%) pair implied 1853× against p0_w's own 100×. See model_profit.value_coherence
        # for why this is a warning rather than an error: the identity binds only at nu_inf = 0,
        # and a user running the Amodei steelman is not doing anything wrong by tripping it.
        **_coherence_tokens(d),
    }


def _coherence_tokens(d):
    """The XM5 read-out tokens. Split out because the identity is the model's, not the card's —
    it lives in `model_profit` and is exercised by tests there; this only formats it."""
    p = m.Params(**d)
    implied, asserted, gap, ok = m.value_coherence(p)
    return {
        "VAL_M_IMPLIED": f"{implied:,.0f}" if implied >= 10.0 else f"{implied:.1f}",
        "VAL_M_ASSERTED": f"{asserted:,.0f}",
        "VAL_M_COHERENT": f"{m.coherent_x_mid(p):.2f}",
        # The one-line warning itself: empty when the pair is coherent, so the sentence simply
        # is not there rather than reading "no problem" at the reader.
        "VAL_M_WARN": "" if ok else (
            f" ⚠ These two readings are {abs(gap):.1f} OOM apart — the pair is incoherent "
            f"under the bounded-market structure."),
    }


def _sub_live(txt, d):
    """Substitute ⟪TOKEN⟫ placeholders with current values (no-op for token-free text)."""
    if txt and "⟪" in txt:
        for k, v in _live_vals(d).items():
            txt = txt.replace("⟪" + k + "⟫", v)
    return txt


def lag_note(level):
    """How strongly the widget may claim the fringe lag holds — the ONE place that decides it.
    The caller supplies the subject ("the lag", "the ~7-month fringe lag"); this ends the phrase.

    X-04's ruling: the stationary construction holds the lag constant for ALL t only under Level
    1's steady growth. From Level 2 the two slowdowns bend the paths and stationarity is
    guaranteed at t = 0 ONLY — measured at the L2 defaults, Δ₀ = 0.6152 falls to Δ(10) = 0.2728,
    so the gap closes 56% over the horizon and "stays constant" is simply false there.

    Why a helper for six words (audit A finding 3): X-04 was implemented at the sidebar caption
    where it was reported, and the merged-δ card and its » panel header went on asserting "stays
    constant" at Level 2 — one screen away from the slider caption saying the opposite about the
    same number. Three sites, one claim; the claim lives here now.
    """
    return "stays constant" if level == 1 else "is stationary today"

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
                     "$\\Delta_0$ for any lag while compute growth is constant. At Level 2 the "
                     "compute slowdown decelerates the leader and the gap starts closing; from "
                     "Level 3 the follower's own engine covers most of its speed and the two "
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
