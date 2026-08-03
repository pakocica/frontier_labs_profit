"""Calibration source menus — the declarative evidence behind every dial (D-042, D-110).

`CAL_SOURCES` is the documented per-parameter source table that drives the widget's calibration
panel. It lived inside `model.py` until D-110 retired the notebook affordance and made
port-readiness a structural criterion: calibration evidence is *data*, and data belongs in a
module that holds nothing else, so the eventual JS/TS port can serialize it rather than re-derive
it.

**This module imports nothing** — not even `model`. That is the property worth protecting, and it
is why the one row whose value is a LIVE model default (γ) is filled in by `bind_live_defaults()`
at import time rather than reaching back into `model.Params`. `model.py` imports this module and
re-exports every public name, so `import model as m` still reaches the menus as `m.CAL_SOURCES`
and no consumer knows the difference.
"""

# CAL_SOURCES is the documented per-parameter source table. It drives the widget's calibration
# panel (source rows with [choose] / [choose range]). Keyed by PARAMETER name; `value` is in the
# TARGET's natural units where a target drives the parameter, in parameter units for free dials.
# Faithful to Notes/calibration/param_docs/ (the briefs Pavel ratified) and to the evidence
# register Notes/calibration/evidence/ -- no invented sources.
#
# Row fields: source · value · unit · note · grade (A solid measurement · B reasonable anchor ·
# C judgment / weakly identified · F free choice) · ci (the source's own documented interval) ·
# ci_default=False (a judgment band, offered for [choose range] but excluded from the default
# span) · group (menu subheading -- rows measuring DIFFERENT OBJECTS are grouped apart) ·
# display_only=True (shown for context, never clickable: a different object, a bound, or a
# retired reading) · triple (RETIRED with the money rows, D-093 -- name reserved, never reuse) ·
# disp (display string replacing "value unit", for a row whose machine value is an exact fraction
# that must not be rounded on screen) · adopt (what to WRITE into the destination control when
# that differs from `value` -- the eta menu's rows report a number so the rail can place them, but
# the control is a selectbox and takes its option LABEL) · basis (money and coverage rows: the ACCOUNTING BASIS --
# calendar / run-rate / mixed -- mandatory disclosure, see the coverage block) · why
# (display_only rows: the short reason THIS row cannot be adopted, replacing the generic caption).

# ---- the fringe lag (brief 05). ONE list, FOUR keys (Delta0, delta_dev, delta_rel,
# delta_total), so every edit here propagates to all four at once.
# LANGUAGE: the follower is the COMPETITIVE FRINGE. Open-weight models are its measurement proxy;
# API-first competitively-priced models count as fringe from their API date (Pavel, 2026-07-26).
#
# D-109 SWEEP: 8 rows -> 5, ordered by how well each matches the model's object rather than by
# value (TL4's table is ascending by months, which is a reading order, not a ranking).
# Dropped (3), all archived with their full series:
#   * UK AISI narrow-task 5.5, ci (4, 7) — grade A, but the GAMEABLE sub-score of a source that is
#     already on the menu through its construct-matched one. The brief rules on this directly:
#     "within AISI, prefer the autonomous-ranges sub-score (~7 mo) over the narrow-task one",
#     and packet §4 files it under "(floor, gameable)". Its floor of 4 is jointly witnessed by
#     the Epoch lenient row, which survives. `capgap-004`; pairings at `lagts-017`.
#   * FrontierMath tiers 7.35, ci (6.6, 8.1) — grade C, only four open models evaluated, and
#     packet §4.2 flags an upward coverage bias. `lagts-025`.
#   * Local ECI daily-grid refit 7.65 — duplicates row 2's instrument AND window, and its +1.7 mo
#     offset against Epoch-strict has never been explained. `lagts-003`, discussed in packet §6.
#
# ENVELOPE UNCHANGED at [4, 12] months over a union of 4.0 … 10.0. The ceiling is documentary
# (TL5(b) stretches METR's 10) and it CANNOT be hugged to the menu the way g_C0's was: the
# ratified Monte-Carlo draw for t_lag_mo is lognormal_from_ci(4, 12), so an envelope narrowed to
# 10 would put the sampling default's own upper tail outside the envelope that bounds it.
_LAG_SOURCES = [
    dict(source="UK AISI cyber, autonomous ranges", value=7.0, unit="mo", grade="A",
         note="private agentic, long-horizon — construct-matched to the model's object, and "
              "the reading the calibrated 7.0-month default is set to"),
    dict(source="Epoch ECI, strict rule, same window", value=6.0, unit="mo", grade="A",
         note="the model's own definition: capability the fringe can actually match. The largest "
              "sample of any row here"),
    dict(source="Fringe-consistent 2026 reading", value=6.5, unit="mo", grade="B",
         note="counts API-first Chinese models (Kimi K3 etc.) as fringe from their API date — "
              "the follower IS the competitive fringe, so this is the population the model means"),
    dict(source="METR / private composite", value=9.0, unit="mo", grade="B", ci=(8.0, 10.0),
         note="agentic time-horizon; blog-composited. The only reading in the upper half of the "
              "[4, 12] sampling range, which is what earns it a place over the closer readings"),
    dict(source="Epoch ECI, published headline (lenient rule)", value=4.0, unit="mo", grade="A",
         note="public composite; a FLOOR, and the wrong catch-up rule for this model — a named "
              "floor rather than an estimate: read against the strict-rule row above it, "
              "the pair shows a 2× spread produced by a RULE, not by a basis or a source"),
]

# ---- the money side. `_MONEY_SOURCES` -- the five (R0, m, k) lab triples -- was DELETED by
# D-093 along with the three parameters it documented. The evidence itself is not lost and was
# never in danger: every one of those rows is restated on the model's own margin definition in
# _COVERAGE_SOURCES below (the four per-lab rows plus the four A-D constructions), derived in
# Notes/calibration/cov0_source_menu_2026-07-27.md, and traced to
# Notes/calibration/fin4_restatement_example_2026-07-27.md and the evidence register. What went
# is the MENU, because the dials it fed no longer exist -- inputs to a calibration are not
# parameters of a model, and only the coverage ratio was ever identified.
#
# (The `triple` row field went with it. It is still documented in the field list above as a
# retired key, so a future money row cannot silently reuse the name for something else.)

# ---- the COVERAGE dial ρ₀ = m/k (D-080 follow-up, Pavel: "I want the calibration of coverage to
# let one choose based on different sources"). ONE ROW = ONE (before-model-building profit,
# model-building outlay) pair reduced to the only identified number, IN PERCENT -- R₀ cancels out
# of ρ₀ entirely. Every value is DERIVED, with the arithmetic on the row and in full in
# Notes/calibration/cov0_source_menu_2026-07-27.md, from the FIN4(b) restatement
# (Notes/calibration/fin4_restatement_example_2026-07-27.md) and through it from the register.
#
# D-104 RESTATED EVERY ROW ON A DATING RULE (FIN4 settled, Pavel 2026-07-29; full working in
# Notes/calibration/param_docs/12_FIN4_resolution.md §5). The rule, in four steps:
#   R1 DATE IT.   Every reported figure estimates the flow at some instant. A calendar-year-Y
#                 total is the flow at Y + ½ + g/24 -- mid-year even at 6×/yr growth -- so it needs
#                 no conversion; an H1 total ×2 is the flow at Y + 0.27; a run-rate "as of month M"
#                 is the flow at the middle of M; a multi-year average is an INTERIOR date past
#                 the window's midpoint, not any year one likes.
#   R2 CHECK REPRESENTATIVENESS. A run-rate on a spiked month (promotional disclosure, a lumpy
#                 contract, a training-run month) is not a trend flow at ANY date.
#   R3 MOVE IT to t = 0 = mid-2026 by the firm's own growth factor, stating the g used.
#   R4 PAIR ONLY SAME-DATED LEGS. A ratio of two differently-dated legs is not a coverage reading
#                 at any t.
# BASIS therefore stops being a CHOICE and becomes disclosure of the R1 route: under R1-R4 every
# admissible row lands on the same instant, and `mixed` is now a DIAGNOSIS -- the row failed R4 --
# rather than a category one could prefer. Consistency is worth ×1.4-1.5 on ρ₀; which common
# instant, only ±3.5% per half-year, ρ₀ being a ratio. That asymmetry is the whole content of FIN4.
#
# CHOOSABLE ⟺ the implied ρ₀ lands inside the vetted [26, 46]% envelope (state.APP_RANGES). Rows
# outside it -- the three cross-dated readings D-104 retires, plus every restatement whose m came
# out ≤ 0 -- are display_only and carry `why`. The envelope was deliberately NOT widened to admit
# any of them: the corner that would argue for a higher ceiling is the trust-the-spike pairing
# (52-63%), and costs-025/026 reject its R2 premise. Flags for Pavel are in §4 of the derivation
# file; the population question (Meta in or out) is still his, §D.2.
# ---- alpha group headings (D-098). The headings carry the CONSTRUCT, because the two readings
# answer different questions and a reader who cannot tell them apart cannot use the menu. The
# eta note is on the headings rather than buried in the log: a row labelled 0.67 that delivers
# 0.44 at another eta is exactly the two-literals trap this session hit repeatedly.
_ALPHA_ETA_NOTE = "values shown at η = 1; the delivered α moves with the substitution setting"
_ALPHA_DEFAULT = f"The shipped default ({_ALPHA_ETA_NOTE})"
_ALPHA_LEVEL = ("Level reading — compute's share of R&D SPEND "
                f"({_ALPHA_ETA_NOTE})")
_ALPHA_GROWTH = ("Growth reading — compute's share of research-effort GROWTH "
                 f"({_ALPHA_ETA_NOTE})")
_ALPHA_BOUND = "Bounds and counter-readings"
# (A fifth heading, "State of practice — not evidence", went with the D-109 sweep: its one row
# was Davidson's assumed 0.5, grade D and explicitly not a measurement. The point it made is
# worth keeping, so brief 10 §3 now carries it in prose -- "Where the state-of-practice reading
# lives now" -- and is its only home.)

# D-104: "basis × population" is gone as a framing. Once every leg is dated to mid-2026 the
# constructions differ only in POPULATION (who is charged, who is credited) and in whether the
# earnings leg's spiked month is trusted — which is a pass/fail on R2, not a second axis.
#
# D-109 (Pavel, 2026-08-02, reviewing this menu live): "keep it clean and include only 3-5 values
# that are most defendable in declining order of relevance. (This is general rule.)" This menu had
# grown to 11 rows, of which only three were both current and inside the envelope; the other eight
# were the retired D-076 default, the superseded FIN4(b) central, the two cross-dated constructions
# A/B, and four per-lab restatements running from +100% to -35% — every one drawn as an
# out-of-envelope chevron or a dead marker, i.e. cards whose rails communicate nothing. All eight
# survive in full in the FIN4 derivation (Notes/calibration/param_docs/12_FIN4_resolution.md, which
# restates every row with its arithmetic) and in the evidence register; the menu is not the
# archive. Kept, in declining order of relevance: C (the default), D (the Meta-population
# alternative, §D.2 open), E (the floor of the envelope).
_COVERAGE_SOURCES = [
    dict(source="Industry 2026, broad population — the widget default",
         value=100.0 * 25.1 / 75.0,
         disp="ρ₀ = 33.5%", unit="%", grade="C", basis="calendar",
         note="\\$25.1B ÷ \\$75B = 33.5% (k/m 2.99). Anthropic's de-spiked calendar-2026 revenue "
              "replaces the \\$47B run-rate, so BOTH legs are calendar-2026 totals — flows at "
              "the SAME date, which is what makes the ratio a coverage reading at all. A "
              "calendar-year total needs no conversion because it is, to within 3% at these "
              "growth rates, the mid-year instantaneous flow, which is why no cost was ever "
              "annualized. This row carries the exact 25.1/75, so choosing it sets the dial to "
              "the exact calibrated default"),
    dict(source="Industry 2026, Meta struck from both sides",
         value=100.0 * 25.1 / 65.0, disp="ρ₀ = 38.6%", unit="%", grade="C", basis="calendar",
         note="\\$25.1B ÷ \\$65B = 38.6% (k/m 2.59). Same date on both legs as the broad-population "
              "row, differing only in POPULATION — whether a lab that builds frontier models "
              "largely for its own use is counted on both sides or on neither. That question is "
              "still open, and it is worth ~5 pp either way"),
    dict(source="Industry 2026, Google struck from both sides",
         value=100.0 * 15.05 / 57.5, disp="ρ₀ = 26.2%", unit="%", grade="C", basis="calendar",
         note="\\$15.05B ÷ \\$57.5B = 26.2%. Google's +\\$10B is the one unsourced and undatable "
              "number on the earnings side, so striking it from the earnings AND its spend from "
              "the cost base is the honest floor rather than a curiosity — it is the FLOOR of "
              "the [26, 46] envelope. Dated to a common instant like the rows above it; read as "
              "a run-rate instead, the same construction gave 50.4%, which is the size of the "
              "error that dating removes"),
]

CAL_SOURCES = {
    # ---------------------------------------------------------------- brief 01: g_C0 (TC1b/TC5)
    # D-109 SWEEP: 7 rows -> 6, and this menu is the one the rule had to bend around. Pavel's TC1
    # answer says of the compute-frontier rows "do not remove them", and his D-109 amendment
    # ("3-5 is just a rough suggestion ... it can be more if there are more reliable sources")
    # settles the conflict in TC1's favour: all four are grade A, so all four stay and the menu
    # keeps six rows. The ONE removal is Epoch's 2025-09 lead-time row (5.0, display_only) — a
    # MECHANISM row belonging to the compute slowdown rather than a competing estimate of today,
    # whose reassignment to brief 04 (ξ) was ordered by ANSWERS.md TC5 and
    # IMPLEMENTATION_base_sync.md §5 and had simply never been executed. Archived at compute-005.
    #
    # ORDER is now defendability, not value: the calibrated spot first, then the capability-
    # frontier measurement it agrees with, then the different-object compute-frontier group in
    # its D-076 order (that grouping survives D-109 — a menu that mixes objects teaches the wrong
    # thing, so the objects stay labelled and apart).
    "g_C0": [
        dict(source="Implied by the bill ÷ hardware price-performance", value=3.24, unit="×/yr",
             grade="B", group="Capability frontier — the model's object",
             note="THE CALIBRATED SPOT. Dollar identity 2.4 (bill growth) × 1.35 (price-perf.) — "
                  "an independent route that lands inside the capability-frontier band"),
        dict(source="Epoch 2025-09, “GPT-5 used less compute”", value=3.5, unit="×/yr", grade="A",
             ci=(3.0, 4.0), group="Capability frontier — the model's object",
             note="frontier-BY-CAPABILITY, 2023–25: the growth rate of the compute behind the "
                  "most capable model. The only grade-A measurement of the model's actual "
                  "object, and the source of the [3, 4] Monte-Carlo sampling band"),
        dict(source="Sevilla & Roldán 2024 (Epoch), frontier post-2018", value=4.2, unit="×/yr",
             grade="A", ci=(3.6, 4.9), group="Compute frontier — largest run (different object)",
             note="the headline series; tracks the largest TRAINING RUN, not the most capable model"),
        dict(source="Same report, full 2010–24 window", value=5.3, unit="×/yr", grade="A",
             group="Compute frontier — largest run (different object)",
             note="includes the pre-2018 catch-up transient. The largest value the menu "
                  "witnesses, and therefore the envelope's right edge"),
        dict(source="Epoch 2026-01, global AI chip capacity", value=3.3, unit="×/yr", grade="A",
             ci=(2.7, 4.1), group="Compute frontier — largest run (different object)",
             note="installed base, supply side"),
        dict(source="Pilz et al. 2025, AI supercomputers", value=2.5, unit="×/yr", grade="A",
             group="Compute frontier — largest run (different object)",
             note="deployed FLOP/s of clusters, not single-run compute. The smallest value the "
                  "menu witnesses, and therefore the envelope's left edge"),
    ],
    # ------------------------------------------------- brief 07: effective-compute growth (GE1c)
    # The dial is t_eff_x; g_a = log10(t_eff_x) − g_C0 is the residual. Values below are t_eff in
    # ×/yr = 3.24 × the source's algorithmic rate, so the menu is directly comparable.
    # D-109 SWEEP: 7 rows -> 5. Both drops were already display_only, so no adoptable row and no
    # part of the union goes with them: "frontier of algorithmic quality" 29.2 (a second point of
    # a paper already on this menu twice, taken at a DIFFERENT frontier definition — geff-004,
    # packet 07 §GE3 row 6) and Ho 2026's whole-stack 32.4 (explicitly the wrong object: it counts
    # inference-cost efficiency, which this model prices separately through g_p, so admitting it
    # would double-count — algo-005, §GE3 row 7). The "Wrong object" heading empties with the
    # second of them.
    #
    # ORDERING NOTE, because it is not the obvious one: rows are ordered BY GROUP, not row by row.
    # calpanel prints a group heading whenever the group changes, so a strict defendability order
    # across rows would split "Lower bound" in two and print its heading twice — D-076's grouping
    # and D-109's ordering have to be satisfied together, and the way to do that is to rank the
    # GROUPS and keep each one contiguous. Default group first; then the pretraining-only lower
    # bounds led by Ho, the flagship source whose own CI GE4 uses to define the sampling envelope;
    # then the upper bound. Within the lower bounds, Ho (grade A, envelope-defining), then
    # Gundlach (grade A, GE3's mandated lower-bound label), then Mertens (grade B, and the row
    # the brief itself names as the next to go if this menu is ever cut to four).
    #
    # ENVELOPE UNCHANGED at [5, 21] ×/yr: the union is 5.8 … 20.6, both ends from Ho's own CI.
    "g_a": [
        dict(source="Epoch “Rosetta Stone” §3.2.2, test-time-deflated", value=11.34, unit="×/yr",
             grade="B", ci=(10.7, 14.3), group="Bias-corrected reading — the calibrated default",
             note="THE CALIBRATED SPOT (11.34 = 3.24 × 3.50). Epoch's delivered-capability "
                  "estimate 5.86×/yr algo, deflated by the DIRECTLY MEASURED test-time share "
                  "(Qwen3 Instruct→Thinking, same weights same date, +5.81 ECI pts): capability "
                  "bought at inference time is in neither a nor c. Its parent, undeflated, is the "
                  "19.0 row at the bottom of this menu"),
        dict(source="Ho et al. 2024 — pretraining efficiency (2.69×/yr algo)", value=8.7,
             unit="×/yr", grade="A", ci=(5.8, 20.6),
             group="Lower bound — pretraining efficiency only",
             note="the flagship source; its own 95% CI on the algorithmic rate is [1.79, 6.35]×/yr "
                  "— and that CI, shifted by the fixed g_C0, IS this dial's sampling envelope"),
        dict(source="Gundlach et al. 2025 — frontier ablation (2.23×/yr algo)", value=7.2,
             unit="×/yr", grade="A", group="Lower bound — pretraining efficiency only",
             note="excludes post-training know-how, which this model counts in the algorithmic term a"),
        dict(source="Mertens et al. — developer-effects design (3.2×/yr algo)", value=10.4,
             unit="×/yr", grade="B", group="Lower bound — pretraining efficiency only",
             note="cross-developer panel; pretraining basis"),
        dict(source="Epoch “Rosetta Stone” §3.2.2, as published (5.86×/yr algo)", value=19.0,
             unit="×/yr", grade="A", group="Upper bound — test-time compute included",
             note="delivered-capability basis: contains capability bought at inference time. "
                  "This row deflated IS the calibrated default, so it is the default's own parent — "
                  "shown so the deflation can be read straight off the menu"),
    ],
    # ---------------------------------------------------------------- brief 05: the fringe lag
    "Delta0": _LAG_SOURCES, "delta_dev": _LAG_SOURCES, "delta_rel": _LAG_SOURCES,
    "delta_total": _LAG_SOURCES,
    # ---------------------------------------------------------------- brief 02R: nu (TV4'/TV5')
    # D-109 SWEEP: 9 rows -> 4, and this is the cheapest of the five sweeps — ALL FIVE drops were
    # already display_only, so the menu loses no adoptable row and its union does not move.
    # Dropped: the four single-channel benchmark slopes (RLI 16.1 valaut-033, SWE-bench 10.7
    # valaut-040, OSWorld 4.66 valaut-038, GDPval 3.11 valaut-026) and the legacy widget default
    # 1.73 (grade F, e^0.55 under a retired natural-log convention — archived in
    # Notes/calibration/param_nu.md §"Hard-case resolution" and named in brief 02R TV4').
    #
    # The benchmark rows carried a real teaching point and it is NOT lost with them: it moves into
    # the pooled row's own note below, where it now reads as an argument rather than as four
    # unclickable cards. That was the condition for removing them.
    #
    # ORDER: the pooled spot first, then C, B, A rather than the brief's alphabetical A/B/C —
    # C is the only construction restated on the ratified ruler, and A carries the chord-vs-slope
    # correction (valaut-047). TV5' lists them alphabetically; that is the record's fallback, not
    # a defendability ranking. ENVELOPE UNCHANGED: tri(1.5, 2.1, 3.0), union 1.62 … 2.84.
    "nu": [
        dict(source="Pooled — the three constructions together", value=2.1, unit="×/OOM",
             grade="B", ci=(1.7, 2.65), group="Aggregate constructions — the model's object",
             note="THE CALIBRATED SPOT. Median of three independent routes on the current ruler. "
                  "Why ~2 and not the 4–16×/OOM the single-channel benchmarks show: those measure "
                  "ONE fixed basket of tasks each, and a dollar-weighted mixture of channels "
                  "whose midpoints are spread over ±2–3 OOM aggregates to ~2×/OOM even when every "
                  "individual channel ramps at 16×. The benchmark slopes CORROBORATE this "
                  "aggregate, they never competed with it"),
        dict(source="Revenue decomposition", value=2.24, unit="×/OOM", grade="B",
             ci=(1.84, 2.72), group="Aggregate constructions — the model's object",
             note="observed revenue growth ÷ the frontier speed this calibration fixes; the only "
                  "route restated on the current ruler"),
        dict(source="GATE ramp + wage-bill ceiling", value=2.30, unit="×/OOM", grade="B",
             ci=(1.98, 2.84), group="Aggregate constructions — the model's object",
             note="automation accounting; OOM-native"),
        dict(source="Davidson value datum", value=1.86, unit="×/OOM", grade="B",
             ci=(1.62, 2.21), group="Aggregate constructions — the model's object",
             note="FLOP-gap arithmetic; OOM-native, and carrying the chord-vs-slope correction"),
    ],
    # ------------------------------------------------------------- brief 01 rider: g_p (TC2b)
    "g_p": [
        dict(source="Trusted hardware leg (the calibration)", value=1.38, unit="×/yr", grade="B",
             ci=(1.30, 1.45),
             note="THE CALIBRATED SPOT, g_p = 0.14 OOM/yr. Prices fall to 72% of the year before; "
                  "implied bill growth 10^(g_C0−g_p) = 2.35×/yr vs Cottier's observed 2.4×/yr"),
        dict(source="Hobbhahn, Heim & Aydos 2023 (Epoch) — ML hardware price-performance",
             value=1.39, unit="×/yr", grade="A", ci=(1.27, 1.54),
             note="FP32 regression; the measured leg the calibration trusts"),
        dict(source="Dollar-identity leg (bill 2.4 ÷ compute 3.24)", value=1.35, unit="×/yr",
             grade="B", note="the same identity read backwards — how the 3.24 spot was built"),
        # (The retired grade-F "bill-residual reading" row (1.75 ×/yr) was removed on Pavel's
        # review of the D-106 panel, 2026-08-02: its value sits outside the envelope, so the rail
        # could only draw it as an edge chevron — a card that shows nothing. The construction
        # itself is preserved in the evidence register: costs-028 documents the wedge between the
        # bill-inverted reading (0.243 OOM/yr) and the measured hardware leg (0.130 OOM/yr).)
    ],
    # ------------------------------------------------------------ brief 04/04b: the money side
    # D-093: the "k_build" / "R0" / "m_margin" entries are gone with their parameters, and
    # so are the two DERIVED read-out rows ("kappa", "B0") -- a menu documenting a number the
    # user can neither set nor see is dead weight. THE MONEY SIDE NOW HAS EXACTLY ONE MENU,
    # which is the point: rho is the one identified object, so it is the one thing with sources.
    "cov0": _COVERAGE_SOURCES,
    # ------------------------------------------------------------ EXTENSIONS — extensions round
    "g_C_inf": [
        dict(source="Hobbhahn 2023 — hardware-only trend", value=1.35, unit="×/yr", grade="C",
             note="floor LEVEL is our extrapolation; power binds ~2030 (Sevilla et al. 2024)"),
    ],
    "ell": [dict(source="Payment-weighted composite, 2026 vintage", value=0.45, unit="yr",
                 grade="B", ci=(0.25, 1.3),
                 note="run duration + finish→release gap, weighted by WHEN the dollars are paid; "
                      "RL dollars paid late shorten it. Enters with the compute slowdown — in "
                      "the base, constant compute growth makes it cancel identically")],
    "phi_RD": [dict(source="Folded into the bill anchor", value=0.0, unit="×", grade="B",
                    display_only=True,
                    note="the cost anchor is the OBSERVED bill — compute AND R&D / "
                         "researcher overhead together (they move roughly proportionally), so no "
                         "separate markup is calibrated in the base")],
    # D-084 position dials -- grade F scenario knobs, no evidence pass yet (flagged with the
    # envelopes for the calibration round). The two rows per dial are the honest bracket: the
    # inherited convention, and Pavel's own worked example of the alternative.
    "p0_c": [
        dict(source="Inherited convention", value=1.0, unit="%", grade="F",
             note="the slowdown has barely started at t = 0 -- what the widget assumed before "
                  "this dial existed, and keeping it as the default leaves those paths "
                  "unchanged"),
        dict(source="Already visibly under way", value=10.0, unit="%", grade="F",
             note="A worked example: we are in the bottom 10% of the S-curve and it flattens "
                  "(middle) in 3 years -- set t_mid = 3 alongside it"),
    ],
    "p0_w": [
        dict(source="Inherited convention", value=1.0, unit="%", grade="F",
             note="the value slope has barely started easing at today's frontier, so nu IS "
                  "essentially today's slope"),
        dict(source="Commoditization already biting", value=10.0, unit="%", grade="F",
             note="today's slope already a tenth of the way from nu down to nu_inf -- the "
                  "reading under which benchmark saturation is visible now"),
    ],
    "p0_F": [
        dict(source="Inherited convention", value=1.0, unit="%", grade="F",
             note="the fringe's own slowdown has barely started; independent of the leader's "
                  "p0_c by construction"),
        dict(source="Already visibly under way", value=10.0, unit="%", grade="F", note=""),
    ],
    "x_mid": [
        dict(source="Early-commoditization reference", value=2.0, unit="OOM", grade="C",
             note="the value-slope transition ν → ν_∞ is half-done 2 OOM out"),
        dict(source="Mid reference", value=5.0, unit="OOM", grade="C", note=""),
        dict(source="Harvest-continues reference", value=10.0, unit="OOM", grade="C",
             note="today's slope ν carries essentially across the horizon (midpoint 10 OOM out)"),
    ],
    "g_a_F": [
        dict(source="Scale-bias low, 0.6 × leader (Gundlach)", value=0.6, unit="× leader",
             grade="B", note=""),
        dict(source="Scale-bias central, 0.7 × leader (Gundlach)", value=0.7, unit="× leader",
             grade="B", note=""),
        dict(source="Scale-bias high, 0.8 × leader (Gundlach)", value=0.8, unit="× leader",
             grade="B", note="one source, three readings; the dial IS this share (g_a^F = share × g_a)"),
    ],
    # eta is the app's one CHOICE dimension, and until 2026-07-29 these two rows carried the
    # selectbox's option LABELS as their `value` -- strings, so nothing could be positioned
    # numerically and the mini rail had to be suppressed. Pavel ruled the rail back on ("I don't
    # see a problem with clicking on descrete point on a line. There won't be interval, MC uses
    # spot value for this parameter"), so `value` is now the NUMBER (which places the dot on the
    # discrete rail and lets test_source_values_lie_inside_their_envelope sweep these rows like
    # any other) and `adopt` carries the option label the selectbox needs written into it. `disp`
    # keeps the card head reading in the dial's own vocabulary rather than as a bare float.
    "eta": [
        dict(source="Whitfill & Wu 2025 — substitutes (σ = 2.58)", value=0.61,
             adopt="0.61", disp="η = 0.61", unit="", grade="B", note="N = 27, 4 labs"),
        dict(source="Whitfill & Wu 2025 — complements (scale control)", value=-2.0,
             adopt="-2 (complements)", disp="η = −2", unit="", grade="B",
             note="sign flips on the K_train control"),
    ],
    # D-091 re-based gamma from nats to base 10 (0.2 -> 0.2/ln10) and swept every quoted literal
    # it could find -- Params.gamma, PARAM_RANGES, the interp, the pane, the spec, N4 -- but not
    # this table, a THIRD home for parameter numbers. The row kept the retired nats value, which
    # sits past the right end of the base-10 envelope [0, 0.1737], so the mini rail drew it as an
    # out-of-envelope chevron and gamma was left with no adoptable source at all (audit A
    # finding 2, and what Pavel saw: no dot on the rail). `disp` keeps the card readable while
    # the machine value stays exactly the shipped default.
    "gamma": [dict(source="Tentative default (no observable yet)", value=None,
                   disp="0.0869 /OOM", unit="/OOM", grade="C",
                   note="0.2/ln 10 — the same tentative default expressed in decades per OOM")],
    "beta0": [dict(source="Tentative default (no observable yet)", value=0.3, unit="", grade="C",
                  note="")],
    "t_mid": [dict(source="Scenario default — sweep it", value=2.3, unit="yr", grade="F",
               note="≈ the earlier ξ = 0.3 half-decay time, ln 2/0.3 = 2.31")],
    "t_mid_F": [dict(source="Scenario default", value=2.3, unit="yr", grade="F",
                 note="same S-curve family and convention as t_mid")],
    "split": [dict(source="Placeholder (open question)", value=0.5, unit="", grade="F", note="")],
    "g_CF0": [dict(source="Scenario knob — no calibration yet (Q-5 ruling)", value=0.5,
                   unit="OOM/yr", grade="F",
                   note="declared a grade-F scenario dial like t_mid_F; follower-compute calibration pass to come")],
    "g_CF_inf": [dict(source="Scenario knob — no calibration yet (Q-5 ruling)", value=0.10,
                      unit="OOM/yr", grade="F",
                      note="declared a grade-F scenario dial like t_mid_F; follower-compute calibration pass to come")],
    # ---- alpha (brief 10, D-098). Values are the OBSERVABLE loss_half_gC in PERCENT, converted
    # from each source's alpha at the BASE eta = 1 (where loss = alpha/2). The delivered alpha
    # then moves with the active eta -- adopting Epoch's 0.67 gives 0.67 at eta = 1 and 0.44 at
    # eta = -2. That is the ratified anti-double-counting property, not a defect, which is why
    # _ALPHA_ETA_NOTE says so in the user's own terms on the menu itself rather than only here.
    #
    # THE FORK IS NOT RESOLVED BY FIAT (Pavel: "This as default and let user choose in the
    # calibration"). The two readings are separate GROUPS, each labelled with its construct: the
    # cost-share evidence identifies a LEVEL elasticity, while the model's alpha-channel argument
    # g_c(t)/g_c(0) is a GROWTH rate. The default sits between them by design.
    #
    # D-109 SWEEP: 13 rows -> 7, most defendable first. SEVEN rather than the rule's 3-5 by
    # Pavel's amendment ("it can be more if there are more reliable sources"): brief 10 §3 lists
    # the two audited HKEX prospectuses as their OWN grade-A evidence item -- "two audited IPO
    # prospectuses measure that share directly" -- second only to Epoch's published estimate, and
    # they are current, on the exact object, and superseded by nothing. Dropping them for a row
    # count would have thrown away the best-graded direct measurements the parameter has.
    # Dropped (6), every one archived with its full arithmetic in brief 10 and the evidence
    # register, because the menu is not the archive: Epoch's OpenAI stack 32.2 (costs-042 --
    # sits inside Epoch ε_K's own CI, so it adds no position), the Anthropic stack 29.3
    # (costs-043 -- already display_only, a residual-driven lower bound), Cottier-internal +
    # growth 26.5 (algo-038/039 -- the growth branch's extreme, the branch survives via its
    # central row), Barnett 25.0 (algo-006 -- a count of innovations under a cap, not an
    # elasticity), AI 2027's 20.0 (algo-050 -- already display_only, a level elasticity at a 10x
    # cut with n = 6) and Davidson 25.0 (algo-049 -- grade D and explicitly not evidence; its
    # teaching point, that the field assumes 0.5 unjustified and so did this widget until D-098,
    # is now written out in brief 10 §3, which is the only place it lives). _ALPHA_PRACTICE went
    # with that last one -- it was the only row in its group.
    #
    # ENVELOPE UNCHANGED at [22, 45]%: the union is still [23.5, 44.5], because the two rows that
    # SET it both survive. Cottier is load-bearing on the left -- dropping it would jump the union
    # to 29.5, delete the whole low branch, and contradict both the ratified alpha in [0.45, 0.90]
    # and the MC draw Tri(22, 35, 45). Gundlach is load-bearing on the right.
    "alpha": [
        dict(source="Midpoint of the two readings — the default", value=35.0, unit="%",
             grade="C", group=_ALPHA_DEFAULT,
             note="α = 0.70 at η = 1. Deliberately between the "
                  "level cluster (32–34%) and the growth cluster (39–44%) — the fork is a user "
                  "choice, and it is the user's, not ours"),
        # -- level reading: alpha = the compute share of R&D spend, S_E --
        dict(source="Epoch AI (Ho & Whitfill 2025) — ε_K, the only published estimate",
             value=33.5, unit="%", grade="B", ci=(29.5, 37.5), group=_ALPHA_LEVEL,
             note="ε_K ≈ 0.67 (range 0.59–0.75): 'the elasticity of output with respect to "
                  "capital should equal the compute share'. Verified verbatim 2026-07-28"),
        dict(source="Z.ai / Zhipu — audited HKEX prospectus", value=41.0, unit="%", grade="A",
             ci=(40.6, 41.4), group=_ALPHA_LEVEL,
             note="compute 82.7% of (compute+labour) R&D spend 2024, 81.2% H1 2025; cash+equity"),
        dict(source="MiniMax — audited HKEX prospectus", value=39.0, unit="%", grade="A",
             ci=(37.9, 40.2), group=_ALPHA_LEVEL,
             note="75.7% (2024) → 80.3% (9M 2025); near-frontier Chinese lab, steep trend"),
        dict(source="Cottier et al. 2024 — frontier training costs", value=27.5, unit="%",
             grade="A", ci=(23.5, 32.0), group=_ALPHA_LEVEL,
             note="hardware 47–64% incl. equity (61–76% excl.); ≤2023 vintage, and a per-model "
                  "amortised cost rather than a flow input share — the oldest object match of "
                  "the four, and the LEFT EDGE of the envelope"),
        # -- growth reading: alpha = S_E g_E / (S_E g_E + S_L g_L) --
        dict(source="Epoch ε_K + measured input growth (the model's own symbols)", value=41.8,
             unit="%", grade="C", ci=(39.2, 44.2), group=_ALPHA_GROWTH,
             note="α = S_E·g_E/(S_E·g_E+S_L·g_L) with R&D compute ~3×/yr vs headcount "
                  "~1.25–1.6×/yr ⇒ α ≈ 0.84. Collapses to the level reading only "
                  "if g_E = g_L. Read off the SAME "
                  "S_E = 0.67 as the Epoch row above, so the two are a controlled comparison: "
                  "one fork, two answers"),
        # -- bounds and counter-readings --
        dict(source="Gundlach et al. 2025 — scale-dependence share", value=44.5, unit="%",
             grade="B", group=_ALPHA_BOUND,
             note="≈89% of measured algorithmic progress is scale-dependent ⇒ α ≈ 0.9; read as "
                  "a MECHANISM instead it is η evidence, so it is not spent twice"),
    ],
    "r": [dict(source="Standard discount rate", value=0.08, unit="/yr", grade="C",
               note="user-cost extension only")],
}


def source_span(pkey):
    """[min, max] across a parameter's documented source values and measured CIs (D-042).
    Display-only rows (different objects, bounds, retired readings) are excluded."""
    vals = []
    for rw in CAL_SOURCES.get(pkey, []):
        if rw.get('display_only'):
            continue
        if isinstance(rw.get('value'), (int, float)):
            vals.append(float(rw['value']))
        ci = rw.get('ci')
        if ci is not None and rw.get('ci_default', True):
            vals += [float(ci[0]), float(ci[1])]
    return (min(vals), max(vals)) if vals else None


def bind_live_defaults(gamma):
    """Fill the rows whose value IS a live model default, called once by `model.py` on import.

    D-091 swept γ from nats to base 10 (0.2 → 0.2/ln 10) through `Params`, `PARAM_RANGES`, the
    interp, the pane, the spec and N4 — but not this table, a THIRD home for parameter numbers.
    The row kept the retired 0.2, which sits past the right end of the base-10 envelope, so the
    mini rail drew it as an out-of-envelope chevron and γ was left with no adoptable source at
    all. The fix was to make the row read the live default instead of a copy of it, and this
    function is how it keeps doing so now that the table cannot import `Params`. A literal
    written here instead would be exactly the two-literals defect all over again.
    """
    CAL_SOURCES["gamma"][0]["value"] = gamma
    missing = [(k, rw.get("source")) for k, rows in CAL_SOURCES.items() for rw in rows
               if rw.get("value") is None]
    assert not missing, f"unbound live defaults still in the table: {missing}"
