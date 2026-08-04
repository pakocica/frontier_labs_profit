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
# retired reading) · tier="hidden" (D-128: the row belongs to the menu's SECOND tier, rendered
# only after the reader opens "show more options" -- bounds, counter-readings, less popular
# constructions, historically ratified values. Absent = the PRIMARY tier, D-109's 3-5 most
# defendable rows. A hidden row is still a ROW: it is choosable unless it also carries
# display_only, and `source_span` unions BOTH tiers, so an envelope end may be witnessed by a
# hidden row but never by no row at all) ·
# triple (RETIRED with the money rows, D-093 -- name reserved, never reuse) ·
# disp (display string replacing "value unit", for a row whose machine value must not be rounded on
# screen -- an exact fraction, or, per D-124, a DEFAULT row carrying the shipped parameter's exact
# image so that [choose] restores the calibration rather than a rounded neighbour of it) · adopt (what to WRITE into the destination control when
# that differs from `value` -- currently UNUSED: it existed for the eta menu, whose destination was
# a selectbox that took an option LABEL rather than a number, and D-125 made eta an ordinary
# continuous dial. Kept in the schema and in `_use_source` because the next menu whose control is
# not a slider will need it again) · basis (money and coverage rows: the ACCOUNTING BASIS --
# calendar / run-rate / mixed -- mandatory disclosure, see the coverage block) · why
# (display_only rows: the short reason THIS row cannot be adopted, replacing the generic caption).

# =============================================================================================
# THE TWO-TIER RULE (D-128, amending D-109). Every menu has a PRIMARY tier -- D-109 unchanged,
# the 3-5 most defendable rows in declining order of relevance -- and an optional HIDDEN tier
# behind a "show more options" toggle at the foot of the menu, carrying bounds, counter-readings,
# less popular constructions and historically ratified values.
#
# Three properties, and they are what the tests pin:
#   1. The ENVELOPE is the union of ALL CHOOSABLE rows across BOTH tiers. `source_span` therefore
#      does not look at `tier` at all -- hiding a row changes what is on screen, never what the
#      dial may reach. `choosable <=> inside the envelope` is unchanged from D-109.
#   2. Every envelope end is witnessed by SOME row, and the witness may be hidden. An
#      unshown-but-reasonable option must exist as a hidden ROW, never as unwritten judgment:
#      envelope ends can never be witness-free.
#   3. Readings OUTSIDE the envelope stay `display_only` with a `why`, exactly as before. They
#      document a rejected construction without stretching the dial, and they may sit in either
#      tier (in practice the hidden one).
#
# The flexibility lives in DATA -- one flag per row -- rather than in when the rule applies, which
# is what separates the menu's job (curation: what a reader should see first) from the envelope's
# job (coverage: what the calibration can defend). HIDDEN ROWS ARE A CONTIGUOUS TAIL of each
# menu, because the toggle is at the foot: the renderer walks the list in order and a hidden row
# interleaved among primary ones would render out of place.
# =============================================================================================

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
# CHOOSABLE ⟺ the implied ρ₀ lands inside the vetted envelope (state.APP_RANGES). Rows
# outside it -- the three cross-dated readings D-104 retires, plus every restatement whose m came
# out ≤ 0 -- are display_only and carry `why`. The envelope is still NOT widened to admit
# any of them: the corner that would argue for a higher ceiling is the trust-the-spike pairing
# (52-63%), and costs-025/026 reject its R2 premise. Remaining flags are in §4 of the derivation
# file. The POPULATION question (§D.2) is closed: D-119 keeps the broad, industry-wide reading and
# its 33.5% default, so a lab that gives its weights away sits on both sides of the ratio rather
# than on neither. The struck-population construction stays on the menu as the labelled
# leader-player alternative -- ruled ON, not left open.
# ---- alpha group headings (D-098). The headings carry the CONSTRUCT, because the two readings
# answer different questions and a reader who cannot tell them apart cannot use the menu. The
# eta note is on the headings rather than buried in the log: a row labelled 0.67 that delivers
# 0.44 at another eta is exactly the two-literals trap this session hit repeatedly.
#
# D-125 REPLACED THE ANCHORING CONVENTION, and the heading text with it. Until then every row was
# converted at η = 1 (loss = α/2), which is right for one of these two groups and wrong for the
# other: a cost SHARE is a Cobb-Douglas object (under CD the output elasticity EQUALS the cost
# share, which is Epoch's own sentence), while the growth reading's α = S_E g_E/(S_E g_E + S_L g_L)
# is natively an η = 1 object — the weighted-average bracket IS a growth-share aggregator. So each
# row now converts through ITS OWN construct's native η, which is data (the `group` field) and not
# new code, and each row's card shows the number its source actually published.
_ALPHA_ETA_NOTE = ("each row restated at its own construct's η; the delivered α still moves with "
                   "the substitution setting")
_ALPHA_DEFAULT = f"The shipped default ({_ALPHA_ETA_NOTE})"
_ALPHA_LEVEL = ("Level reading — compute's share of R&D SPEND "
                "(a cost share, so read at η = 0)")
_ALPHA_GROWTH = ("Growth reading — compute's share of research-effort GROWTH "
                 "(a growth share, so read at η = 1)")
_ALPHA_BOUND = "Bounds and counter-readings"

# ---- eta group headings (D-125). The fork here is CONVENTION vs MEASUREMENT, not two constructs:
# η = 0 and η = 1 are both conventions and only one has a literature behind it, which is the whole
# content of the default flip. The headings say so, so a reader can see at a glance that the menu's
# one estimate sits between them.
_ETA_DEFAULT = "The shipped default — the literature's convention"
_ETA_SUBS = "Substitutes — the estimated reading"
_ETA_COMP = "Complements readings"
_ETA_CONV = "Superseded conventions"
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
#
# D-128 RE-ADDED THE CEILING'S WITNESS as the menu's first HIDDEN-tier row. The D-109 trim had
# deleted the labs-favourable corner, leaving the envelope's top end witnessed by nothing on
# screen: the floor 26.2% was a row, the ceiling was not, and the highest number a reader could
# see was 38.6% while the slider ran to 46. D-119 had ruled that ceiling INTENTIONAL (the envelope
# spans rival constructions on purpose), so the defect was never the number -- it was that the
# reasoning lived only in the log. The two-tier rule fixes it in the place it broke: the corner is
# a ROW again, in the tier that exists for exactly this kind of row, and the envelope's top end is
# now that row's own value. Consequence, mechanical and recorded: the ceiling moves 46 -> 46.3
# (+0.3 pp), which is the arithmetic of the construction rather than a re-ratification of it.
_COV_BOUNDS = "Bounds and counter-readings"
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
              "largely for its own use is counted on both sides or on neither. The default "
              "counts it on BOTH: \\$0 of earnings against its ~\\$10B share of the bill, so ρ₀ "
              "reads the whole industry's books, frontier-scale spending included wherever it "
              "happens. This row is the alternative reading — the LEADER PLAYER's own books, "
              "with the open-weight lab treated as part of the fringe it supplies. Worth ~5 pp, "
              "and the only discretionary population choice left"),
    dict(source="Industry 2026, Google struck from both sides",
         value=100.0 * 15.05 / 57.5, disp="ρ₀ = 26.2%", unit="%", grade="C", basis="calendar",
         note="\\$15.05B ÷ \\$57.5B = 26.2%. Google's +\\$10B is the one unsourced and undatable "
              "number on the earnings side, so striking it from the earnings AND its spend from "
              "the cost base is the honest floor rather than a curiosity — it is the FLOOR of "
              "the [26, 46.3] envelope. Dated to a common instant like the rows above it; read as "
              "a run-rate instead, the same construction gave 50.4%, which is the size of the "
              "error that dating removes"),
    dict(source="Industry 2026, Meta struck and the Google leg at its top",
         value=100.0 * 30.1 / 65.0, disp="ρ₀ = 46.3%", unit="%", grade="D", basis="calendar",
         tier="hidden", group=_COV_BOUNDS,
         note="\\$30.1B ÷ \\$65B = 46.3%. The same Meta-struck population as the row above, but "
              "with the Google earnings leg — the one unsourced and undatable number on the "
              "earnings side, carried at \\$10B in every other row — taken at the TOP of its "
              "\\$5–15B judgment span. Both discretionary choices are made in the labs' favour "
              "at once, which is what makes it a CORNER rather than an estimate, and why it is "
              "graded below the rows above it. It is shown because it is the reading that sets "
              "this dial's CEILING: the range deliberately spans two rival constructions, and "
              "the top of it should be a number you can read rather than one you have to "
              "reconstruct. Dated on the same rule as every row here"),
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
        # D-122 (Pavel, 2026-08-03): ci corrected (10.7, 14.3) -> the register's test-time-deflated
        # band (geff-010), and the NOTE rewritten. Pavel: "I thought that g_a was calibrated based
        # on the effective compute growth. I think it is all correct, you just were not careful in
        # the description." He is right on both halves -- the value stands, the description was
        # loose in two ways. It never said the row is an EFFECTIVE-COMPUTE rate (the menu header
        # says so 40 lines up; a card is read on its own), and it attributed the deflation to the
        # Qwen3 pair DIRECTLY. The Qwen3 pair (geff-006) sizes the reasoning-mode step; the share
        # the deflation actually uses is 10-20%, from geff-009 (inference tiers = 4-9% of the
        # measured envelope speed) plus geff-007 (the step as a one-time ~15% level shift).
        dict(source="Epoch “Rosetta Stone” §3.2.2, test-time-deflated", value=11.34, unit="×/yr",
             grade="B", ci=(10.5, 14.1), group="Bias-corrected reading — the calibrated default",
             note="THE CALIBRATED SPOT, and like every row on this menu it is an "
                  "EFFECTIVE-COMPUTE rate: 11.34 ×/yr = g_C0 3.24 × the algorithmic rate 3.50. "
                  "The dial the menu writes is t_eff_x, and g_a = log10(t_eff_x) − g_C0 falls out "
                  "as the residual — so this row calibrates the algorithmic term through "
                  "effective-compute growth, it does not measure it directly. Construction: "
                  "Epoch's delivered-capability estimate, 5.86 ×/yr algo on the ECI basis, "
                  "DEFLATED by the test-time share of the measured envelope speed. That share is "
                  "10–20%: top inference tiers contribute 4–9% of the envelope's speed, and the "
                  "one-time reasoning-mode step adds ~15% (sized on the zero-confound Qwen3 "
                  "Instruct→Thinking pair — same weights, same date, +5.81 ECI points). "
                  "Capability bought at inference time belongs in neither a nor c, which is what "
                  "the deflation is for. The interval is that share's own 10 / 15 / 20% spread — "
                  "4.36 / 3.77 / 3.25 ×/yr algo ⇒ [10.5, 14.1] ×/yr effective. Its parent, "
                  "undeflated, is the 19.0 row at the bottom of this menu"),
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
    # a defendability ranking. ENVELOPE UNCHANGED: the ratified box is still [1.5, 3.0] ×/OOM
    # with the mode at 2.1 and the union at 1.62 … 2.84.
    #
    # D-120 RE-KEYED THE UNIT, NOT THE EVIDENCE. The dial is the value growth RATE, so every row
    # is stated as one — each construction's own ×/OOM reading carried through the leader's t = 0
    # speed ẋ^L(0) = ×11.34/yr of effective compute (the anchor ⟺ ×2.10/OOM, the D-118 rider's
    # rounding of the 118.68 %/yr that ×2.1 exactly implies; only the pooled row moves, because
    # only it is the anchor — the three construction rows keep their own readings). The ×/OOM
    # figure stays on each row, because it is what the source measured and what the literature is
    # quoted in; the rate is what the model grants. A row's rate is its reading AT THIS
    # CALIBRATION's frontier speed — dial the speed elsewhere and the same ×/OOM evidence implies
    # a different rate, which is exactly the coupling the re-key makes visible instead of hiding
    # inside a single number.
    #
    # D-133 (Pavel, 2026-08-03) states that rate in ×/yr rather than %/yr, for consistency with
    # every other rate dial. The map is affine — ×/yr = 1 + (%/yr)/100 — so no row's evidence,
    # ordering or grade moves; `value` and `ci` are re-expressed and `disp` leads with the ×/yr
    # reading. THE %/yr FIGURE STAYS ON EVERY ROW as the side mention, which is not decoration:
    # the growth literature these rows come from is quoted in percent, and a row must keep saying
    # what its source said.
    #
    # D-122 (Pavel, 2026-08-03) FIXED THE POOLED ROW'S "MEDIAN" CLAIM, which was false: the three
    # siblings below read 1.86 / 2.24 / 2.30, whose median is 2.24, not the shipped 2.08. The
    # register's pooled figure (valaut-091) is an equally-weighted Monte Carlo over the three
    # routes' own input bands, p50 = 2.08 x/OOM -- a central value of a pooled distribution, which
    # is a different object from the median of three point estimates. The same sentence lived in
    # two other places (ui/content.py's `nu` INTERP and `t_value_growth`, plus the Params.nu
    # comment); all were fixed in the same batch.
    "nu": [
        dict(source="Pooled — the three constructions together", value=None,
             disp="×2.19/yr (119 %/yr; ≈×2.10 per OOM)", unit="×/yr",
             grade="B", ci=(1.75, 2.79), group="Aggregate constructions — the model's object",
             note="THE CALIBRATED SPOT — ×2.19/yr, i.e. 119 %/yr, the ruled anchor, which is "
                  "×2.1029/OOM at this frontier speed. The POOLED CENTRAL VALUE across the three "
                  "routes: an "
                  "equally-weighted Monte Carlo over their own input bands, whose p50 is "
                  "×2.08/OOM. It is NOT the median of the three numbers below — those are each "
                  "restated on this calibration's ruler, and a central value of the pooled "
                  "distribution is not a median of three point estimates. The 90% band "
                  "[1.7, 2.65]×/OOM is [×1.75, ×2.79]/yr — [75, 179] %/yr — here. "
                  "Why ~2×/OOM and not the 4–16×/OOM the single-channel benchmarks show: those "
                  "measure ONE fixed basket of tasks each, and a dollar-weighted mixture of "
                  "channels whose midpoints are spread over ±2–3 OOM aggregates to ~2×/OOM even "
                  "when every individual channel ramps at 16×. The benchmark slopes CORROBORATE "
                  "this aggregate, they never competed with it"),
        dict(source="Revenue decomposition", value=2.341,
             disp="×2.34/yr (134 %/yr; ×2.24 per OOM)",
             unit="×/yr", grade="B",
             ci=(1.902, 2.873), group="Aggregate constructions — the model's object",
             note="observed revenue growth ÷ the frontier speed this calibration fixes; the only "
                  "route restated on the current ruler. ×2.24/OOM, 90% band [1.84, 2.72]"),
        dict(source="GATE ramp + wage-bill ceiling", value=2.407,
             disp="×2.41/yr (141 %/yr; ×2.30 per OOM)", unit="×/yr", grade="B",
             ci=(2.055, 3.007), group="Aggregate constructions — the model's object",
             note="automation accounting; OOM-native. ×2.30/OOM, 90% band [1.98, 2.84]"),
        dict(source="Davidson value datum", value=1.924,
             disp="×1.92/yr (92 %/yr; ×1.86 per OOM)",
             unit="×/yr", grade="B",
             ci=(1.663, 2.308), group="Aggregate constructions — the model's object",
             note="FLOP-gap arithmetic; OOM-native, and carrying the chord-vs-slope correction. "
                  "×1.86/OOM, 90% band [1.62, 2.21]"),
    ],
    # ------------------------------------------------- brief 11: nu_inf (D-107 / D-118 / D-120)
    # THE MENU THIS DIAL NEVER HAD. Until now `CAL_SOURCES` had no `nu_inf` key at all, so its »
    # panel opened with an empty Sources block while its prose called it a grade-F placeholder —
    # both of which D-107 had already superseded by calibrating it.
    #
    # WHAT MAKES A GROWTH-LITERATURE ROW A READING OF nu_inf. Nothing measures an asymptotic
    # willingness-to-pay elasticity, and nothing will (brief 11 §4: ~50 queries, eight databases,
    # no published estimate, no AI price index, one hedonic that withholds its coefficients). What
    # earns the C is not a measurement but the model's own identity: long-run value growth is
    # nu_inf × the leader's long-run capability speed, so every published view of how fast an
    # AI-era economy grows IS a view of nu_inf — and the growth literature is grade A. The
    # denominator is the model's, ẋ^L(T) = 0.409 OOM/yr at these defaults, which is also why the
    # dial is stated as a RATE: the rate is what the sources bound, the slope is what the model
    # uses. D-133 keys that rate in ×/yr; every row keeps the %/yr reading its source published,
    # because the growth literature is quoted in percent and these rows must go on saying so.
    #
    # ORDER is D-109's — declining relevance, five rows, and the menu is not the archive. The
    # eight-row table in Notes/audit_decisions_round2_2026-07-29.tex §A.5 keeps every row that is
    # not here (Gordon's 0.3–0.8%, Jones's weak-links 5%, Nordhaus's 20% singularity threshold,
    # the retired ×1.25 placeholder), as does brief 11 §3.1. Dropped from the menu, not from the
    # record: Gordon and Jones-weak-links sit between rows that are already here and add no
    # position, and Nordhaus's threshold is a marker one step below Davidson's.
    #
    # ENVELOPE [×1.00, ×1.30]/yr = [0, 30] %/yr contains every row and is witnessed at both ends:
    # ×1.00 (no growth) is the stagnation limit (Nordhaus Eq. 6 / Baumol Prop. 4), ×1.30 is
    # Davidson's explosive threshold — the top of this menu, which is the D-109 rule working the
    # way round it is meant to (menu → range).
    "nu_inf": [
        dict(source="Amodei, Dwarkesh interview 2026 — the granted premise", value=None,
             disp="×1.10/yr (10 %/yr; ×1.22 per OOM)", unit="×/yr", grade="B",
             ci=(1.10, 1.20),
             group="Growth worldviews — the model's own identity",
             note="THE CALIBRATED SPOT, and deliberately adversarial: the optimist's own forecast "
                  "of long-run AI-era growth, 10–20 %/yr, granted at its FLOOR. If the leader "
                  "fails to cover its bill even on the most favourable published view of what "
                  "capability is worth, the verdict does not rest on our pessimism. The 10–20 % "
                  "band — ×1.10 to ×1.20/yr on this dial — is also the default Monte-Carlo draw"),
        dict(source="Jones 2026 — US trend, “roughly 2% for 150 years”", value=1.02,
             disp="×1.02/yr (2 %/yr; ×1.04 per OOM)", unit="×/yr", grade="A",
             group="Growth worldviews — the model's own identity",
             note="the no-acceleration reading: AI is another general-purpose technology and the "
                  "trend that survived electricity and the computer survives it too. The "
                  "mainstream position, and by far the best-evidenced number on this menu"),
        dict(source="Acemoglu / PWBM / Goldman — mainstream, AI-boosted", value=1.03,
             disp="×1.03/yr (2.5–3.5 %/yr; ×1.05–1.08 per OOM)", unit="×/yr", grade="A",
             ci=(1.025, 1.035),
             group="Growth worldviews — the model's own identity",
             note="three independent macro estimates of what AI adds to trend growth, all "
                  "landing within a percentage point of each other and just above Jones's trend"),
        dict(source="Korinek & Suh — region-2 steady state", value=1.18,
             disp="×1.18/yr (18 %/yr; ×1.42 per OOM)", unit="×/yr", grade="A",
             group="Growth worldviews — the model's own identity",
             note="a transformative but non-explosive regime: substantial automation, growth an "
                  "order of magnitude above trend, still a steady state. The nearest published "
                  "worldview to the granted premise above"),
        dict(source="Davidson — explosive-growth threshold", value=1.30,
             disp="×1.30/yr (30 %/yr; ×1.74 per OOM)", unit="×/yr", grade="A",
             group="Growth worldviews — the model's own identity",
             note="the threshold above which growth is called explosive — a marker rather than a "
                  "central estimate, and the TOP of this menu, which is what sets the envelope's "
                  "ceiling. Pairing it with this model's 0.41 OOM/yr horizon speed is generous "
                  "to the high end: the explosive-growth literature generally assumes faster "
                  "capability growth than that, so it buys more value per OOM here than there"),
    ],
    # ------------------------------------------------------------- brief 01 rider: g_p (TC2b)
    # D-124 (Pavel, 2026-08-03) gave this default row the parameter's EXACT IMAGE, 10**0.14,
    # with "1.38 ×/yr" kept on the card as the display string -- morning brief §B.1 option (i),
    # ruled as the general rule for every menu. The row used to carry a literal 1.38, which
    # inverts to g_p = 0.13987908640123647 and NOT to the ratified 0.14, so [choose] on the
    # default row landed on a different model than [reset] did. The fix is on the ROW, not on the
    # parameter: g_p = 0.14 OOM/yr is what D-076 ratified and what every golden contains.
    "g_p": [
        dict(source="Trusted hardware leg (the calibration)", value=10.0 ** 0.14,
             disp="1.38 ×/yr", unit="×/yr", grade="B", ci=(1.30, 1.45),
             note="THE CALIBRATED SPOT, g_p = 0.14 OOM/yr. Prices fall to 72% of the year before; "
                  "implied bill growth 10^(g_C0−g_p) = 2.35×/yr vs Cottier's observed 2.4×/yr. "
                  "The card reads 1.38; the row itself carries the exact image of 0.14 OOM/yr, "
                  "so choosing it sets the dial to the calibrated default exactly rather than to "
                  "a rounded neighbour of it"),
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
    # D-129 (Pavel, 2026-08-03) REBUILT THIS MENU, one row -> six, on his ruling that the floor is
    # "hardware price-performance + the asymptotic economic growth". Every row is now a PAIR of
    # legs added in OOM/yr, because that is what the parameter is:
    #
    #     g_C_inf = hardware price-performance  +  real training-budget growth
    #
    # and the model's implied long-run budget growth 10^(g_C_inf − g_p) is an exact read-out of
    # the choice. The shipped single row (the Epoch-lineage cluster midpoint, ×1.35) was the
    # HARDWARE LEG ALONE — an assertion that the leader's real budget shrinks 2.33 %/yr forever,
    # which is not what its own card claimed. It is not archived as a row here because it is not
    # a pair: rows 2 and 5 below are its two coherent readings (the trusted hardware leg and the
    # broadest one, each with a flat budget), and its full three-reading provenance rides on
    # row 1's note.
    #
    # TIERS (D-128). PRIMARY: the calibrated spot, the flat-budget world it replaces, and the top
    # of the range — the three a reader needs to see the construct. HIDDEN: the two rows built on
    # the broadest hardware reading (1.30 rather than the trusted 1.38), which are the low corner
    # and the envelope's floor, plus the display_only horizon-floor reading below.
    #
    # THE ENVELOPE IS THE UNION, exactly: 1.30 (row 5) … 1.5184226910631733 (row 3).
    #
    # D-124's exact-image rule holds on row 1: it carries 10**0.15, so [choose] restores the
    # calibrated parameter rather than a rounded neighbour, and the card reads "1.41 ×/yr".
    "g_C_inf": [
        dict(source="Hardware price-performance + trend economic growth",
             value=10.0 ** 0.15, disp="1.41 ×/yr", unit="×/yr", grade="C",
             group="Two legs — hardware price-performance × budget growth",
             note="THE CALIBRATED SPOT. Two legs: a dollar buys ~38% more compute each year, and "
                  "the training budget grows with the economy at ~2.3 %/yr. The hardware leg is "
                  "MEASURED — three Epoch-lineage readings of FLOP per dollar spanning "
                  "1.30–1.39×/yr (Hobbhahn, Heim & Aydos 2023 at 1.39 on an FP32 basis; its 2022 "
                  "predecessor at 1.30–1.40; Epoch's 2024 real-dollar whole-node series at 1.30) "
                  "— and the model's own price of compute, 1.38×/yr, sits inside it, which is "
                  "why this row uses that rather than the cluster's midpoint: there is one price "
                  "series in the model, and reading the floor against a second one would make "
                  "the card's arithmetic and the model's disagree. The budget leg is the "
                  "mainstream long-run growth figure. Being three readings of one construct from "
                  "one research lineage, they are not fully independent; and the floor LEVEL "
                  "remains an extrapolation — no series measures post-2030 frontier compute "
                  "growth, and power binds ~2030 (Sevilla et al. 2024). The card reads 1.41; the "
                  "row carries the exact image of 0.15 OOM/yr, so choosing it sets the dial to "
                  "the calibrated default exactly"),
        dict(source="Hardware price-performance alone — real budget flat forever",
             value=10.0 ** 0.14, disp="1.38 ×/yr", unit="×/yr", grade="C",
             group="Two legs — hardware price-performance × budget growth",
             note="the flat-budget world: hardware improves, the real training budget never grows "
                  "again. This is what the previous calibration meant to assert, and it is the "
                  "natural lower reading if AI capex plateaus in real terms. It is also the one "
                  "row where the two legs are the same number, so the implied budget growth "
                  "beside the dial reads exactly 1.00"),
        dict(source="Hardware price-performance + AI-boosted growth",
             value=10.0 ** 0.14 * 1.10, disp="1.52 ×/yr", unit="×/yr", grade="C",
             group="Two legs — hardware price-performance × budget growth",
             note="sustained AI-era growth at the optimist's own FLOOR, 10 %/yr, carried into the "
                  "training budget. The top of this menu, and what sets the range's ceiling. "
                  "Above this the budget's share of output rises without bound, which is not "
                  "something a floor can assert forever — which is why the range stops here"),
        dict(source="Broadest hardware reading + trend growth", value=1.3325,
             disp="1.33 ×/yr", unit="×/yr", grade="C", tier="hidden",
             group="Lower readings and counter-readings",
             note="the most conservative defensible pair: the broadest hardware cost basis "
                  "(whole-node, real dollars, ~30 %/yr) with the budget still growing at trend"),
        dict(source="Broadest hardware reading, real budget flat", value=1.30,
             disp="1.30 ×/yr", unit="×/yr", grade="C", tier="hidden",
             group="Lower readings and counter-readings",
             note="the low corner: the broadest hardware reading with a budget that stops "
                  "growing. The floor of the range"),
        dict(source="Read as a 2031–2036 horizon floor, not an asymptote",
             value=10.0 ** 0.14 * 1.25, disp="1.73 ×/yr", unit="×/yr", grade="C",
             tier="hidden", display_only=True,
             group="Lower readings and counter-readings",
             why="this reads the dial as a five-year horizon rate rather than as an asymptote, "
                 "which is a different object — see the note",
             note="the same hardware leg with the budget growing 25 %/yr: what the floor would be "
                  "if it governed only ~2031–2036 rather than the long run. Over a window that "
                  "short the training budget's share of output CAN still be rising (AI datacentre "
                  "investment is ~0.8% of US GDP today), so this is a defensible reading of the "
                  "next decade. It is shown rather than offered because this dial is the "
                  "ASYMPTOTE: a floor a user can hold above trend growth forever asserts a "
                  "budget that outgrows the economy without limit. The realized rate at the "
                  "horizon can sit above the asymptote, and nothing here says otherwise"),
    ],
    # D-127 removed two cost-side rows with the parameters they calibrated: "ell" (the
    # payment-weighted composite, 0.45 yr, grade B, ci [0.25, 1.3] — D-123) and "phi_RD" (folded
    # into the bill anchor, 0.0 ×, grade B, the register's ONLY display_only row — D-126). Both
    # rows are preserved verbatim in Notes/calibration/retired/README.md, which is the reversal
    # path. The semantics phi_RD's row carried — the cost anchor is the OBSERVED bill, compute AND
    # R&D / researcher overhead together, so a separate markup would double-count — now live at
    # the cost anchor itself, in model_profit.cost_flow's docstring.
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
    # ------------------------------------------------ x_mid (brief v2, D-130 commission, D-132)
    # THE MENU THAT CITED NOBODY. D-130 regraded the three shipped rows C -> F because they were
    # REFERENCE POINTS on a dial nothing measured -- "here is what 2 / 5 / 10 would mean" -- and
    # the C chip claimed competing estimates while the rows only ever did the first. Two of them
    # (5.0, 10.0) were values no publication holds; one had an empty note. All three are GONE.
    #
    # WHAT MAKES A SOURCE A READING OF x_mid, which is the discipline the old document lacked.
    # There is no ceiling any more (D-083 retired it), so x_mid is a PURE SCALE on the capability
    # axis: f depends on x and x_mid only through x/x_mid, and x_mid is where the value slope has
    # fallen halfway from nu to nu_inf. A source therefore pins it only by supplying a LOCATION
    # and a COMPLETION FRACTION together -- "by X* further OOMs the marginal value of an OOM has
    # fallen a fraction phi of the way to its floor" -- through
    #
    #     x_mid = X* / (1 + log10(phi/(1-phi)) / lam(p0_w)).
    #
    # At p0_w = 1% that reads phi = 50% -> X*, phi = 90% -> 0.677 X*, phi = 99% -> HALF of X*.
    # Most published statements are full-automation DATES, phi ~ 0.99, so the model reads them at
    # half their face value; reading a saturation point straight into the dial doubles it. This
    # single conversion is what re-triaged the literature, and it is why Korinek-Suh's 3-16
    # became 1.5-8 and why the old document's ratified spot of 7 does not survive.
    #
    # FOUR THINGS x_mid IS NOT, each of which cost the old menu a row: not a saturation point
    # (see the box); not a half-VALUE point (that was the pre-D-083 bounded logistic, and
    # Notes/calibration/param_x_mid.md is written entirely in that retired parameterisation);
    # not a benchmark-saturation date (GDPval, SWE-bench and the AA index are bounded METRICS
    # that saturate by construction -- zero weight); and not a diffusion lag (slow enterprise
    # adoption bends REALISED value, which is a drift term this model does not carry, not W(x)).
    #
    # TIERS (D-128), most defendable first. PRIMARY: the two compute-denominated GATE readings
    # and the two Davidson readings of one datum. HIDDEN: the no-bend regime marker, which sets
    # the dial's top, and the Davidson low tail, now display_only (see PARAM_RANGES for Pavel's
    # rounding ruling). Rows (b) and (d) are deliberately BOTH present and deliberately NOT
    # averaged: same source, same datum, two construct spaces, and a midpoint would manufacture
    # a reading nobody holds. The factor-2.7 disagreement between them IS the honest uncertainty.
    #
    # THE GRADE CEILING IS B, and the reason is a hard negative result rather than caution: NO
    # source fits or parameterises a dollar-valued logistic in capability OOMs. Every formal
    # demand-side or task-based growth paper is calibrated to calendar time or to an
    # automation-share variable, never to a compute/OOM axis (lit_W_saturation.md sec.1.6,
    # register valaut-112). B becomes reachable when somebody measures M -- the total remaining
    # addressable-value multiple -- which under the identity below pins the whole curve at once.
    "x_mid": [
        dict(source="Epoch AI GATE — where half of tasks are automated", value=6.0,
             unit="OOM", grade="B", ci=(4.9, 7.5),
             group="Compute-denominated automation ramps",
             note="THE CALIBRATED SPOT, and the only anchor that is natively in this dial's own "
                  "units. GATE puts full automation at 10^36.5 effective FLOP and runs its "
                  "automation ramp over Δf = 55% of the distance from today's largest training "
                  "run (5e25 FLOP), i.e. 0.55 × (36.5 − 25.70) = 5.94 OOM. The ramp does NOT "
                  "start from zero — GATE's own FAQ calibrates ~10% of tasks as already "
                  "automated — so half of TASKS sits (0.5 − 0.10)/(1 − 0.10) = 44.4% along it, "
                  "not at its geometric midpoint. Netting the vintage adjustment gives 6.00. "
                  "Read live off the playground 2026-08-03, because Epoch publishes no ramp "
                  "width: it is a ratio in the docs, and the width has to be closed on (T, Δf, "
                  "C_T(0)). The band spans both presets, the f_init ambiguity and the vintage"),
        dict(source="Davidson — “20% of value automated”, read in dollars", value=7.4,
             unit="OOM", grade="B", ci=(6.7, 7.8),
             group="Compute-denominated automation ramps",
             note="the 20%-automation milestone is worth ~$10T/yr on his own accounting; fitted "
                  "against the ~$62T global wage bill that puts the "
                  "half-way point 6.7–7.8 OOM out. The DOLLAR reading of the datum row (d) "
                  "reads as a share — same author, same number, and see (d) for why they differ"),
        dict(source="Epoch AI — the automation-share growth thresholds", value=4.3,
             unit="OOM", grade="B", ci=(4.1, 4.4),
             group="Compute-denominated automation ramps",
             note="Epoch's own thresholds — 30% of tasks automated is where growth can pass "
                  "20%/yr, 50–70% is the explosive bar — placed on GATE's ramp. The 50% point is "
                  "the half-way point by definition, which is what makes this a location claim "
                  "rather than a growth claim. The most aggressive grade-B value on the menu"),
        dict(source="Davidson — the same milestone, as a share of value", value=2.7,
             unit="OOM", grade="C", ci=(1.7, 4.5), ci_default=False,
             group="The same datum in value-share space",
             note="Davidson's native object is a value-weighted SHARE, and it bends far earlier "
                  "than the dollar reading: 20% of value at +1.5 OOM, full automation 4 OOM "
                  "further. Factor 2.7 below row (b) on the same datum, and the disagreement is "
                  "REAL rather than a rounding — which is why both rows are here and neither is "
                  "averaged away. Grade C, not B: it inherits a judgement-based [1, 9] OOM gap "
                  "distribution, an unresolved ±1 OOM AGI-FLOP anchor discrepancy (10^35 vs "
                  "10^36) and an undocumented f_∞ convention. Its band is that judgement gap, "
                  "hence ci_default=False — offered for [choose range], excluded from the span"),
        dict(source="No bend inside the horizon", value=12.0, unit="OOM", grade="C",
             tier="hidden", group="Regime markers",
             note="the skeptic branch (Nordhaus: six supply/demand singularity tests fail) and "
                  "the takeoff branch (AI-2027, Aschenbrenner) share ONE prediction — value per "
                  "capability-OOM does not ease within ten years. Past ~12 the transition is "
                  "under 9% done at the horizon and the parameter stops being identified at all "
                  "(only the horizon-average slope is), so this is a MARKER for a regime rather "
                  "than an estimate, and it is what sets the top of the slider. Exactly the role "
                  "the asymptotic value dial's own no-acceleration marker plays"),
        dict(source="Davidson's value-share reading at his own low gap tail", value=1.7,
             unit="OOM", grade="C", tier="hidden", display_only=True,
             group="Regime markers",
             why="below the dial: the envelope's endpoints are round numbers, which puts the "
                 "floor at 2.0 and this reading 0.3 OOM outside it",
             note="row (d)'s own published low end, across Davidson's stated 1–9 OOM gap "
                  "distribution — a candidate for the envelope's floor, which the dial rounds "
                  "to 2, so it is shown here instead of offered. "
                  "It still earns its place on the menu: it is the deepest "
                  "published reading inside the region where the Level-2 verdict FAILS, and the "
                  "reader should be able to see how far the evidence reaches below the dial"),
        # ---- readings that are NOT this object, shown so the refusals are visible ----
        dict(source="Korinek & Suh — task-difficulty tail index", value=3.464,
             disp="1.5–8 OOM", unit="OOM", grade="C", tier="hidden", display_only=True,
             group="Different objects and bounds",
             why="a scenario DATE spanning the whole envelope, not a location",
             note="THE THEORY OF WHETHER A FINITE x_mid EXISTS AT ALL, and on that it is the "
                  "most valuable item in this file: bounded or thin-tailed task complexity "
                  "implies saturation, a Pareto thick tail implies none — and under the current "
                  "parameterisation it is also what licenses nu_inf being REACHED. But its "
                  "calendar calibration is a scenario choice anchored on Hinton's 5–20 yr, and "
                  "it is a FULL-AUTOMATION date (phi ≈ 0.99), so the conversion box halves it on "
                  "the way in: the old document's 3–16 is 1.5–8 here. Disciplines nothing alone"),
        dict(source="METR task-length plateau", value=2.5,
             disp="1.5–3.5 OOM", unit="OOM", grade="C", tier="hidden",
             display_only=True, group="Different objects and bounds",
             why="ONE value channel, ~10^-3 of the wage bill in dollar weight",
             note="genuine demand-side saturation, and genuinely measured — but of a single "
                  "channel, and admissible only after an explicit doublings-per-OOM conversion. "
                  "The dial is a dollar-weighted mixture over channels whose midpoints are "
                  "spread over ±2–3 OOM. Folding a single channel's bend into an aggregate "
                  "midpoint teaches the reader the wrong thing"),
        dict(source="Remote Labor Index — channel bend", value=0.65,
             disp="0.6–0.7 OOM", unit="OOM", grade="C", tier="hidden",
             display_only=True, group="Different objects and bounds",
             why="ONE value channel, same grouping rule as METR above",
             note="the only LIVE paid-outcome logistic in existence, and the reason to watch it "
                  "quarterly rather than adopt it: it carries a clean pre-registered prediction "
                  "— the leader crosses ~50% around early-to-mid 2027 and the series visibly "
                  "bends thereafter. That is a falsification test this dial does not otherwise "
                  "have, on a horizon short enough to matter"),
        dict(source="Register pooled value — three routes together", value=6.24,
             disp="6.24 OOM [4.77, 8.56]", unit="OOM", grade="B", tier="hidden",
             display_only=True, group="Different objects and bounds",
             why="a construction over three routes, not a reading any publication states",
             note="an equally-weighted Monte Carlo over the three routes' own input "
                  "bands. Recorded because its BY-ROUTE spread — Davidson 7.38, GATE 5.50, "
                  "revenue 6.19 — is the honest uncertainty; excluded because no source holds "
                  "it. Its GATE leg rests on a DIFFERENT chain from row (a)'s, and a later "
                  "register pass found the two do not reconcile: on the register's own reading "
                  "the leg corrects 5.50 → 5.13 and the pooled to ~6.1, while on the live-"
                  "playground reading it barely moves. The fork is Epoch's own documentation "
                  "inconsistency (the Δf numerator anchors at C(20%), the FAQ says ~10%), worth "
                  "±0.4 OOM, and it is out for an email rather than resolved here"),
        dict(source="The retired default", value=10.0, disp="10.0 OOM", unit="OOM", grade="F",
             tier="hidden", display_only=True, group="Different objects and bounds",
             why="no publication supports it, and it is internally inconsistent by 18.5×",
             note="what this dial shipped before the value-block calibration, behind a row called "
                  "'Harvest-continues reference' that named no source. Under the bounded-market "
                  "identity the pair (x_mid = 10, p0_w = 1%) implies W(∞)/W(0) = 1853× while "
                  "p0_w's own reading asserts 100× — see model_profit.value_coherence, the check "
                  "this row is the reason for. Kept reachable to read, never to choose"),
    ],
    "g_a_F": [
        dict(source="Scale-bias low, 0.6 × leader (Gundlach)", value=0.6, unit="× leader",
             grade="B", note=""),
        dict(source="Scale-bias central, 0.7 × leader (Gundlach)", value=0.7, unit="× leader",
             grade="B", note=""),
        dict(source="Scale-bias high, 0.8 × leader (Gundlach)", value=0.8, unit="× leader",
             grade="B", note="one source, three readings; the dial IS this share (g_a^F = share × g_a)"),
    ],
    # eta was the app's one CHOICE dimension until D-125 (Pavel, 2026-08-03) made it a continuous
    # dial and flipped its default 1 -> 0. Its rows therefore stop being a special case: `value`
    # is the number, the dot lands on an ORDINARY rail, and [choose] writes the number itself --
    # the `adopt` field these rows used to carry existed only because the destination control was
    # a selectbox that needed its option LABEL written into it, and there is no selectbox now.
    # `disp` still keeps the card head reading in the dial's own vocabulary.
    #
    # THE ENVELOPE IS THIS MENU'S UNION, by D-128's clause 2 and mechanically: [-1.20, +1.00] =
    # the Davidson row's CI floor to the perfect-substitution convention, matching
    # PARAM_RANGES['eta'] bitwise at both ends. Both witnesses live in the HIDDEN tier, which is
    # exactly what that tier is for -- an envelope end may be witnessed by a hidden row, never by
    # no row at all. sigma = 1/(1-eta) is the reading throughout: sigma = 2.58 <-> eta = 0.612,
    # sigma = 1 <-> eta = 0, sigma -> 0 <-> eta -> -inf.
    #
    # TIERS, per D-128's declining-relevance rule:
    #   PRIMARY  eta = 0    the shipped default and the field's own convention (grade A as a
    #                       documented convention);
    #            eta = 0.61 the only econometric ESTIMATE the parameter has (grade B).
    #   HIDDEN   eta = -0.20 a counter-reading -- the complements side, grade D -- which the menu
    #                       carried nowhere at all after D-122, even though the register holds
    #                       three complements readings (algo-013, algo-016, algo-049);
    #            eta = +1.00 a HISTORICALLY RATIFIED VALUE (D-018's simplification, superseded by
    #                       D-125) and grade F, which is two of the hidden tier's own categories
    #                       at once. Showing the F-graded superseded convention above the
    #                       estimate would work against the ruling that retired it.
    # Hidden rows are a contiguous tail, so the order is primary-then-hidden and, inside each
    # tier, declining relevance.
    #
    # D-122 (Pavel, 2026-08-03): the η = −2 ROW IS GONE, not relabelled. It carried Whitfill &
    # Wu's complements spec, which estimates σ = −0.10 (SE 0.176) — i.e. the evidence points at
    # η → −∞, and algo-013 says verbatim that this is NOT the η = −2 menu point, that "an
    # arbitrary strong-complements stand-in was chosen instead". Pavel: "I don't want eta→−∞ to
    # be included in the option, perhaps you should remove this option at all." An arbitrary
    # stand-in has no place on an evidence menu, so the menu offers only what a source actually
    # states. η = −2 remains a legal Params value (the model is continuous in η, and the port
    # fixtures' `ces_negative_eta` scenario still probes it); it is simply off the dial.
    "eta": [
        dict(source="Cobb–Douglas (σ = 1) — the shipped default", value=0.0,
             disp="η = 0", unit="", grade="A", group=_ETA_DEFAULT,
             note="Davidson, Halperin, Houlden & Korinek 2026 (NBER WP 35155) take σ = 1 as the "
                  "baseline, chosen to rule out bottlenecks — grade A as a DOCUMENTED "
                  "CONVENTION, not as a measurement. It is also the only setting at which α is "
                  "identified by its own best evidence: under Cobb–Douglas the output elasticity "
                  "EQUALS the cost share, so the audited compute-share readings are α directly"),
        dict(source="Whitfill & Wu 2025 — substitutes (σ = 2.58)", value=0.61,
             disp="η = 0.61", unit="", grade="B", group=_ETA_SUBS,
             note="N = 27, 4 labs; SE 0.341, and the IV spec reads σ = 2.77 (η = 0.639). The "
                  "only econometric estimate on this menu"),
        dict(source="Davidson 2025 — economy-wide CES estimates", value=-0.20,
             ci=(-1.20, -0.15), disp="η = −0.20", unit="", grade="D", tier="hidden",
             group=_ETA_COMP,
             note="his own stated view is 'most likely, −0.2 < ρ < 0' — ρ is our η, same letter "
                  "and same functional form — citing economy-wide CES estimates spanning "
                  "σ = 0.87–0.455, i.e. η = −0.15 … −1.20. Grade D: a judgment over borrowed "
                  "economy-wide estimates, not a measurement of AI R&D. It WITNESSES the "
                  "envelope's complements end, which is why the dial reaches −1.20 and no lower"),
        dict(source="Perfect substitution — the model's earlier default", value=1.0,
             disp="η = 1", unit="", grade="F", tier="hidden", group=_ETA_CONV,
             note="the perfect-substitution simplification this model shipped with until the "
                  "Cobb-Douglas flip: a tractability convention with "
                  "no literature behind it, which is what grade F records. It is the CES "
                  "family's mathematical ceiling — σ = 1/(1−η) is negative above it — so it also "
                  "sets the dial's right end. Kept reachable because a reader should be able to "
                  "put the model back where it was and see what the flip is worth"),
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
    # ------------------------------------------------ beta0 (brief v2, D-130 commission, D-132)
    # D-130 regraded the one shipped row C -> F: its own source line said "no observable yet",
    # a grade and a note contradicting each other on the same card. D-132 gives it a menu, an
    # envelope and a CONSTRUCT -- and deliberately does NOT give it a calibrated spot.
    #
    # B1 (Pavel, 2026-08-03) RULES THE INPUT READING. beta0 = psi(0) - 1 is the counterfactual
    # multiplier on frontier-lab research THROUGHPUT: sigma_0 = beta0/(1+beta0) is the share of
    # today's throughput attributable to AI assistance, equivalently the fraction that would be
    # lost if today's assistance were withdrawn with the human workforce held fixed. It is NOT
    # an output speedup. Pavel's own way of putting it -- "shouldn't we be choosing an observable
    # instead" -- is exactly the argument for it: the input reading is the ETA-INVARIANT one.
    # An output speedup has to be inverted through the CES to become beta0, so its beta0 moves
    # whenever eta or alpha moves (D-125/D-131 shifted every such row), while a with-versus-
    # without throughput trial reads psi(0) straight off with no conversion at all. The widget
    # shipped BOTH readings on the same card -- `interp` said throughput, `cal_target` said "AI
    # makes AI R&D ~30% faster" -- and at eta = 1 they differed by 10x. D-132 fixes the label.
    #
    # WHAT beta0 DOES, and does not do. It never changes TODAY'S rate: algo_growth_L divides psi
    # by psi(0), so adot_L(0) = g_a bitwise at every beta0, and the measured g_a is a residual
    # referenced to 2026 frontier practice that already contains whatever assistance labs enjoy.
    # What beta0 fixes is the SHAPE of the assistance path, through R_psi(x) = (1 + beta0*
    # 10^(gamma x))/(1 + beta0). So the object is HEADROOM, not speed.
    #
    # THE EVIDENCE IS PLENTIFUL AND NONE OF IT MEASURES THE OBJECT, which is why the grade stays
    # F after this menu lands. Every grade-A row below is general software developers on coding
    # tasks, mostly 2022-25 tool vintages; every frontier-lab-RESEARCH row is self-report or
    # telemetry derived from code volume. The cross-lab controlled trial (1.04-1.20) and the
    # frontier-lab self-reports (2.83-4.00) are approximately the same population months apart
    # and DISAGREE BY 15x. Three findings close that question rather than opening it: the lab
    # code-share statistics (>80%, 75%) are SHARES, not multipliers, and Anthropic says of its
    # own that it "measures quantity over quality"; METR's task-substitution result shows one
    # underlying 5x task speedup reading +67% / +124% / +200% depending on whether you measure
    # observed tasks, new tasks or VALUE; and Epoch's own interviews cap the channel, since
    # hypothesis and planning "occupy relatively little time". beta0 earns C when B1's construct
    # meets an actual frontier-lab RESEARCH-throughput measurement. See register algo-052/-053.
    #
    # AND THE 2025-26 MOVEMENT IS MOSTLY NOT STRUCTURAL. Amodei's two dated readings (~5% Aug
    # 2025 -> 15-20% Feb 2026) imply the AI term of psi grew 5.3x over dx ~ 0.53 OOM, which needs
    # gamma ~ 1.36 -- roughly 8x above gamma's envelope top (0.1737) and deep inside the blow-up
    # regime. So the movement is adoption and tooling diffusion, which the model has no term for.
    # beta0 is a snapshot of a fast-moving transient, and the wide envelope is not caution: it is
    # the correct representation of a quantity that moved 5x in six months for reasons outside
    # the model. (Brief v2 sec.9 states this as "16x above gamma's envelope top"; re-derived
    # against PARAM_RANGES['gamma'] it is 7.8x at eta = 1 and 6.4x at eta = 0. The conclusion is
    # untouched -- the register carries the corrected figures.)
    #
    # RIDER THAT ANY FUTURE beta0 RULING MUST CARRY: beta0 is only identified JOINTLY with gamma.
    # On a ten-year horizon what the model sees is the scalar R_psi(X), and a 75x move in beta0
    # changes the ten-year answer by ~9% if gamma moves with it, while flipping the L3 profit
    # sign if gamma is held fixed. gamma has no adoptable source row at all, so the pair cannot
    # be closed from this menu.
    "beta0": [
        dict(source="METR Frontier Risk Report 2026 — cross-lab trial, low end", value=0.04,
             disp="ψ(0) = 1.04×", unit="", grade="B",
             group="Direct throughput readings — no conversion",
             note="the ONLY controlled measurement on the right population (frontier-lab staff, "
                  "Feb–Mar 2026), reported as a range with no published method. It carries the "
                  "whole envelope floor by itself, which is why it is worth an email. The source "
                  "calls its own trial an underestimate"),
        dict(source="METR Frontier Risk Report 2026 — same trial, high end", value=0.20,
             disp="ψ(0) = 1.20×", unit="", grade="B",
             group="Direct throughput readings — no conversion",
             note="same trial, top of the stated ~4–20% band"),
        dict(source="Cui et al. 2025 — pooled 3-company RCT, N = 4,867", value=0.26,
             disp="ψ(0) = 1.26×", unit="", grade="A",
             group="Direct throughput readings — no conversion",
             note="+26.1% weekly completed tasks. THE ONE GRADE-A CAUSAL THROUGHPUT ESTIMATE the "
                  "parameter has, and the reason the shipped 0.3 survives as bracket-consistent "
                  "— but it measures general enterprise developers on coding tasks, not frontier "
                  "research. If you want a source-backed number today, this is it: it costs "
                  "almost nothing on the horizon against the held 0.3"),
        dict(source="METR technical-worker survey, Mar 2026 — median 2× value", value=1.00,
             disp="ψ(0) = 2.00×", unit="", grade="B",
             group="Direct throughput readings — no conversion",
             note="n = 349, bias-corrected, broad technical workers. A VALUE multiplier rather "
                  "than a task count, which is the right object and the wrong population"),
        dict(source="METR on Anthropic telemetry, Jul 2026 — 2.83× researcher uplift",
             value=1.83, disp="ψ(0) = 2.83×", unit="", grade="B",
             group="Direct throughput readings — no conversion",
             note="telemetry + CES, frontier-lab researchers — the right population and the "
                  "wrong instrument (derived from code volume). Its authors say 'plausibly >2×'. "
                  "Note where this sits: the L3 profit sign flip is at beta0 = 1.837"),
        dict(source="Anthropic research-staff poll, Mar 2026 — median 4× output", value=3.00,
             disp="ψ(0) = 4.00×", unit="", grade="D", tier="hidden",
             group="Bounds and the output family",
             note="n = 130, self-report, and its own publisher says the true figure is lower. "
                  "The strongest published reading of the model's own population, which is "
                  "exactly why it sets the envelope CEILING and is not primary"),
        # Row (g) is FILLED IN BY bind_live_defaults -- see there, and see D-132. It is the one
        # row on this menu whose value is not a source's number but a CONVERSION of one, and
        # freezing that conversion is precisely the mistake the whole beta0 pass exists to stop.
        dict(source="Amodei, Feb 2026 — 15–20% R&D speedup, inverted through the model's CES",
             value=None, unit="", grade="D", tier="hidden",
             group="Bounds and the output family",
             note="A DIFFERENT OBJECT — an OUTPUT speedup where this dial is an INPUT level — "
                  "and therefore the only row here that must be computed at the LIVE α and η "
                  "rather than written down; the value and band on this row are FILLED IN at "
                  "import from the shipped α and η, and the note below states which. Kept "
                  "because it is the only source whose object matches the sentence the paper "
                  "used to make"),
        # ---- readings the model cannot represent, shown so the refusals are visible ----
        dict(source="METR RCTs — −19% (2025), −18% / −4% (2026)", value=-0.19,
             disp="ψ(0) = 0.81× ⇒ β₀ = −0.19", unit="", grade="A", tier="hidden",
             display_only=True, group="Structurally unrepresentable",
             why="β₀ < 0 makes ψ DECREASING in capability, which no source claims",
             note="grade A, on experienced OSS developers working in their own repositories, and "
                  "the model has no slot for it: ψ = 1 + β₀·10^{γx} with β₀ < 0 falls as "
                  "capability rises. The honest representation of 'no net uplift today' is "
                  "β₀ = 0, reachable through the freeze switch. Shown so the refusal is visible "
                  "rather than silent — this is the best-graded evidence on the whole menu"),
        # Also filled in by bind_live_defaults, for the same reason as row (g): its β₀ is a
        # CONVERSION of an output statement, so freezing it would rot the moment α or η moves.
        dict(source="AI-2027 — 1.5× current-day R&D multiplier", value=None,
             unit="", grade="C", tier="hidden",
             display_only=True, group="Structurally unrepresentable",
             why="an OUTPUT multiplier, and one no beta0 could express under the earlier convention",
             note="an exact match to the output object. Under the retired η = 1 it sits past the "
                  "hard cap 1/α = 1.43×, so no β₀ could express it; under Cobb–Douglas it is an "
                  "ordinary row. That the same statement is 'impossible' or 'ordinary' depending "
                  "on a convention is the sharpest available argument for B1's input reading"),
        dict(source="Google 75% / Anthropic >80% of new code AI-generated", value=4.0,
             disp="σ₀ ≤ 80% ⇒ β₀ ≤ 4.0", unit="", grade="D", tier="hidden",
             display_only=True, group="Structurally unrepresentable",
             why="a share of OUTPUT, which bounds σ₀ from above at best — never a multiplier",
             note="Anthropic says of its own metric that it 'measures quantity over quality'. "
                  "The most-quoted numbers in this area and the least usable"),
    ],
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
    # from each source's alpha at ITS OWN construct's native eta (D-125): the level / cost-share
    # rows and the Gundlach bound convert at eta = 0, loss = 1 - 2^(-alpha), because a cost share
    # is a Cobb-Douglas object; the GROWTH row converts at eta = 1, loss = alpha/2, unchanged,
    # because the weighted-average bracket is itself a growth-share aggregator. NOT ONE SOURCE'S
    # ALPHA MOVED -- only the observable each alpha implies, which is why the group headings state
    # the convention on the menu itself. The delivered alpha then moves with the ACTIVE eta:
    # adopting Epoch's 37.15% gives 0.67 at eta = 0 and 0.743 at eta = 1. That is the ratified
    # anti-double-counting property, not a defect.
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
    # ENVELOPE MOVED at D-125, [22, 45] -> [27, 47]%, and only because the CONVERSION moved: the
    # union goes [23.5, 44.5] -> [27.80, 46.04] when each row is read at its own construct's eta.
    # It NARROWS on the left and WIDENS on the right, so both ends had to move at once. Cottier is
    # still load-bearing on the left (its CI low end 27.80 sets the floor) and Gundlach on the
    # right (46.04 now sits above the old 45 ceiling). ALPHA_LOSS_CEILING = 0.5 is untouched:
    # loss = 50% <=> alpha = 1 holds at EVERY eta by construction, which is the invariant that
    # lets one slider serve the whole eta dial, and 47% still sits below it.
    #
    # Every percent below is 100*(1 - 2^-alpha) for the level rows and 100*alpha/2 for the growth
    # row -- written as the arithmetic for the DEFAULT row (D-124's exact-image rule, so [choose]
    # restores the calibration bitwise) and as the rounded reading for the rest, whose sources are
    # nowhere near four-figure precision.
    "alpha": [
        dict(source="Midpoint of the two readings — the default",
             value=100.0 * (1.0 - 2.0 ** -0.70), disp="38.44%", unit="%",
             grade="C", group=_ALPHA_DEFAULT,
             note="α = 0.70, read at the base η = 0 (Cobb–Douglas), where loss = 1 − 2^(−α). "
                  "Deliberately between the two READINGS that answer the question most directly: "
                  "the published cost-share estimate (Epoch, 37.15% here) and the growth reading "
                  "(41.8%). The two constructs answer DIFFERENT questions — a "
                  "share of the level of spend, and a share of its growth — so which one α "
                  "should be is a fork, and the fork is the user's, not ours. The row carries the "
                  "exact image of α = 0.70, so choosing it restores the calibrated spot exactly"),
        # -- level reading: alpha = the compute share of R&D spend, S_E. Natively eta = 0: under
        #    Cobb-Douglas the output elasticity EQUALS the cost share, which is what makes these
        #    rows readable as alpha at all. loss = 1 - 2^(-alpha).
        dict(source="Epoch AI (Ho & Whitfill 2025) — ε_K, the only published estimate",
             value=37.15, unit="%", grade="B", ci=(33.57, 40.54), group=_ALPHA_LEVEL,
             note="ε_K ≈ 0.67 (range 0.59–0.75): 'the elasticity of output with respect to "
                  "capital should equal the compute share'. Verified verbatim 2026-07-28. That "
                  "sentence is a COBB-DOUGLAS statement, which is why this row is read at η = 0"),
        dict(source="Z.ai / Zhipu — audited HKEX prospectus", value=43.36, unit="%", grade="A",
             ci=(43.04, 43.67), group=_ALPHA_LEVEL,
             note="compute 82.7% of (compute+labour) R&D spend 2024, 81.2% H1 2025; cash+equity "
                  "(α = 0.82)"),
        dict(source="MiniMax — audited HKEX prospectus", value=41.76, unit="%", grade="A",
             ci=(40.87, 42.72), group=_ALPHA_LEVEL,
             note="75.7% (2024) → 80.3% (9M 2025); near-frontier Chinese lab, steep trend "
                  "(α = 0.78)"),
        dict(source="Cottier et al. 2024 — frontier training costs", value=31.70, unit="%",
             grade="A", ci=(27.80, 35.83), group=_ALPHA_LEVEL,
             note="hardware 47–64% incl. equity (61–76% excl.) ⇒ α = 0.55; ≤2023 vintage, and a "
                  "per-model amortised cost rather than a flow input share — the oldest object "
                  "match of the four, and the LEFT EDGE of the envelope"),
        # -- growth reading: alpha = S_E g_E / (S_E g_E + S_L g_L). Natively eta = 1 -- the
        #    weighted-average bracket IS a growth-share aggregator -- so loss = alpha/2 and this
        #    row is the one D-125 did NOT move.
        dict(source="Epoch ε_K + measured input growth (the model's own symbols)", value=41.8,
             unit="%", grade="C", ci=(39.2, 44.2), group=_ALPHA_GROWTH,
             note="α = S_E·g_E/(S_E·g_E+S_L·g_L) with R&D compute ~3×/yr vs headcount "
                  "~1.25–1.6×/yr ⇒ α ≈ 0.84. Collapses to the level reading only "
                  "if g_E = g_L. Read off the SAME "
                  "S_E = 0.67 as the Epoch row above, so the two are a controlled comparison: "
                  "one fork, two answers"),
        # -- bounds and counter-readings --
        dict(source="Gundlach et al. 2025 — scale-dependence share", value=46.04, unit="%",
             grade="B", group=_ALPHA_BOUND,
             note="≈89% of measured algorithmic progress is scale-dependent ⇒ α ≈ 0.9; read as "
                  "a MECHANISM instead it is η evidence, so it is not spent twice. The RIGHT "
                  "EDGE of the envelope"),
    ],
    "r": [dict(source="Standard discount rate", value=0.08, unit="/yr", grade="C",
               note="user-cost extension only")],
}


def source_span(pkey):
    """[min, max] across a parameter's documented source values and measured CIs (D-042).
    Display-only rows (different objects, bounds, retired readings) are excluded.

    D-128: `tier` is deliberately NOT consulted. The span is the union over every CHOOSABLE row
    of BOTH tiers, so a menu's envelope is fixed by what the calibration can defend rather than
    by what the panel shows first, and moving a row into the hidden tier can never silently
    narrow a dial."""
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


def beta0_from_output_speedup(S, alpha, eta):
    """Invert an OUTPUT speedup S into the INPUT level β₀ = ψ(0) − 1, through the model's own
    CES bracket. Returns NaN when S is unrepresentable at this (α, η).

    ȧ^L(0) is g_a for every β₀ (ψ is normalised by ψ(0)), so S is read as the counterfactual:
    the ratio of today's rate to the rate with assistance WITHDRAWN, ψ → 1 at fixed compute.
    That gives S^{-η} = (1−α)(1+β₀)^{-η} + α, i.e.

        η = 0 :  β₀ = S^{1/(1−α)} − 1                       (Cobb–Douglas, unbounded in S)
        η ≠ 0 :  β₀ = [(S^{-η} − α)/(1−α)]^{-1/η} − 1

    THE η = 1 CAP IS AN ARTEFACT AND THIS FUNCTION IS WHY IT IS VISIBLE. Under perfect
    substitution S is bounded by 1/α = 1.43 at the ratified α, so AI-2027's 1.5× is
    unrepresentable at ANY β₀; under Cobb–Douglas the same statement is an ordinary row. D-132
    binds both output-family rows through here rather than freezing their numbers, because a
    number whose meaning changes with a convention is exactly what the β₀ pass exists to stop."""
    S, alpha, eta = float(S), float(alpha), float(eta)
    if abs(eta) < 1e-12:
        return float(S ** (1.0 / (1.0 - alpha)) - 1.0)
    inner = (S ** (-eta) - alpha) / (1.0 - alpha)
    if inner <= 0.0:
        return float('nan')
    return float(inner ** (-1.0 / eta) - 1.0)


def bind_live_defaults(gamma, t_value_growth=None, t_value_growth_inf=None,
                       alpha=None, eta=None):
    """Fill the rows whose value IS a live model default, called once by `model.py` on import.

    D-091 swept γ from nats to base 10 (0.2 → 0.2/ln 10) through `Params`, `PARAM_RANGES`, the
    interp, the pane, the spec and N4 — but not this table, a THIRD home for parameter numbers.
    The row kept the retired 0.2, which sits past the right end of the base-10 envelope, so the
    mini rail drew it as an out-of-envelope chevron and γ was left with no adoptable source at
    all. The fix was to make the row read the live default instead of a copy of it, and this
    function is how it keeps doing so now that the table cannot import `Params`. A literal
    written here instead would be exactly the two-literals defect all over again.

    D-120 binds the two VALUE dials' default rows the same way, for a sharper version of the
    same reason. D-109 requires the default row to carry the dial's value BITWISE, so that
    clicking it restores the calibrated spot exactly. In ×/OOM those spots were the literals 2.1
    and 1.25; as growth rates they were the forward images of ν and ν_∞ through two capability
    speeds the model computes (one of them simulated), so any literal here would have been a
    rounded copy of a derived number and [choose] would have landed a hair off the calibrated spot.

    D-118's rider (Pavel, 2026-08-02) turned those two rates into the RULED anchors — 119 and
    10 %/yr, whole percent, which D-133 re-keys to ×2.19 and ×1.10 per year — and made ν / ν_∞
    their images. So what `model.py` passes in is the anchor itself: still one number, still not a
    second copy, but now the primitive rather than the derived side, and the card shows the ruling
    instead of a float artefact of it. The `disp` strings carry the reading for the card; the
    machine value is the anchor the dial inverts.
    """
    CAL_SOURCES["gamma"][0]["value"] = gamma
    if t_value_growth is not None:
        CAL_SOURCES["nu"][0]["value"] = float(t_value_growth)
    if t_value_growth_inf is not None:
        CAL_SOURCES["nu_inf"][0]["value"] = float(t_value_growth_inf)
    if alpha is not None and eta is not None:
        # D-132, B4: the two OUTPUT-family β₀ rows are CONVERSIONS, not source numbers, so they
        # are computed here at the live (α, η) instead of being written down. Brief v2 sec.11
        # states row (g) as "0.45–1.25", which is a SPLICE of two conventions — 0.447 is η = 0's
        # low end and 1.250 is η = 1's high end. The coherent pairs are [0.447, 0.619] at η = 0,
        # α = 0.6215 and [0.769, 1.250] at η = 1, α = 0.700; at the α this model actually ships
        # (0.70, D-098) the Cobb-Douglas pair is [0.593, 0.836]. Binding it removes the choice.
        lo = beta0_from_output_speedup(1.15, alpha, eta)
        hi = beta0_from_output_speedup(1.20, alpha, eta)
        conv = "Cobb–Douglas" if abs(float(eta)) < 1e-12 else f"η = {float(eta):g}"
        g_row = next(r for r in CAL_SOURCES["beta0"] if r["source"].startswith("Amodei"))
        g_row["value"] = 0.5 * (lo + hi)
        g_row["ci"] = (lo, hi)
        g_row["ci_default"] = False      # a converted band, not the source's own interval
        g_row["disp"] = f"β₀ = {lo:.2f}–{hi:.2f} at α = {float(alpha):.2f}, {conv}"
        ai2027 = next(r for r in CAL_SOURCES["beta0"] if r["source"].startswith("AI-2027"))
        b_ai = beta0_from_output_speedup(1.50, alpha, eta)
        ai2027["value"] = 0.0 if b_ai != b_ai else b_ai        # NaN -> 0.0, the row is dead data
        ai2027["disp"] = ("ψ(0) = 1.50× — UNREPRESENTABLE at this α, η"
                          if b_ai != b_ai else f"ψ(0) = 1.50× ⇒ β₀ = {b_ai:.2f} ({conv})")
    missing = [(k, rw.get("source")) for k, rows in CAL_SOURCES.items() for rw in rows
               if rw.get("value") is None]
    assert not missing, f"unbound live defaults still in the table: {missing}"
