"""Progressive levels (widget ladder): labels, per-level ranged keys, and the pins that make
each lower level EXACTLY the full model with later mechanisms pinned. (The level CARD renderer
retired with the Introduction tab — Pavel, round 2; content.LEVEL_INTRO carries the short
per-level intro now.)
"""
from .model_access import P0


# ======================================================================= progressive levels
# The widget is layered: each level adds a mechanism BLOCK to the model AND reveals the
# parameters that drive it. Lower levels PIN the not-yet-introduced parameters (ℓ=0, γ=0, τ=0,
# merged δ, extensions off) so the simplified model is exactly the full model with those pins.
# D-081 (2026-07-27): the old levels 2–5 (training in advance · growth engine · compute
# slowdown · value saturation) MERGED into one Level 2 "Dynamics" — Pavel's story: Level 1 is
# the steady-growth model; Level 2 introduces the dynamics, two opposing forces (compute growth
# slows down — which is also what makes ℓ matter — while algorithmic progress accelerates via
# RSI), plus the value-saturation bend. The old-ladder ↔ new-ladder map: 1↔1, {2,3,4,5}↔2,
# 6↔3 (tests/golden_level_pins.json locks the correspondence bit-for-bit).
# Pavel's ladder amendment (2026-07-27, supersedes the D-044 "parked, reversible" tail): "we
# only have levels 1-6 [old numbering], 7-9 were retired" — the widget ladder ENDS at 3, with
# NO parked slots after it. Release delay is x^R-parked in the spec (N9); the cost-mechanism
# level and φ_RD are RETIRED outright (φ_RD is provably inert under the observed-bill anchor);
# the extensions level is retired, its dials (χ, conduct, own-compute II.6, labor II.7) parked
# in the SPEC with N9-style revival notes — the model machinery stays in the notebook, always
# pinned off below. Revival = adding levels back, not flipping a constant.
MAX_LEVEL = 3
_ALL_LEVEL_LABELS = ["1 · Basics", "2 · Dynamics", "3 · Catch-up channels"]
LEVEL_LABELS = _ALL_LEVEL_LABELS[:MAX_LEVEL]

# The RETIRED mechanisms' values of record: τ = 0 (immediate release), φ_RD = 0 (the observed
# bill k·R₀ is the anchor, so the overhead split moves nothing — hence its retirement), and
# g_p stays pinned at P0.g_p (the measured hardware leg). apply_level_pins applies these
# UNCONDITIONALLY now.
PIN_TAU = 0.0
PIN_PHI_RD = 0.0

# Keys first exposed / sampled at each level (cumulative up to the current level). D-037: keys
# starting with t_ are TARGETS (drawn in natural units from TARGET_RANGES and inverted per draw);
# the rest are free dials drawn in parameter space from PARAM_RANGES.
LEVEL_RANGED = {
    # D-076: the BASE (Level 1) is the whole calibrated model — compute scaling, EFFECTIVE-compute
    # growth (g_a is its residual), value per OOM, the money side and the fringe lag. D-080: the
    # ONE money dimension is coverage, cov0 (app-side, state.APP_RANGES, in percent) — the only
    # identified object; mc_prepare converts its crop onto the `rho` draw. D-093 made that the
    # whole story: there is no longer a triple standing behind the dial, so nothing on the money
    # side can be sampled by accident. t_price_x is sampled without a dial: g_p is a trusted
    # measured leg, not a base control, but its uncertainty belongs in the fan.
    1: ["t_compute_x", "t_eff_x", "t_value_x", "cov0", "t_lag_mo", "t_price_x"],  # ℓ pinned 0
    # D-081 merged Level 2 — Dynamics: ALL of old 2–5 unpin at once, in the FROZEN old-ladder
    # concatenation order (order is load-bearing — mc_draw_batch consumes the list in order):
    # ℓ (pay ℓ ahead for the NEXT model), the ψ growth engine (g_a's LEVEL is set at L1), the
    # compute slowdown S-curve g_c0 → g_c∞ half-done at t_mid (D-082 — ξ retired), and the
    # D-083 value-slope transition (x_mid re-keyed as its midpoint; the NEW asymptotic-slope
    # target t_value_inf_x appended last — the merge lock's key list re-frozen for it). D-084
    # appends the two POSITION dials of the Dynamics curves, p0_c (slowdown) and p0_w (value
    # easing), last again — both POINT defaults in the MC, so no default draw changes.
    # D-098 appends the alpha observable LAST, the same placement D-084 used for the position
    # dials: mc_draw_batch consumes this list in order, so appending keeps every existing
    # target's position in the draw sequence.
    #
    # It does NOT keep the drawn VALUES, and the tempting claim that it does is false --
    # measured, not assumed. Placement only buys the FIRST attempt of the first draw: at L3,
    # draw #1 differs solely in g_a_F (a `scale_of`, drawn after the targets). Beyond that the
    # one extra rng value consumed per attempt re-phases the whole stream, and the bill-coherence
    # rejection loop amplifies it -- at L2 the first accepted draw took 4 rejections with this
    # dimension against 1 without, so even draw #1 lands somewhere else entirely. From draw #2
    # onward every dimension moves at both levels.
    #
    # That is inherent to adding a DRAWN dimension and is the D-083 precedent, not a defect: the
    # fan is re-drawn, deliberately. L1 is untouched (this key is not in its list) -- verified
    # bit-identical.
    2: ["ell", "gamma", "beta0", "t_floor_x", "t_mid", "x_mid", "t_value_inf_x",
        "p0_c", "p0_w", "loss_half_gC"],
    # lag inversion switches merged → channels; D-084 appends the follower curve's own position
    3: ["g_a_F", "g_CF0", "g_CF_inf", "t_mid_F", "split", "p0_F"],
}
X_MID_EXP = 200.0   # Level-1 pin AND the W byte-route sentinel (D-083): with ν_∞ ≡ ν and x_mid
                    # ≥ 100, W runs the pre-D-083 bounded-logistic evaluation byte-for-byte
                    # (≡ 10^{νx} far beyond double precision on any reachable x)


def merged_delta(level):
    """Inversion-branch predicate (audit X-18): True while the follower is PURE catch-up — the
    merged single δ of levels 1–2; the channels inversion starts at Level 3. Every consumer
    (mc.py, sidebar.py, calibration.py) routes through this ONE name because a bare `level <= 2`
    missed in a future renumbering would not raise — it would silently mix the merged δ with
    the two-channel inversion."""
    return level <= 2


def level_sample_keys(level):
    """Ranged keys the Monte-Carlo samples at this level (everything else is pinned)."""
    keys = []
    for L in range(1, level + 1):
        keys += LEVEL_RANGED[L]
    return keys


def apply_level_pins(d, LEVEL):
    """Pin every not-yet-introduced mechanism so the simplified model is exactly the full
    model with those pins (ℓ=0, γ=0, τ=0, merged δ, exponential value, no follower engine).
    Moved verbatim from the monolith sidebar body; runs after the L1/L2 controls set d.
    """
    # the base model (Level 1) is exact — later levels ADD mechanism blocks on top of it.
    # D-081: the whole Dynamics block (old levels 2–5) pins/unpins as ONE unit at Level 2 —
    # the individual pins inside the LEVEL < 2 branch are byte-for-byte the old per-level pins,
    # so the full ladder (now Level 3) is bit-identical to the old Level 6 (the merge lock in
    # tests/test_level_merge.py holds the machine guarantee).
    if LEVEL < 2:
        # ---- Dynamics pins (merged old L2–L5): Level 1 is the STEADY-GROWTH model ----------
        # L1 pays for the CURRENT model's compute: B_t = B₀·10^{c^L(t)−c^L(0)}·10^{−g_p t}, so
        # cost(0) = B₀ exactly (D-090: B₀ IS today's bill, k·R₀). Training in advance (ℓ > 0) debuts at Level 2 — under steady
        # growth the ℓ-timing only re-anchors the internal cost constant; the slowdown (also
        # Level 2) is what makes it matter.
        d["ell"] = 0.0
        d["A1"] = True          # base model: algo progress at the constant rate g_a (exactly)
        d["gamma"] = 0.0        # (redundant under A1, kept for clarity)
        # NOTE (D-076): g_a is NOT pinned here. Its LEVEL is a base-model quantity — the
        # residual of the Level-1 effective-compute dial, g_a = g_eff − g_C0 — so the sidebar
        # sets d["g_a"] at every level. Level 2 changes how g_a is PRODUCED (the ψ feedback),
        # not how big it is.
        # L1 compute growth is CONSTANT at g_C0 (the slowdown debuts at Level 2). Pinning the
        # floor to today's rate makes the D-082 curve exactly constant for ANY midpoint AND any
        # position p0_c (its amplitude is zero — D-084 adds the second "any"), so the leader's
        # speed holds and the base gap stays at Δ0. The position dials therefore need no pin:
        # like t_mid / t_mid_F they are unreachable at L1 by construction, not by convention.
        # D-088 makes "amplitude zero ⇒ exactly constant" a MACHINE guarantee rather than an
        # algebraic one: gamma_shape short-circuits y_inf == y(0) to the exact constant, so the
        # tie below cannot leave a one-ulp amplitude behind (see its docstring).
        d["g_C_inf"] = d["g_C0"]
        # D-083: Level 1 switches the value-slope transition OFF — ν_∞ ≡ ν makes w = νx (pure
        # exponential value); x_mid stays at the X_MID_EXP sentinel, which routes W through the
        # pre-D-083 evaluation byte-for-byte (the merge lock's L1 digest is absolute).
        d["nu_inf"] = d["nu"]
        d["x_mid"] = X_MID_EXP
    if LEVEL < 3:
        # base-model follower = PURE catch-up: no engine of its own. Pinning its algo rate and its whole
        # compute path to zero makes c^F constant and ẋ^F = ȧ^F = δ·(x^L − x^F) exactly (the merged δ
        # routes through δ_rel on the full gap). The follower's own engine is introduced at Level 3.
        d["g_a_F"] = 0.0
        d["g_CF0"] = 0.0
        d["g_CF_inf"] = 0.0
    # ---- RETIRED mechanisms (Pavel's ladder amendment): always pinned, at every level ------
    # base cost = the compute-path mechanism with NO R&D overhead: B_t = B₀·10^{c^L(t+ℓ)−c^L(ℓ)}·10^{−g_p t}
    # (anchor c^L(0), D-036 4th amendment). φ_RD retired outright; τ (release delay) is
    # x^R-parked in the spec (N9); g_p is the measured hardware leg (D-076).
    d["tau"] = PIN_TAU          # immediate release
    d["phi_RD"] = PIN_PHI_RD
    d["g_p"] = P0.g_p           # explicit pin, so every consumer reads d (never a default fallback)
    d["dt"] = P0.dt
