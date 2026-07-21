"""Progressive levels (widget ladder): labels, per-level ranged keys, the level card, and
the pins that make each lower level EXACTLY the full model with later mechanisms pinned.
"""
import streamlit as st

from .content import LEVEL_CARDS, _sub_live
from .model_access import P0


# ======================================================================= progressive levels
# The widget is layered: each level adds a mechanism to the model AND reveals the parameters that
# drive it. Lower levels PIN the not-yet-introduced parameters (ℓ=0, γ=0, τ=0, merged δ,
# extensions off) so the simplified model is exactly the full model with those pins.
# D-044 (2026-07-20): levels 7–9 removed for now — REVERSIBLE: set MAX_LEVEL back to 9 and
# every gated `if LEVEL >= 7/8/9` branch across ui/ (release-delay section, cost-mechanism
# group, extensions, under-the-hood) comes back to life. Their per-level data below is kept.
MAX_LEVEL = 6
_ALL_LEVEL_LABELS = ["1 · Basics", "2 · Training in advance", "3 · Growth engine",
                     "4 · Compute slowdown", "5 · Value saturation", "6 · Catch-up channels",
                     "7 · Release delay", "8 · Cost mechanism", "9 · Extensions"]
LEVEL_LABELS = _ALL_LEVEL_LABELS[:MAX_LEVEL]

# D-044: with MAX_LEVEL = 6 the L7/L8 pins in apply_level_pins ALWAYS fire, so these named
# constants are the values of record for the removed levels' parameters. PIN_PHI_RD is the ONE
# place to touch when the φ_RD calibration patch (central 1.0 → 3.0,
# Notes/calibration/param_phi_RD.md) is integrated after merge — set it to P0.phi_RD to
# inherit the notebook default instead of the pre-D-044 level story's 0 (cost(0) = S0 exactly).
# g_p stays pinned at P0.g_p (the D-039 bill-growth anchor pins it from Basics).
PIN_TAU = 0.0
PIN_PHI_RD = 0.0

# Keys first exposed / sampled at each level (cumulative up to the current level). D-037: keys
# starting with t_ are TARGETS (drawn in natural units from TARGET_RANGES and inverted per draw);
# the rest are free dials drawn in parameter space from PARAM_RANGES.
LEVEL_RANGED = {
    1: ["t_compute_x", "t_value_x", "theta", "S0", "t_lag_mo"],   # ℓ pinned 0 at L1
    2: ["ell"],                                # training in advance: pay ℓ ahead for the NEXT model
    3: ["t_algo_x", "gamma", "rho0"],          # the ψ growth engine
    4: ["t_floor_x", "xi"],                    # compute slowdown: g_c(t) decays g_c0 → g_c∞ at rate ξ
    5: ["x_mid"],                              # the logistic saturation bend x_mid debuts here
    6: ["g_a_F", "g_CF0", "g_CF_inf", "xi_F", "split"],   # lag inversion switches merged → channels
    7: [],                                     # τ is swept, not sampled
    8: ["phi_RD", "t_bill_x"],                 # cost mechanism ADDS the R&D overhead, frees g_p
    9: ["t_profit_B", "alpha", "eta", "r"],
}
X_MID_EXP = 200.0   # low-level pin: x_mid this large makes the logistic W indistinguishable from W0·10^{νx}


def level_sample_keys(level):
    """Ranged keys the Monte-Carlo samples at this level (everything else is pinned)."""
    keys = []
    for L in range(1, level + 1):
        keys += LEVEL_RANGED[L]
    return keys


def level_views(level):
    v = ["Model paths", "Finance"]
    if level >= 7:
        v.append("Release delay")
    v.append("Monte Carlo")
    if level >= 9:
        v.append("Under the hood")
    return v

def level_card(level, d):
    """Render the 'what's new at this level' educative card (headline equation lives in the panel)."""
    heading, words, _eq, note = LEVEL_CARDS[level]
    with st.container(border=True):
        st.markdown(f"**Level {level} · {heading}**")
        col_words, col_note = st.columns(2, gap="large")
        col_words.markdown(_sub_live(words, d))
        col_note.caption(_sub_live(note, d))


def apply_level_pins(d, LEVEL):
    """Pin every not-yet-introduced mechanism so the simplified model is exactly the full
    model with those pins (ℓ=0, γ=0, τ=0, merged δ, exponential value, no follower engine).
    Moved verbatim from the monolith sidebar body; runs after the L1/L2 controls set d.
    """
    # the base model (Level 1) is exact — later levels ADD mechanisms on top of it
    if LEVEL < 2:
        # L1 pays for the CURRENT model's compute: cost(t) = S0·10^{c^L(t)−c^L(0)}·10^{−g_p t}, so
        # cost(0) = S0 exactly. Training in advance (ℓ > 0) debuts at Level 2.
        d["ell"] = 0.0
    if LEVEL < 3:
        d["A1"] = True          # base model: algo progress at the constant rate g_a (exactly)
        d["gamma"] = 0.0        # (redundant under A1, kept for clarity)
        d["g_a"] = P0.g_a       # explicit pin, so every consumer reads d (never a default fallback)
    if LEVEL < 4:
        # levels 1-3: compute growth is CONSTANT at g_C0 (the slowdown debuts at Level 4). Pinning the
        # floor to today's rate makes g_c(t) = g_C0 for all t regardless of ξ, so the leader's speed is
        # constant and the base-model gap holds exactly at Δ0.
        d["g_C_inf"] = d["g_C0"]
    if LEVEL < 5:
        d["x_mid"] = X_MID_EXP  # levels 1-4: value is pure exponential W0·10^{νx} (saturation debuts at L5)
    if LEVEL < 6:
        # base-model follower = PURE catch-up: no engine of its own. Pinning its algo rate and its whole
        # compute path to zero makes c^F constant and ẋ^F = ȧ^F = δ·(x^L − x^F) exactly (the merged δ
        # routes through δ_rel on the full gap). The follower's own engine is introduced at Level 6.
        d["g_a_F"] = 0.0
        d["g_CF0"] = 0.0
        d["g_CF_inf"] = 0.0
    if LEVEL < 7:
        d["tau"] = PIN_TAU      # immediate release
    if LEVEL < 8:
        # base cost = the compute-path mechanism with NO R&D overhead: cost(t) = S0·10^{c^L(t+ℓ)−c^L(0)}·10^{−g_p t}
        # (anchor c^L(0), D-036 4th amendment). Level 8 ADDS the R&D overhead φ_RD (>0).
        d["phi_RD"] = PIN_PHI_RD
        d["g_p"] = P0.g_p       # explicit pin, so every consumer reads d (never a default fallback)
    d["dt"] = P0.dt
