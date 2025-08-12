# core/elo_config_cw.py

import random

##############################################
# CW Specific Constants
##############################################
# Max length for text passed to pairwise judge (from original CW elo.py)
LENGTH_TRUNCATION_CHARS = 4000

# Prompt IDs to ignore for ELO (from original CW elo.py)
# Note: Original CW file had this defined twice with different values.
# I'm using the second definition as it appeared later. Please verify which one is correct.
IGNORE_PROMPTS_FOR_ELO = [
    "5", "16", "20", "21", "26", "28", "30"
]
# Example of the first definition if needed:
# IGNORE_PROMPTS_FOR_ELO = [
# "119","45","124","123","125","208","212","210",
# "197","207","209","215","200","196","216","217"
# ]


##############################################
# Constants & Configuration (from EQB3, adapted for CW)
##############################################
# Default ELO rating for models with no prior data
DEFAULT_ELO = 1200.0

# For TrueSkill solver, determines how win margins are expanded into pseudo-wins
# This value was WIN_MARGIN_BIN_SIZE in EQB3, used by bin_fraction in trueskill_solver.
# CW's Glicko used BIN_SIZE = 4 for its bin_fraction.
# Let's use a similar concept for TrueSkill's bin_fraction.
# This value is passed to trueskill.bin_fraction's bin_size parameter.
# A smaller value means margins have less impact.
# A common value in EQB3 was 20, but for Glicko it was 4.
# Let's start with a value that might be more conservative like Glicko's.
# This will be used by the bin_fraction function inside trueskill_solver_cw.py
TRUESKILL_BIN_SIZE_FOR_WIN_EXPANSION = 4 # Default, can be tuned.
TRUESKILL_BIN_SIZE_FOR_CI_CALCULATION = 3 # For CI, typically smaller to be more conservative

# Rank window for final ELO solve (limits comparisons to models within this rank distance)
RANK_WINDOW = 16 # From EQB3, can be adjusted

# Sampling schedule for ELO calculation (from EQB3)
# Each tuple: ( (radii_to_sample_at_depth_1, 2, 3...), num_samples_for_closest_tier )
SAMPLING_SCHEDULE = [
    # stage‑1 – sparse anchors across the whole ladder
    ((None,), 10),  # (None,) means sample across whole ladder, 10 samples

    # stage‑2 – first zoom‑in
    ((1, 2, 3), 4),  # Sample 4 from rank +/-1, 2 from +/-2, 1 from +/-3
    ((1, 2, 3), 8),  # Sample 8 from rank +/-1, 4 from +/-2, 2 from +/-3
    ((1, 2, 3), 16), # Sample 16 from rank +/-1, 8 from +/-2, 4 from +/-3
    #((1, 2, 3), 32), # Sample 16 from rank +/-1, 8 from +/-2, 4 from +/-3

    # stage‑3 – comprehensive zoom (many samples from close neighbors)
    
    ((1, 2, 3), 48), # 9999 effectively means "all available" for CW's item/iteration structure
    #((1, 2, 3), 9999),
]
MAX_STAGE_LOOPS = 4  # Safety guard per stage (loops until rank stabilizes or max_loops)

# Random number generator for reproducible sampling if needed
rng = random.Random(42)

# Anchor models for normalization (from original CW elo.py)
# These are used by normalize_elo_scores in elo_cw.py
CW_ANCHOR_MODELS = {
    'deepseek/deepseek-r1': 1500,
    'meta-llama/llama-3.2-1b-instruct': 200
}

# TrueSkill environment parameters (can be tuned)
# mu: initial rating (set by DEFAULT_ELO)
# sigma: initial uncertainty (larger means more volatile ratings initially)
# beta: skill variance (how much skill is needed to have a high chance of winning)
# tau: dynamic factor (how much ratings can change over time, usually small or 0 for static benchmarks)
# draw_probability: set by TrueSkill library, but we handle draws by converting to win/loss sequences
TS_SIGMA = DEFAULT_ELO / 3  # Standard TrueSkill recommendation
TS_BETA = TS_SIGMA / 2      # Standard TrueSkill recommendation
TS_TAU = TS_SIGMA / 100   # Small dynamic factor, can be 0.0

# Flag to control if win margins are expanded to extra wins or if beta is adjusted
# EXPAND_MARGINS_TO_EXTRA_WINS = True is simpler and was the primary method in EQB3's TrueSkill.
# The alternative (adjusting beta) is more complex.
EXPAND_MARGINS_TO_EXTRA_WINS = True
# GAMMA is only used if EXPAND_MARGINS_TO_EXTRA_WINS is False
TS_GAMMA_FOR_BETA_ADJUSTMENT = 40.0