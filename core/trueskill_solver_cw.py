# core/trueskill_solver_cw.py

import logging
import math
import trueskill # type: ignore
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from collections import defaultdict

from .elo_config_cw import (
    DEFAULT_ELO,
    TRUESKILL_BIN_SIZE_FOR_WIN_EXPANSION, # Used as default bin_size
    TS_SIGMA,
    TS_BETA,
    TS_TAU,
    EXPAND_MARGINS_TO_EXTRA_WINS,
    TS_GAMMA_FOR_BETA_ADJUSTMENT
)
# CW's compute_fraction_for_test will be used externally to prepare comparisons
# but this solver needs its own bin_fraction for win expansion.

# This is CW's function, brought here for direct use if needed by _fraction_from_plus_cw
# However, the primary path is that comparisons already have 'fraction_for_test'
from .elo_helpers_cw import compute_fraction_for_test_cw


def _fraction_from_plus_cw(group: List[Dict[str, Any]],
                           m1: str,
                           m2: str) -> Optional[float]:
    """
    Aggregate plus‑counts for the two models across the forward/reverse
    comparisons in *group*, then compute fraction_for_test (for m1)
    using CW's logic.
    This is adapted from EQB3's _fraction_from_plus.
    """
    plus_m1 = 0
    plus_m2 = 0
    # Check if all comparisons in the group have the required plus counts
    # and are from the same logical matchup (ignoring who was test/neighbor initially)
    for comp in group:
        if "plus_for_test" not in comp or "plus_for_other" not in comp:
            # logging.debug(f"Comparison missing plus_for_test/plus_for_other: {comp.get('item_id')}")
            return None  # Can't use this rule if data is missing

        pair = comp.get("pair", {})
        comp_test_model = pair.get("test_model")
        # comp_neighbor_model = pair.get("neighbor_model") # Not strictly needed for this logic

        if comp_test_model == m1: # m1 was test_model in this specific comparison
            plus_m1 += comp["plus_for_test"]
            plus_m2 += comp["plus_for_other"]
        elif comp_test_model == m2: # m2 was test_model in this specific comparison
            plus_m2 += comp["plus_for_test"]
            plus_m1 += comp["plus_for_other"]
        else:
            # This comparison in the group doesn't involve m1 or m2 as test_model directly
            # This shouldn't happen if grouping is correct (m1, m2 are the group key models)
            logging.warning(f"Unexpected model pair in _fraction_from_plus_cw group. Group for ({m1}, {m2}), comp pair ({comp_test_model}, {pair.get('neighbor_model')})")
            return None


    if plus_m1 > plus_m2:
        outcome = 1.0
    elif plus_m1 < plus_m2:
        outcome = 0.0
    else:
        outcome = 0.5

    # Use CW's compute_fraction_for_test
    frac, _, _, _ = compute_fraction_for_test_cw(outcome, plus_m1, plus_m2)
    return frac


def bin_fraction_trueskill(frac: float, bin_size: int) -> Tuple[int, int]:
    """
    Convert fraction_for_test into asymmetric pseudo-match counts for TrueSkill.
    A fraction of 1.0 means test model wins `bin_size` times, other model 0 times.
    A fraction of 0.0 means test model wins 0 times, other model `bin_size` times.
    A fraction of 0.5 means test model wins 1 time, other model 1 time (effectively a draw for TrueSkill if rate_1vs1 drawn=True is used, or 1 win each).
    This version is from EQB3.
    """
    frac = max(0.0, min(1.0, frac))
    eps = 1e-9

    if abs(frac - 0.5) < eps: # Exact draw
        return 1, 1 # Represents one draw, or one win each if not using drawn=True

    # Test-model win
    if frac > 0.5:
        # Scale margin [0, 0.5] to [0, bin_size-1], then add 1 for the base win
        # margin = frac - 0.5
        # wins_test = 1 + math.floor(margin * 2 * (bin_size -1)) # Ensure at least 1 win
        # wins_other = 0
        # A simpler scaling:
        # Map frac (0.5, 1.0] to wins_test [ceil(bin_size/2) ... bin_size]
        # and wins_other [floor(bin_size/2) ... 0]
        # For simplicity and consistency with Glicko binning:
        # Let's use a linear scaling of the fraction to wins.
        # total_matches = bin_size # if frac = 1.0, test gets bin_size wins.
        # wins_test = round(frac * bin_size)
        # wins_other = bin_size - wins_test
        # This is too simple. Let's use the EQB3 version:
        step = 0.5 / bin_size        # width of each chunk for margin
        margin = frac - 0.5
        wins_test = max(1, min(bin_size, math.ceil(margin / step)))
        wins_other = 0
        return int(wins_test), int(wins_other)

    # Test-model loss (frac < 0.5)
    step = 0.5 / bin_size
    margin = 0.5 - frac
    wins_other = max(1, min(bin_size, math.ceil(margin / step)))
    wins_test = 0
    return int(wins_test), int(wins_other)


def solve_with_trueskill_cw(
    all_models: List[str],
    pairwise_comparisons: List[Dict[str, Any]],
    initial_ratings: Dict[str, float], # Mu values
    debug: bool = False,
    use_fixed_initial_ratings: bool = True, # If True, active models start at DEFAULT_ELO
    bin_size_override: Optional[int] = None, # Allows overriding global bin_size
    return_sigma: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Calculates ratings using TrueSkill for Creative Writing.
    - Comparison objects must have 'item_id', 'pair': {'test_model', 'neighbor_model'},
      'fraction_for_test'.
    - They should also have 'test_model_iteration_id' and 'neighbor_model_iteration_id'
      within the 'pair' dict or at the top level of the comparison object for unique grouping.
    """
    current_bin_size = bin_size_override if bin_size_override is not None else TRUESKILL_BIN_SIZE_FOR_WIN_EXPANSION

    # --- Pre-processing: Group comparisons for the same logical matchup (item_id, model1, model2, iter1, iter2) ---
    grouped_comparisons = defaultdict(list)
    for c in pairwise_comparisons:
        if "error" in c or "pair" not in c or "item_id" not in c:
            continue

        pair = c.get("pair", {})
        test_model = pair.get("test_model")
        neighbor_model = pair.get("neighbor_model")
        item_id = c.get("item_id")

        # Iteration IDs are crucial for CW to define a unique matchup instance
        # Assuming they are stored in the comparison object, e.g., in pair or top-level
        # For this example, let's assume they are in the pair dict.
        # If not, this part needs adjustment based on where they are stored.
        test_iter_id = pair.get("test_model_iteration_id", "UNKNOWN_ITER_TEST")
        neigh_iter_id = pair.get("neighbor_model_iteration_id", "UNKNOWN_ITER_NEIGH")

        if not all([test_model, neighbor_model, item_id]):
            continue
        if test_model == neighbor_model and test_iter_id == neigh_iter_id : # Self-comparison with same iteration
            continue

        # Sort model names and their iteration IDs together for consistent grouping
        # This ensures (m1,i1) vs (m2,i2) is grouped the same as (m2,i2) vs (m1,i1)
        model1_key_part = tuple(sorted([(test_model, test_iter_id), (neighbor_model, neigh_iter_id)]))
        group_key = (model1_key_part[0][0], model1_key_part[0][1], # model A, iter A
                       model1_key_part[1][0], model1_key_part[1][1], # model B, iter B
                       item_id)
        grouped_comparisons[group_key].append(c)

    if debug:
        logging.debug(f"[TrueSkill-CW] Original comparisons: {len(pairwise_comparisons)}")
        logging.debug(f"[TrueSkill-CW] Grouped into {len(grouped_comparisons)} logical matchups.")

    # --- Process grouped comparisons to get one fraction per logical matchup ---
    processed_paired_comparisons: List[Dict[str, Any]] = []
    for group_key, group_list in grouped_comparisons.items():
        # Unpack group key
        m1_name, m1_iter, m2_name, m2_iter, item_id_key = group_key

        frac = None
        # Attempt 1: Use _fraction_from_plus_cw if plus counts are reliable and consistent in the group
        # This requires all items in group_list to have plus_for_test and plus_for_other
        if all("plus_for_test" in c and "plus_for_other" in c for c in group_list):
            frac = _fraction_from_plus_cw(group_list, m1_name, m2_name)

        # Attempt 2: Average 'fraction_for_test' if available
        if frac is None:
            fractions_in_group = []
            for c in group_list:
                if "fraction_for_test" in c:
                    # Ensure fraction is relative to m1_name
                    if c.get("pair", {}).get("test_model") == m1_name:
                        fractions_in_group.append(c["fraction_for_test"])
                    elif c.get("pair", {}).get("test_model") == m2_name:
                        fractions_in_group.append(1.0 - c["fraction_for_test"])
                    # Else: comparison doesn't match m1/m2 as test_model, skip (should not happen)
            if fractions_in_group:
                frac = sum(fractions_in_group) / len(fractions_in_group)
            elif debug and not fractions_in_group:
                 logging.warning(f"[TrueSkill-CW] Group {group_key} has no usable 'fraction_for_test'. Count: {len(group_list)}")


        if frac is not None:
            processed_paired_comparisons.append({
                "item_id": item_id_key,
                "pair": {
                    "test_model": m1_name, "neighbor_model": m2_name,
                    "test_model_iteration_id": m1_iter, "neighbor_model_iteration_id": m2_iter
                },
                "fraction_for_test": frac,
                # Include original comparisons if needed for debugging, but not strictly necessary for solver
                # "_original_group_size": len(group_list)
            })
        elif debug:
            logging.warning(f"[TrueSkill-CW] Could not determine fraction for group {group_key}. Skipping.")


    active_models: Set[str] = set()
    for c in processed_paired_comparisons:
        active_models.add(c["pair"]["test_model"])
        active_models.add(c["pair"]["neighbor_model"])

    logging.info(
        f"[TrueSkill-CW] Solver received {len(processed_paired_comparisons)} unique paired comparisons "
        f"for {len(active_models)} active models (out of {len(all_models)} total)."
    )

    # --- Setup TrueSkill Environment and Players ---
    ts_env = trueskill.TrueSkill(mu=DEFAULT_ELO, sigma=TS_SIGMA, beta=TS_BETA,
                                 tau=TS_TAU, draw_probability=0.0) # draw_probability=0 as we expand draws

    ratings: Dict[str, trueskill.Rating] = {}
    for m_name in all_models:
        start_mu = DEFAULT_ELO
        if m_name in active_models and use_fixed_initial_ratings:
            start_mu = DEFAULT_ELO # Reset active models to default for a fresh solve
        elif m_name in initial_ratings and not use_fixed_initial_ratings :
            start_mu = initial_ratings[m_name]
        elif m_name in initial_ratings and use_fixed_initial_ratings and m_name not in active_models:
            # Model is not active in current comparisons but has a prior rating
            start_mu = initial_ratings[m_name]

        ratings[m_name] = ts_env.Rating(mu=start_mu, sigma=TS_SIGMA) # Use global TS_SIGMA for initial uncertainty


    # --- Apply TrueSkill updates ---
    if not EXPAND_MARGINS_TO_EXTRA_WINS:
        # Method: Adjust beta based on margin (more complex)
        base_beta = ts_env.beta
        env_cache: Dict[float, trueskill.TrueSkill] = {}
        logging.info(f"[TrueSkill-CW] Applying margin-weighted updates by adjusting beta (GAMMA={TS_GAMMA_FOR_BETA_ADJUSTMENT}, base_beta={base_beta:.2f}). Bin size for win/loss: {current_bin_size}")

        for comp in processed_paired_comparisons:
            frac = comp["fraction_for_test"]
            m1 = comp["pair"]["test_model"]
            m2 = comp["pair"]["neighbor_model"]

            if m1 not in ratings or m2 not in ratings or m1 == m2:
                continue

            margin_signal = abs(frac - 0.5) * 2.0  # m in [0,1]
            k_eff_matches = 1.0 + TS_GAMMA_FOR_BETA_ADJUSTMENT * margin_signal
            beta_eff = base_beta / math.sqrt(k_eff_matches)

            current_ts_env = env_cache.get(beta_eff)
            if current_ts_env is None:
                current_ts_env = trueskill.TrueSkill(mu=DEFAULT_ELO, sigma=TS_SIGMA, beta=beta_eff, tau=TS_TAU, draw_probability=0.0)
                env_cache[beta_eff] = current_ts_env

            # Use simple win/loss based on frac for this method, beta carries margin
            if frac > 0.5 + 1e-9: # m1 wins
                ratings[m1], ratings[m2] = current_ts_env.rate_1vs1(ratings[m1], ratings[m2])
            elif frac < 0.5 - 1e-9: # m2 wins
                ratings[m2], ratings[m1] = current_ts_env.rate_1vs1(ratings[m2], ratings[m1])
            else: # Draw
                ratings[m1], ratings[m2] = current_ts_env.rate_1vs1(ratings[m1], ratings[m2], drawn=True)

    else: # EXPAND_MARGINS_TO_EXTRA_WINS = True (simpler, preferred)
        logging.info(f"[TrueSkill-CW] Applying updates by expanding margins to pseudo-wins (bin_size={current_bin_size})")
        for comp in processed_paired_comparisons:
            frac = comp["fraction_for_test"]
            m1 = comp["pair"]["test_model"]
            m2 = comp["pair"]["neighbor_model"]

            if m1 not in ratings or m2 not in ratings or m1 == m2:
                continue

            wins_m1, wins_m2 = bin_fraction_trueskill(frac, current_bin_size)

            try:
                if wins_m1 == 1 and wins_m2 == 1: # Interpreted as a draw from bin_fraction
                    r_m1, r_m2 = ts_env.rate_1vs1(ratings[m1], ratings[m2], drawn=True)
                    ratings[m1], ratings[m2] = r_m1, r_m2
                else:
                    for _ in range(wins_m1):
                        r_m1, r_m2 = ts_env.rate_1vs1(ratings[m1], ratings[m2])
                        ratings[m1], ratings[m2] = r_m1, r_m2
                    for _ in range(wins_m2):
                        r_m2, r_m1 = ts_env.rate_1vs1(ratings[m2], ratings[m1])
                        ratings[m1], ratings[m2] = r_m1, r_m2 # Assign back correctly
            except ValueError as e:
                logging.warning(f"[TrueSkill-CW] Update failed between {m1} and {m2}: {e}. Frac: {frac}, Wins: ({wins_m1},{wins_m2}). Skipping.")
            except Exception as e:
                logging.error(f"[TrueSkill-CW] Unexpected error during update ({m1} vs {m2}): {e}", exc_info=True)


    # --- Collect final ratings ---
    final_mu_map: Dict[str, float] = {m_name: r.mu for m_name, r in ratings.items()}
    final_sigma_map: Dict[str, float] = {m_name: r.sigma for m_name, r in ratings.items()}

    if debug:
        for m_name in sorted(final_mu_map.keys(), key=lambda k: final_mu_map[k], reverse=True):
            logging.debug(f"[TrueSkill-CW] {m_name:<30}: Mu={final_mu_map[m_name]:.2f}, Sigma={final_sigma_map[m_name]:.2f}")

    if return_sigma:
        return final_mu_map, final_sigma_map
    return final_mu_map