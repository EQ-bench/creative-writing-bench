# core/elo_helpers_cw.py

import logging
import math
from typing import Dict, Any, List, Tuple, Optional, Set, Union

from .elo_config_cw import IGNORE_PROMPTS_FOR_ELO, RANK_WINDOW # RANK_WINDOW for filtering

# --- CW Specific Helpers (from original elo.py) ---

def should_ignore_prompt_cw(prompt_id: str) -> bool:
    """Check if prompt_id is in IGNORE_PROMPTS_FOR_ELO."""
    # CW's original logic for prompt_id format (e.g., "77_3_1" -> "77")
    # The current IGNORE_PROMPTS_FOR_ELO seems to contain base IDs.
    base_id = prompt_id.split("_")[0] if "_" in prompt_id else prompt_id
    return base_id in IGNORE_PROMPTS_FOR_ELO

def interpret_pairwise_result_cw(result_dict: Optional[Dict[str, str]]) -> Tuple[float, int, int]:
    """
    Return (outcome_for_A, plus_for_A, plus_for_B) in {0,0.5,1}, plus_for_A, plus_for_B as int tallies.
    This is unchanged from original CW elo.py.
    """
    if not result_dict:
        return 0.5, 0, 0

    a_score = 0
    b_score = 0
    for key, val in result_dict.items():
        if key in ["improvement_suggestions", "theory_of_mind", "_item_order_idx"]: # Added _item_order_idx
            continue
        # "A0493" => means model A is better for that dimension
        if "A0493" in val: # Model A preferred
            plus_count = val.count('+')
            # Original logic had subtractions for certain keys, which seems complex for simple plus counts.
            # Re-evaluating: The original code adds to a_score if A0493, and subtracts from b_score for specific keys.
            # This means for "avoids_poetic_overload", if A is better (A0493), A gets points, B loses points.
            # If B is better (A0488), B gets points, A loses points.
            # This double-counts the effect.
            # Let's simplify to: A gets points if A0493, B gets points if A0488.
            # The "punish these" logic seems to be an attempt to invert negative criteria directly in score.
            # For ELO, simpler plus counts are usually better. The "invert_if_negative" handles rubric scores.
            # Sticking to original interpretation for now:
            if plus_count > 0:
                a_score += plus_count
            if key in ["avoids_poetic_overload", "coherence", "avoids_verbosity"]: # Negative criteria
                 b_score -= plus_count # If A is good on negative, B is penalized
        elif "A0488" in val: # Model B preferred
            plus_count = val.count('+')
            if plus_count > 0:
                b_score += plus_count
            if key in ["avoids_poetic_overload", "coherence", "avoids_verbosity"]: # Negative criteria
                 a_score -= plus_count # If B is good on negative, A is penalized

    if a_score > b_score:
        return 1.0, a_score, b_score
    elif b_score > a_score:
        return 0.0, a_score, b_score
    else:
        return 0.5, a_score, b_score

def custom_blend_cw(x: float, linear_gradient=5, sigmoid_power=0.75, transition_start=0.0, transition_end=0.11) -> float:
    """
    Transforms a value in [0,1] by blending a linear slope with a sigmoid curve
    around [transition_start..transition_end].
    From original CW elo.py. User confirmed this should be x.
    """
    # Per user direction, this function should just return x.
    # Keeping the original signature if it needs to be reverted.
    # x = max(0.0, min(1.0, x))
    # if x <= transition_start: blend = 0.0
    # elif x >= transition_end: blend = 1.0
    # else:
    #     t = (x - transition_start)/(transition_end - transition_start)
    #     blend = t*t*(3-2*t) # smoothstep
    # linear_val = linear_gradient * x
    # k = 3 # sigmoid steepness
    # sig_val = (1.0 - math.exp(-k * (x**sigmoid_power))) / (1.0 - math.exp(-k)) if (1.0 - math.exp(-k)) != 0 else (1.0 if x > 0 else 0.0)
    # return (1.0 - blend)*linear_val + blend*sig_val
    return x


def compute_fraction_for_test_cw(outcome_for_test: float, plus_for_test: int, plus_for_other: int) -> Tuple[float, int, float, float]:
    """
    Calculates fraction_for_test based on CW's margin logic.
    Returns: (final_fraction, diff, diff_norm, diff_blend)
    From original CW elo.py.
    """
    diff = abs(plus_for_test - plus_for_other)
    # Original had diff/45.0. Max possible plus_diff needs to be determined if not 45.
    # Assuming 45 is a reasonable upper bound for normalization.
    diff_norm = diff / 45.0
    diff_blend = custom_blend_cw(diff_norm) # Uses the simplified custom_blend
    margin = diff_blend / 2.0 + 0.5  # [0.5..1.0]

    if outcome_for_test == 0.5:
        final_fraction = 0.5
    elif outcome_for_test == 1.0:
        final_fraction = margin
    else:  # outcome_for_test == 0.0
        final_fraction = 1.0 - margin

    return final_fraction, diff, diff_norm, diff_blend

# --- EQB3-style Helpers (adapted for CW) ---

def create_matchup_signature_cw(
    test_model: str,
    neighbor_model: str,
    item_id: str,
    test_model_iteration_id: str,
    neighbor_model_iteration_id: str
) -> str:
    """
    Creates a unique string signature for a CW matchup, including iterations.
    Order of models is normalized (sorted) to ensure M1vsM2 is same as M2vsM1.
    """
    # Sort the (model, iteration_id) pairs to ensure consistent ordering
    # The result of sorted() will be a list of two tuples:
    # e.g., [('modelA_name', 'iterA_id'), ('modelB_name', 'iterB_id')]
    # if modelA_name comes before modelB_name alphabetically, or if they are the same
    # and iterA_id comes before iterB_id.
    sorted_model_iter_pairs = sorted([
        (test_model, test_model_iteration_id),
        (neighbor_model, neighbor_model_iteration_id)
    ])

    # Unpack the sorted pairs
    model1_name, model1_iter_id = sorted_model_iter_pairs[0]
    model2_name, model2_iter_id = sorted_model_iter_pairs[1]
    
    return f"{model1_name}|{model1_iter_id}|{model2_name}|{model2_iter_id}|{item_id}"

def build_existing_matchup_set_cw(comparisons: List[Dict[str, Any]]) -> Set[str]:
    """
    Builds a set of existing matchup signatures from a list of CW comparison dicts.
    """
    existing_matchups: Set[str] = set()
    for comp in comparisons:
        pair = comp.get("pair")
        item_id = comp.get("item_id")
        if pair and item_id:
            test_model = pair.get("test_model")
            neighbor_model = pair.get("neighbor_model")
            # Iteration IDs are crucial for CW
            test_iter_id = pair.get("test_model_iteration_id", "UNKNOWN_ITER_TEST") # Provide default if missing
            neigh_iter_id = pair.get("neighbor_model_iteration_id", "UNKNOWN_ITER_NEIGH")

            if test_model and neighbor_model:
                sig = create_matchup_signature_cw(
                    test_model, neighbor_model, item_id, test_iter_id, neigh_iter_id
                )
                existing_matchups.add(sig)
    return existing_matchups

def update_existing_matchups_from_comparisons_cw(
    new_comparisons: List[Dict[str, Any]],
    existing_matchups_set: Set[str]
) -> int:
    """
    Adds signatures of new_comparisons to existing_matchups_set.
    Returns the count of newly added unique signatures.
    """
    added_count = 0
    for comp in new_comparisons:
        if "error" in comp: # Don't add errors to the "judged" set
            continue
        pair = comp.get("pair")
        item_id = comp.get("item_id")
        if pair and item_id:
            test_model = pair.get("test_model")
            neighbor_model = pair.get("neighbor_model")
            test_iter_id = pair.get("test_model_iteration_id", "UNKNOWN_ITER_TEST")
            neigh_iter_id = pair.get("neighbor_model_iteration_id", "UNKNOWN_ITER_NEIGH")

            if test_model and neighbor_model:
                sig = create_matchup_signature_cw(
                    test_model, neighbor_model, item_id, test_iter_id, neigh_iter_id
                )
                if sig not in existing_matchups_set:
                    existing_matchups_set.add(sig)
                    added_count += 1
    return added_count


def _is_valid_comp_cw(c: Dict[str, Any]) -> bool:
    """Return True if comparison *c* is usable by the CW solver."""
    return (
        "error" not in c
        and not should_ignore_prompt_cw(c.get("item_id", "")) # Use CW's ignore logic
        and c.get("pair", {}).get("test_model")
        and c.get("pair", {}).get("neighbor_model")
        and "fraction_for_test" in c # Crucial for solver
    )

def filter_comparisons_for_solver_cw(comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Basic validity filter (ignores rank window)."""
    return [c for c in comps if _is_valid_comp_cw(c)]

def filter_comps_within_rank_window_cw(
    comps: List[Dict[str, Any]],
    elo_snapshot: Dict[str, float], # {model_name: elo_rating}
    window: int,
) -> List[Dict[str, Any]]:
    """
    Keep only comps where the two models are <= *window* ladder positions apart.
    *elo_snapshot* is a dict {model: rating}.
    """
    if not elo_snapshot: # Avoid error if snapshot is empty
        return comps

    # Sort models by ELO to get ladder positions
    # Models not in elo_snapshot are effectively at a very low/high rank or filtered out
    ladder = sorted(elo_snapshot.keys(), key=lambda m: elo_snapshot.get(m, -float('inf')))
    pos = {m: i for i, m in enumerate(ladder)}

    def _ok(pair: Dict[str, Any]) -> bool:
        a = pair.get("test_model")
        b = pair.get("neighbor_model")
        # Only consider if both models are in the current ELO snapshot (and thus the ladder)
        if a in pos and b in pos:
            return abs(pos[a] - pos[b]) <= window
        return False # If one model isn't in snapshot, can't determine rank diff

    return [c for c in comps if _ok(c.get("pair", {}))]

def get_solver_comparisons_cw(
    comps: List[Dict[str, Any]],
    elo_snapshot: Optional[Dict[str, float]] = None,
    rank_window_override: Optional[int] = None, # Allow overriding global RANK_WINDOW
) -> List[Dict[str, Any]]:
    """
    1. Applies the basic validity filter for CW.
    2. If *rank_window_override* or global RANK_WINDOW is given, applies the ±window filter.
    """
    current_rank_window = rank_window_override if rank_window_override is not None else RANK_WINDOW

    valid_comps = filter_comparisons_for_solver_cw(comps)
    if current_rank_window is not None and elo_snapshot is not None and valid_comps:
        return filter_comps_within_rank_window_cw(valid_comps, elo_snapshot, current_rank_window)
    return valid_comps

def models_in_comparisons_cw(comps: List[Dict[str, Any]]) -> Set[str]:
    """Return the set of model names present in *comps*."""
    mods: Set[str] = set()
    for c in comps:
        p = c.get("pair", {})
        if p.get("test_model"):    mods.add(p["test_model"])
        if p.get("neighbor_model"): mods.add(p["neighbor_model"])
    return mods

def recompute_fractions_for_comparisons_cw(comparisons: List[Dict[str, Any]]) -> int:
    """
    Recomputes 'fraction_for_test' and related fields for all comparisons in place.
    Uses CW's `compute_fraction_for_test_cw`.
    Returns the number of comparisons changed.
    """
    changed_count = 0
    for comp in comparisons:
        if "error" in comp:
            continue

        outcome = comp.get("outcome_for_test_model")
        plus_test = comp.get("plus_for_test")
        plus_other = comp.get("plus_for_other")
        old_fraction = comp.get("fraction_for_test")

        if outcome is not None and plus_test is not None and plus_other is not None:
            new_fraction, diff, diff_norm, diff_blend = compute_fraction_for_test_cw(
                outcome, plus_test, plus_other
            )
            if new_fraction != old_fraction:
                changed_count +=1
            comp["fraction_for_test"] = new_fraction
            comp["plus_diff"] = diff
            comp["plus_diff_normalized"] = diff_norm
            comp["plus_diff_blended"] = diff_blend
        elif "fraction_for_test" not in comp : # if essential fields are missing and no fraction exists
             logging.warning(f"Cannot recompute fraction for comp due to missing fields (outcome/plus_counts) and no existing fraction: {comp.get('item_id')}")


    if changed_count > 0:
        logging.info(f"[ELO-CW] Recomputed fractions for {changed_count}/{len(comparisons)} comparisons.")
    return changed_count