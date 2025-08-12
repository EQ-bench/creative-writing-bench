# core/matchup_selection_cw.py

import random
from typing import List, Tuple, Optional, Set

# Using elo_config_cw.rng for randomness if needed, but _pick_matchups is deterministic based on input.
from .elo_config_cw import rng as cw_rng

def _pick_matchups(
    my_rank: int,
    ladder_len: int,
    radius_tiers: Tuple[Optional[int], ...],
    samples_per_tier: int,
) -> List[int]:
    """
    Select opponent indices from a ladder based on rank and sampling parameters.
    (Directly from EQB3)
    """
    if ladder_len <= 1:
        return []

    # Stage 1: sample across the whole ladder
    if radius_tiers == (None,):
        if ladder_len <= samples_per_tier:
            # If ladder is small, include everyone except self
            return [i for i in range(ladder_len) if i != my_rank]

        # Sample 'samples_per_tier' unique opponents
        possible_opp_indices = [i for i in range(ladder_len) if i != my_rank]
        if len(possible_opp_indices) <= samples_per_tier:
            return possible_opp_indices
        return random.sample(possible_opp_indices, samples_per_tier)


    # Stage 2/3: sample from specific radius tiers
    selected_indices: Set[int] = set()
    for tier_idx, radius in enumerate(radius_tiers):
        if radius is None: continue # Should not happen with current SAMPLING_SCHEDULE for stage 2/3

        num_to_sample_this_tier = samples_per_tier
        if tier_idx == 1: # radius tier 2 (e.g., rank +/-2)
            num_to_sample_this_tier = max(1, samples_per_tier // 2)
        elif tier_idx >= 2: # radius tier 3+ (e.g., rank +/-3 and beyond)
            num_to_sample_this_tier = max(1, samples_per_tier // 4)

        tier_candidates: List[int] = []
        # Add candidates from above current rank
        if my_rank + radius < ladder_len:
            tier_candidates.append(my_rank + radius)
        # Add candidates from below current rank
        if my_rank - radius >= 0:
            tier_candidates.append(my_rank - radius)

        # Remove self and already selected candidates
        tier_candidates = [
            idx for idx in tier_candidates
            if idx != my_rank and idx not in selected_indices
        ]

        if not tier_candidates:
            continue

        # Sample from the unique candidates for this tier
        if len(tier_candidates) <= num_to_sample_this_tier:
            selected_indices.update(tier_candidates)
        else:
            selected_indices.update(random.sample(tier_candidates, num_to_sample_this_tier))

    return sorted(list(selected_indices))