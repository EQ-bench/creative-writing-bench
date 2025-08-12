# CW bench elo.py:

# core/elo_cw.py

import os
import logging
import json
from pathlib import Path
import random # For _pick_matchups if not using cw_rng explicitly
import copy # For deep copying
from typing import Dict, Any, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone # For timestamps
from collections import defaultdict

from utils.file_io import load_json_file, save_json_file # Assuming CW has this

# Import from new CW-specific ELO modules
from .elo_config_cw import (
    DEFAULT_ELO,
    SAMPLING_SCHEDULE,
    MAX_STAGE_LOOPS,
    TRUESKILL_BIN_SIZE_FOR_WIN_EXPANSION,
    TRUESKILL_BIN_SIZE_FOR_CI_CALCULATION,
    RANK_WINDOW,
    CW_ANCHOR_MODELS,
    # Constants from original CW elo.py that are now in elo_config_cw
    LENGTH_TRUNCATION_CHARS,
    TS_SIGMA
)
from .elo_helpers_cw import (
    should_ignore_prompt_cw,
    interpret_pairwise_result_cw,
    compute_fraction_for_test_cw,
    create_matchup_signature_cw,
    build_existing_matchup_set_cw,
    update_existing_matchups_from_comparisons_cw,
    get_solver_comparisons_cw,
    models_in_comparisons_cw,
    recompute_fractions_for_comparisons_cw # For re-calculating stored fractions
)
from .trueskill_solver_cw import solve_with_trueskill_cw
from .matchup_selection_cw import _pick_matchups

# --- Original CW Helper Functions (to be kept and used) ---
# These are used in data preparation or the core judging logic

def invert_if_negative(metric: str, val: float, neg_list: List[str]) -> float:
    """ From original CW elo.py. """
    if metric in neg_list:
        return 20.0 - val # Assuming 0-10 scale, so 20.0 makes sense for 0-20 scale. Adjust if scale is different.
    return val

def deduplicate_comparisons_cw(comps: List[Dict[str, Any]], model_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Takes a list of comparison dicts and returns a new list with duplicates removed.
    Duplicates are identified by a signature of the comparison content.
    If model_name_filter is provided, only keeps comparisons where test_model matches.
    (Adapted from original CW elo.py)
    """
    seen_signatures = set()
    unique_comps = []
    for c in comps:
        if model_name_filter and c.get("pair", {}).get("test_model") != model_name_filter:
            continue
        # Create a robust signature for deduplication
        # Using item_id, model pair (sorted), and iteration IDs, and fraction for exactness
        pair = c.get("pair", {})
        item_id = c.get("item_id")
        test_model_comp = pair.get("test_model") # Renamed to avoid conflict with outer scope test_model
        neighbor_model = pair.get("neighbor_model")
        test_iter_id = pair.get("test_model_iteration_id", "N/A")
        neigh_iter_id = pair.get("neighbor_model_iteration_id", "N/A")
        fraction = round(c.get("fraction_for_test", -1.0), 4) # Round for float precision

        if not all([item_id, test_model_comp, neighbor_model]):
            # Fallback to JSON dump if essential parts for signature are missing
            sig_content = {k: v for k, v in c.items() if k not in ['judge_response']} # Exclude verbose fields
            sig = json.dumps(sig_content, sort_keys=True)
        else:
            m1_sig, m2_sig = tuple(sorted([(test_model_comp, test_iter_id), (neighbor_model, neigh_iter_id)]))
            sig = f"{item_id}|{m1_sig[0]}|{m1_sig[1]}|{m2_sig[0]}|{m2_sig[1]}|{fraction}"

        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_comps.append(c)
    return unique_comps


def do_pairwise_judge_cw( # Renamed to avoid conflict if other do_pairwise_judge exists
    textA: str,
    textB: str,
    prompt_id: str, # This is item_id in CW context
    pairwise_prompt_template: str,
    writing_prompts: Dict[str, Any], # {item_id: {"writing_prompt": "..."}}
    judge_model: str,
    api_clients: Dict[str, Any], # {"judge": client_object}
    # item_order_idx=None # Original CW had this, seems for ordering within a batch
):
    """ Core judging function from original CW elo.py. """
    # If prompt_id is something like "77_3_1", extract the actual prompt ID part ("77")
    # This logic is for accessing writing_prompts, which uses base IDs.
    raw_prompt_id = prompt_id.split("_", 1)[0] if "_" in prompt_id else prompt_id

    if raw_prompt_id not in writing_prompts:
        logging.error(f"[Judge-CW] Writing prompt for raw_prompt_id '{raw_prompt_id}' (from item_id '{prompt_id}') not found.")
        return {"error": f"Writing prompt for {raw_prompt_id} not found"}

    writing_prompt_content = writing_prompts[raw_prompt_id]["writing_prompt"]

    final_prompt = pairwise_prompt_template.replace("{writing_prompt}", writing_prompt_content)
    final_prompt = final_prompt.replace("{model_a_analysis}", textA)
    final_prompt = final_prompt.replace("{model_b_analysis}", textB)
    response_text = ""
    try:
        # Assuming api_clients["judge"] has a .generate method
        response_text = api_clients["judge"].generate(
            judge_model, final_prompt, temperature=0.0, max_tokens=16000, # Original CW params
            # include_seed=True, min_p=None # Original CW params, adapt if your client differs
        )
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logging.warning(f"[Judge-CW] Malformed JSON response for item {prompt_id}:\n{response_text}")
            return {"error": "Malformed JSON response from judge model"}
        json_str = response_text[start:end + 1]
        result = json.loads(json_str)
        # if item_order_idx is not None: result["_item_order_idx"] = item_order_idx # If needed
        return result
    except Exception as e:
        logging.error(f"[Judge-CW] Pairwise judge API error for item {prompt_id} with judge {judge_model}: {e}", exc_info=True)
        logging.debug(f"[Judge-CW] Failing response content:\n{response_text}")
        return {"error": f"API error: {str(e)}"}


def _judge_item_iteration_pairs_in_parallel_cw(
    test_model_name: str,
    neighbor_model_name: str,
    # List of tuples: (item_id, test_iter_id, test_item_text, test_item_score, neigh_iter_id, neigh_item_text, neigh_item_score)
    matchups_to_judge: List[Tuple[str, str, str, float, str, str, float]],
    pairwise_prompt_template: str,
    writing_prompts: Dict[str, Any],
    judge_model: str,
    api_clients: Dict[str, Any],
    max_workers: int,
) -> List[Dict[str, Any]]:
    """
    Judges specific item-iteration pairs in parallel.
    This adapts CW's _judge_items_in_parallel to operate on pre-selected item-iteration pairs.
    Returns a list of comparison result dictionaries.
    """
    comparisons_results: List[Dict[str, Any]] = []
    
    # Cap workers to avoid overwhelming APIs or system resources
    # Original CW used 500, which might be too high for many setups.
    # Let's use the passed max_workers, which will be derived from concurrency.
    num_workers = min(len(matchups_to_judge) * 2, max_workers) if matchups_to_judge else 0 # x2 for fwd/rev

    if num_workers == 0: # No matchups to judge or no workers allowed
        return []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_matchup_info: Dict[Any, Dict[str, Any]] = {}

        for item_id, test_iter_id, test_text, test_score, neigh_iter_id, neigh_text, neigh_score in matchups_to_judge:
            # Truncate texts
            textA = test_text[:LENGTH_TRUNCATION_CHARS]
            textB = neigh_text[:LENGTH_TRUNCATION_CHARS]

            match_info_base = {
                "item_id": item_id,
                "test_model_name": test_model_name, "test_iter_id": test_iter_id,
                "neighbor_model_name": neighbor_model_name, "neigh_iter_id": neigh_iter_id,
                "len_A": len(textA), "len_B": len(textB), # Original lengths before truncation for info
                "score_A": test_score, "score_B": neigh_score
            }

            # Forward: (test_model_name vs neighbor_model_name)
            fwd_future = executor.submit(
                do_pairwise_judge_cw,
                textA, textB, item_id,
                pairwise_prompt_template, writing_prompts,
                judge_model, api_clients
            )
            future_to_matchup_info[fwd_future] = {**match_info_base, "direction": "forward"}

            # Reverse: (neighbor_model_name vs test_model_name)
            rev_future = executor.submit(
                do_pairwise_judge_cw,
                textB, textA, item_id, # Swapped texts
                pairwise_prompt_template, writing_prompts,
                judge_model, api_clients
            )
            future_to_matchup_info[rev_future] = {**match_info_base, "direction": "reversed"}

        for future in as_completed(future_to_matchup_info):
            match_info = future_to_matchup_info[future]
            item_id = match_info["item_id"]
            current_test_model = match_info["test_model_name"] # The overall test_model for this ELO run
            current_neigh_model = match_info["neighbor_model_name"] # The overall neighbor for this ELO run
            
            # Iteration IDs for this specific comparison
            comp_test_iter_id = match_info["test_iter_id"]
            comp_neigh_iter_id = match_info["neigh_iter_id"]

            try:
                judge_api_result = future.result()

                comp_entry: Dict[str, Any] = {
                    "item_id": item_id,
                    "pair": {
                        # These are the models involved in THIS specific comparison instance
                        "test_model": current_test_model, # This will be consistent for forward
                        "neighbor_model": current_neigh_model, # Consistent for forward
                        "test_model_iteration_id": comp_test_iter_id,
                        "neighbor_model_iteration_id": comp_neigh_iter_id,
                    },
                    "judge_response": judge_api_result, # Keep the raw judge response
                    "item_length": { # Store truncated lengths used in judgement
                        "test_model": match_info["len_A"] if match_info["direction"] == "forward" else match_info["len_B"],
                        "neighbor_model": match_info["len_B"] if match_info["direction"] == "forward" else match_info["len_A"],
                    },
                    "creative_writing_rubric_scores": { # Rubric scores of the compared iterations
                        current_test_model: match_info["score_A"],
                        current_neigh_model: match_info["score_B"],
                    }
                }

                if "error" in judge_api_result:
                    comp_entry["error"] = judge_api_result["error"]
                    # Fill with neutral values if error, but still log the attempt
                    comp_entry["outcome_for_test_model"] = 0.5
                    comp_entry["plus_for_test"] = 0
                    comp_entry["plus_for_other"] = 0
                    comp_entry["fraction_for_test"] = 0.5
                else:
                    if match_info["direction"] == "forward":
                        # A0493 is test_model (A), A0488 is neighbor_model (B)
                        outcome, plus_A, plus_B = interpret_pairwise_result_cw(judge_api_result)
                        comp_entry["order"] = "A0493:test / A0488:other"
                        comp_entry["outcome_for_test_model"] = outcome
                        comp_entry["plus_for_test"] = plus_A
                        comp_entry["plus_for_other"] = plus_B
                    else: # Reversed
                        # A0493 is neighbor_model (A), A0488 is test_model (B)
                        outcome_for_judged_A, plus_judged_A, plus_judged_B = interpret_pairwise_result_cw(judge_api_result)
                        comp_entry["order"] = "A0493:other / A0488:test" # Original test_model is "other" here

                        # Convert outcome back relative to original test_model
                        if outcome_for_judged_A == 1.0: # neighbor_model won
                            comp_entry["outcome_for_test_model"] = 0.0
                        elif outcome_for_judged_A == 0.0: # test_model won
                            comp_entry["outcome_for_test_model"] = 1.0
                        else: # Draw
                            comp_entry["outcome_for_test_model"] = 0.5
                        
                        comp_entry["plus_for_test"] = plus_judged_B # plus for original test_model
                        comp_entry["plus_for_other"] = plus_judged_A # plus for original neighbor_model

                    # Compute fraction for test model based on its outcome and plus counts
                    frac, diff, diff_norm, diff_blend = compute_fraction_for_test_cw(
                        comp_entry["outcome_for_test_model"],
                        comp_entry["plus_for_test"],
                        comp_entry["plus_for_other"]
                    )
                    comp_entry["plus_diff"] = diff
                    comp_entry["plus_diff_normalized"] = diff_norm
                    comp_entry["plus_diff_blended"] = diff_blend
                    comp_entry["fraction_for_test"] = frac
                
                comparisons_results.append(comp_entry)

            except Exception as e:
                logging.error(f"[Judge-CW] Error processing result for item {item_id}, direction {match_info['direction']}: {e}", exc_info=True)
                # Create an error entry
                comparisons_results.append({
                    "item_id": item_id,
                    "pair": {
                        "test_model": current_test_model, "neighbor_model": current_neigh_model,
                        "test_model_iteration_id": comp_test_iter_id, "neighbor_model_iteration_id": comp_neigh_iter_id,
                    },
                    "error": f"Failed to process judging result: {str(e)}",
                    "outcome_for_test_model": 0.5, "plus_for_test": 0, "plus_for_other": 0, "fraction_for_test": 0.5
                })
    return comparisons_results


def normalize_elo_scores_cw(raw_scores: Dict[str, float], anchor_models: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Normalizes ELO scores using anchor models. (From original CW elo.py)
    Operates on model names.
    """
    if anchor_models is None:
        anchor_models = CW_ANCHOR_MODELS # Use CW's default anchors
        logging.info(f"[ELO-CW] Using default CW anchor models for ELO normalization: {anchor_models}")

    valid_anchors = {k: v for k, v in anchor_models.items() if k in raw_scores and isinstance(raw_scores.get(k), (int, float))}

    if len(valid_anchors) < 2:
        logging.warning(
            f"[ELO-CW] Not enough valid anchor models found in scores ({len(valid_anchors)} found of {len(anchor_models)} required from {list(anchor_models.keys())}). "
            "Returning raw scores."
        )
        return {k: v for k, v in raw_scores.items()} # Return a copy

    anchor_items = list(valid_anchors.items())
    model_a, target_a = anchor_items[0]
    model_b, target_b = anchor_items[1]

    raw_a = raw_scores[model_a]
    raw_b = raw_scores[model_b]

    if abs(raw_a - raw_b) < 1e-6: # Avoid division by zero or near-zero
        logging.warning("[ELO-CW] Anchor models have nearly identical raw scores. Normalization might be unstable. Using scale=1.0, shift=0.0 (effectively raw).")
        scale = 1.0
        shift = target_a - raw_a # Shift to match one anchor if scale is 1
    else:
        scale = (target_a - target_b) / (raw_a - raw_b)
        shift = target_a - (scale * raw_a)

    normalized_scores: Dict[str, float] = {}
    for model, score in raw_scores.items():
        if isinstance(score, (int, float)):
            normalized_scores[model] = (score * scale + shift)
        # else: # Non-numeric scores (e.g. if a model had an error and no ELO) are not included

    logging.info(f"[ELO-CW] ELO normalization applied using anchors {model_a} ({raw_a:.2f}->{target_a}) and {model_b} ({raw_b:.2f}->{target_b}). Scale={scale:.4f}, Shift={shift:.2f}")
    return normalized_scores


def interpolate_elo_from_rubric_scores_cw(model_name: str, model_rubric_score: float, existing_analyses: Dict[str, Any]) -> float:
    """ Interpolate ELO from rubric scores. (From original CW elo.py) """
    other_models_data = []
    for m, info in existing_analyses.items():
        if m == model_name or m == "__metadata__":
            continue
        rubric = info.get("creative_writing_rubric_score_agg")
        elo = info.get("elo")
        if rubric is not None and elo is not None:
            other_models_data.append({"name": m, "rubric": float(rubric), "elo": float(elo)})

    if len(other_models_data) < 2:
        logging.info(f"[ELO-CW] Not enough other models with rubric and ELO to interpolate for {model_name}. Using DEFAULT_ELO.")
        return DEFAULT_ELO

    other_models_data.sort(key=lambda x: x["rubric"])

    # Find position for interpolation
    if model_rubric_score <= other_models_data[0]["rubric"]:
        return other_models_data[0]["elo"]
    if model_rubric_score >= other_models_data[-1]["rubric"]:
        return other_models_data[-1]["elo"]

    for i in range(len(other_models_data) - 1):
        lower = other_models_data[i]
        upper = other_models_data[i+1]
        if lower["rubric"] <= model_rubric_score <= upper["rubric"]:
            if upper["rubric"] == lower["rubric"]: # Avoid division by zero
                return (lower["elo"] + upper["elo"]) / 2.0
            ratio = (model_rubric_score - lower["rubric"]) / (upper["rubric"] - lower["rubric"])
            return lower["elo"] + ratio * (upper["elo"] - lower["elo"])

    # Fallback, should ideally not be reached if logic above is correct
    logging.warning(f"[ELO-CW] Interpolation failed to find a slot for {model_name}. Using DEFAULT_ELO.")
    return DEFAULT_ELO


##############################################
# Main ELO Analysis Function for Creative Writing (Refactored)
##############################################
def run_elo_analysis_creative(
    run_key: str, # Identifies the current run data within creative_bench_runs_file
    elo_results_file: str, # Path to the main ELO JSON file to load/save
    test_model: str, # The primary model being evaluated in this ELO run
    judge_model: str, # Name of the model used for pairwise judging
    api_clients: Dict[str, Any], # API client for the judge_model
    writing_prompts: Dict[str, Any], # Loaded writing prompts data
    concurrency: int = 10, # For parallel processing of opponent matchups
    pairwise_prompt_file: str = "data/pairwise_prompt.txt", # Path to the pairwise prompt template
    negative_criteria: List[str] = [], # For rubric score processing
    creative_bench_runs_file: str = "data/creative_bench_runs.json", # Path to run data
    # max_items_per_model: int = 500, # This was CW's old global cap, now SAMPLING_SCHEDULE.samples controls it
    # ladder_sample_size: int = 7, # This was for CW's old partial_match_test_vs, now SAMPLING_SCHEDULE
    recompute_all_fractions: bool = False # If true, recompute fraction_for_test for all loaded comparisons
) -> Tuple[Dict[str, Any], Optional[str]]: # Returns final solved ratings snapshot and error message
    """
    Refactored ELO analysis for Creative Writing using TrueSkill and EQB3-style sampling.
    Maintains compatibility with CW's elo_results_file.json structure.
    """
    logging.info(f"[ELO-CW] Starting ELO analysis for test_model: '{test_model}', run_key: '{run_key}'")
    elo_error_message: Optional[str] = None

    # 1. Load existing ELO data (the main elo_results_file.json)
    if os.path.exists(elo_results_file):
        existing_analyses = load_json_file(elo_results_file)
        if not existing_analyses: # File exists but is empty or malformed
            logging.warning(f"[ELO-CW] ELO results file {elo_results_file} is empty or malformed. Initializing fresh.")
            existing_analyses = {}
    else:
        logging.info(f"[ELO-CW] ELO results file not found: {elo_results_file}. Initializing fresh.")
        existing_analyses = {}
    
    # Ensure __metadata__ exists for potential future use, though CW doesn't use it like EQB3
    existing_analyses.setdefault("__metadata__", {})


    # 2. Load Creative Bench run data for the current run_key
    if not os.path.exists(creative_bench_runs_file):
        msg = f"Creative bench runs file not found: {creative_bench_runs_file}"
        logging.error(f"[ELO-CW] {msg}")
        return {}, msg
    all_run_data = load_json_file(creative_bench_runs_file)
    if not all_run_data or run_key not in all_run_data:
        msg = f"Run key '{run_key}' not found in {creative_bench_runs_file}"
        logging.error(f"[ELO-CW] {msg}")
        return {}, msg
    
    current_run_tasks = all_run_data[run_key].get("creative_tasks", {})
    if not current_run_tasks:
        logging.info(f"[ELO-CW] No 'creative_tasks' found for run_key '{run_key}'. Processing existing data only.")

    # 3. Load pairwise prompt template
    try:
        pairwise_prompt_template = Path(pairwise_prompt_file).read_text(encoding="utf-8")
    except Exception as e:
        msg = f"Failed to load pairwise prompt template from {pairwise_prompt_file}: {e}"
        logging.error(f"[ELO-CW] {msg}", exc_info=True)
        return {}, msg

    # 4. Aggregate data from current_run_tasks into existing_analyses (CW's original logic)
    # This updates rubric scores, item texts, etc., for models in the current run.
    temp_data_agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(lambda: {"items": {}, "scores_accum": 0.0, "scores_count": 0, "item_scores": {}}))
    for iter_str, prompt_map in current_run_tasks.items():
        for item_id, task_info in prompt_map.items():
            if should_ignore_prompt_cw(item_id):
                continue
            status = task_info.get("status")
            if status not in ["completed", "judged"]: # Assuming these are the valid statuses
                continue
            
            model_name_from_task = task_info.get("test_model") # Original CW used this key
            if not model_name_from_task:
                logging.warning(f"[ELO-CW] Task for item {item_id} in iter {iter_str} missing 'test_model'. Skipping.")
                continue

            # Accumulate text & judge scores (rubric scores)
            block_sum = 0.0
            block_count = 0
            combined_text = ""
            results_by_modifier = task_info.get("results_by_modifier", {})
            for _mod_seed, block_content in results_by_modifier.items():
                txt = block_content.get("model_response", "").strip()
                if txt: combined_text += txt + "\n" # Concatenate responses if multiple modifiers
                
                judge_scores = block_content.get("judge_scores", {}) # These are rubric scores
                for metric, val in judge_scores.items():
                    if isinstance(val, (int, float)):
                        val_adjusted = invert_if_negative(metric, val, negative_criteria)
                        block_sum += val_adjusted
                        block_count += 1
            
            if combined_text: # Only add if there's text
                temp_data_agg[model_name_from_task][iter_str]["items"][item_id] = combined_text.strip()
            if block_count > 0:
                temp_data_agg[model_name_from_task][iter_str]["scores_accum"] += block_sum
                temp_data_agg[model_name_from_task][iter_str]["scores_count"] += block_count
                avg_item_score = round(block_sum / block_count, 2)
                temp_data_agg[model_name_from_task][iter_str]["item_scores"][item_id] = avg_item_score

    # Integrate aggregated temp_data into existing_analyses
    all_model_names_in_system: Set[str] = set(existing_analyses.keys()) - {"__metadata__"}

    for model_name, iter_map_data in temp_data_agg.items():
        all_model_names_in_system.add(model_name)
        model_entry = existing_analyses.setdefault(model_name, {})
        model_entry.setdefault("iterations", {})
        
        total_rubric_score_accum = 0
        total_rubric_score_count = 0
        best_iter_score = -float('inf')
        current_best_iter_id = model_entry.get("best_iteration")

        for iter_id, iter_data in iter_map_data.items():
            iter_rub_score = 0
            if iter_data["scores_count"] > 0:
                iter_rub_score = round(iter_data["scores_accum"] / iter_data["scores_count"], 2)
            
            model_entry["iterations"][iter_id] = {
                "creative_writing_rubric_score_iter": iter_rub_score,
                "items": iter_data["items"],
                "item_scores": iter_data["item_scores"] # Per-item average rubric scores
            }
            total_rubric_score_accum += iter_data["scores_accum"]
            total_rubric_score_count += iter_data["scores_count"]
            if iter_rub_score > best_iter_score:
                best_iter_score = iter_rub_score
                current_best_iter_id = iter_id
        
        if total_rubric_score_count > 0:
            agg_rubric_score = round(total_rubric_score_accum / total_rubric_score_count, 2)
            model_entry["creative_writing_rubric_score_agg"] = agg_rubric_score
            if "elo" not in model_entry: # New model, interpolate ELO
                model_entry["elo"] = round(interpolate_elo_from_rubric_scores_cw(model_name, agg_rubric_score, existing_analyses), 2)
        elif "creative_writing_rubric_score_agg" not in model_entry: # No new data, no old data
             model_entry["creative_writing_rubric_score_agg"] = 0.0 # Default rubric if no scores
             if "elo" not in model_entry: model_entry["elo"] = DEFAULT_ELO


        if current_best_iter_id: model_entry["best_iteration"] = str(current_best_iter_id)
        model_entry.setdefault("elo_analysis", {"pairwise_comparisons": [], "final_elo_ratings": {}})

    if test_model not in existing_analyses:
        logging.warning(f"[ELO-CW] Test model '{test_model}' has no data after aggregation. Adding with default ELO.")
        existing_analyses[test_model] = {
            "elo": DEFAULT_ELO, "creative_writing_rubric_score_agg": 0.0,
            "iterations": {}, "best_iteration": None,
            "elo_analysis": {"pairwise_comparisons": [], "final_elo_ratings": {}}
        }
        all_model_names_in_system.add(test_model)


    # 5. Prepare for ELO Sampling Loop
    # Aggregate all existing comparisons from all models for the solver
    all_comparisons_global: List[Dict[str, Any]] = []
    for m_name, m_data in existing_analyses.items():
        if m_name == "__metadata__": continue
        comps = m_data.get("elo_analysis", {}).get("pairwise_comparisons", [])
        all_comparisons_global.extend(comps)
    
    # Deduplicate this global list (in-memory version)
    all_comparisons_global = deduplicate_comparisons_cw(all_comparisons_global) # Global dedupe
    logging.info(f"[ELO-CW] Loaded {len(all_comparisons_global)} unique global comparisons from existing data.")

    if recompute_all_fractions:
        recompute_fractions_for_comparisons_cw(all_comparisons_global)
        # Also recompute for per-model storage (will be saved later)
        for m_name_iter in all_model_names_in_system:
            if m_name_iter in existing_analyses and "elo_analysis" in existing_analyses[m_name_iter]:
                recompute_fractions_for_comparisons_cw(existing_analyses[m_name_iter]["elo_analysis"]["pairwise_comparisons"])


    # Build initial set of matchups already judged (globally)
    initial_existing_matchups_global = build_existing_matchup_set_cw(all_comparisons_global)
    logging.info(f"[ELO-CW] Built initial existing matchup set with {len(initial_existing_matchups_global)} unique signatures.")

    # Current ELO snapshot for sampling
    elo_snapshot: Dict[str, float] = {
        m: data.get("elo", DEFAULT_ELO)
        for m, data in existing_analyses.items() if m != "__metadata__"
    }
    # Ensure all models that might have tasks or exist in comparisons are in snapshot
    for m_name_iter in all_model_names_in_system:
        if m_name_iter not in elo_snapshot:
            elo_snapshot[m_name_iter] = DEFAULT_ELO
    if test_model not in elo_snapshot: # Should be covered, but as a safeguard
        elo_snapshot[test_model] = DEFAULT_ELO


    # Helper to get item texts and scores for a model's specific iteration
    def get_iteration_details(model_name: str, iter_id: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        items = existing_analyses.get(model_name, {}).get("iterations", {}).get(iter_id, {}).get("items", {})
        item_scores = existing_analyses.get(model_name, {}).get("iterations", {}).get(iter_id, {}).get("item_scores", {})
        return items, item_scores

    # Helper to get top N iteration IDs for a model, sorted by rubric score
    def get_top_n_iterations(model_name: str, n: int) -> List[str]:
        iters_with_scores = []
        for iter_id, iter_data in existing_analyses.get(model_name, {}).get("iterations", {}).items():
            iters_with_scores.append((iter_id, iter_data.get("creative_writing_rubric_score_iter", 0.0)))
        iters_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in iters_with_scores[:n]]


    # 6. ELO Sampling Loop
    new_comparisons_generated_this_run_for_test_model: List[Dict[str, Any]] = []
    # current_existing_matchups tracks matchups judged *during this ELO run* to avoid re-judging within the run.
    # It starts with global known matchups to also avoid re-judging things already in the file.
    current_existing_matchups_this_run = initial_existing_matchups_global.copy()
    
    # Max iterations to consider from each model for pairing up
    # Moved here to be accessible by the helper function defined below if it were outside this scope,
    # or just as a clear constant for the loop.
    MAX_ITERS_PER_MODEL_FOR_PAIRING = 2 

    for stage_idx, (radius_tiers, samples_at_closest_tier) in enumerate(SAMPLING_SCHEDULE, start=1):
        loops, stable = 0, False
        while (
            (radius_tiers == (None,) and loops == 0) or # Stage 1: exactly one loop
            (radius_tiers != (None,) and not stable and loops < MAX_STAGE_LOOPS)
        ):
            loops += 1
            logging.info(f"[ELO-CW] Stage {stage_idx}, Loop {loops}. Test Model: {test_model} (ELO: {elo_snapshot.get(test_model, DEFAULT_ELO):.2f})")

            # Ensure test_model is in snapshot
            if test_model not in elo_snapshot: elo_snapshot[test_model] = DEFAULT_ELO
            
            # Create ladder of opponent models (excluding test_model itself)
            opponent_models_ladder = sorted(
                [m for m in elo_snapshot.keys() if m != test_model],
                key=lambda m: elo_snapshot.get(m, DEFAULT_ELO)
            )
            if not opponent_models_ladder:
                logging.info("[ELO-CW] No opponent models in ladder. Skipping sampling stage.")
                stable = True # End this stage
                continue

            full_ladder_for_ranking = sorted(elo_snapshot.keys(), key=lambda m: elo_snapshot.get(m, DEFAULT_ELO))
            try:
                rank_old_in_full_ladder = full_ladder_for_ranking.index(test_model)
            except ValueError:
                logging.error(f"[ELO-CW] Test model '{test_model}' not found in ELO snapshot for ranking. Aborting stage.")
                elo_error_message = f"Test model '{test_model}' lost during ELO update."
                stable = True; break # Exit while loop

            picked_opponent_indices = _pick_matchups(
                rank_old_in_full_ladder,
                len(full_ladder_for_ranking),
                radius_tiers,
                samples_at_closest_tier
            )
            
            if not picked_opponent_indices:
                logging.debug(f"[ELO-CW Stage {stage_idx}] No opponents picked for test_model '{test_model}'.")
                stable = True 
                continue

            opponents_to_process: List[Tuple[str, int]] = []
            for opp_idx in picked_opponent_indices:
                opponents_to_process.append((full_ladder_for_ranking[opp_idx], opp_idx))

            if not opponents_to_process: 
                logging.debug(f"[ELO-CW Stage {stage_idx}] No opponents picked for test_model '{test_model}'.")
                stable = True
                continue

            round_comparisons_from_judging: List[Dict[str, Any]] = []
            
            # START OF PARALLEL OPPONENT PROCESSING MODIFICATION
            # Define the worker function for processing one opponent
            # This function will capture necessary variables from the outer scope like test_model, elo_snapshot, etc.
            def _process_one_opponent(opponent_details: Tuple[str, int]) -> List[Dict[str, Any]]:
                opponent_model_name, opponent_rank_in_full_ladder_for_opponent = opponent_details # Renamed to avoid clash
                
                logging.debug(f"[ELO-CW] Judging {test_model} (rank {rank_old_in_full_ladder}) vs {opponent_model_name} (rank {opponent_rank_in_full_ladder_for_opponent})")
                
                test_model_top_iters = get_top_n_iterations(test_model, MAX_ITERS_PER_MODEL_FOR_PAIRING)
                opponent_top_iters = get_top_n_iterations(opponent_model_name, MAX_ITERS_PER_MODEL_FOR_PAIRING)

                if not test_model_top_iters or not opponent_top_iters:
                    logging.debug(f"Skipping {opponent_model_name}, not enough iterations for {test_model} or opponent.")
                    return []

                matchups_for_this_opponent: List[Tuple[str, str, str, float, str, str, float]] = []
                
                current_opponent_depth = abs(opponent_rank_in_full_ladder_for_opponent - rank_old_in_full_ladder)

                if radius_tiers == (None,):
                    comparisons_budget_for_opponent = 1
                else:
                    if current_opponent_depth == 1:
                        comparisons_budget_for_opponent = samples_at_closest_tier
                    elif current_opponent_depth == 2:
                        comparisons_budget_for_opponent = max(1, samples_at_closest_tier // 2)
                    else: 
                        comparisons_budget_for_opponent = max(1, samples_at_closest_tier // 4)
                
                logging.debug(f"[ELO-CW] Opponent: {opponent_model_name}, Depth: {current_opponent_depth}, Budget: {comparisons_budget_for_opponent} comparison pairs.")

                test_model_all_items_texts_scores: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
                for tm_iter_id in test_model_top_iters:
                    items, item_scores = get_iteration_details(test_model, tm_iter_id)
                    for item_id, text in items.items():
                        test_model_all_items_texts_scores[item_id].append((tm_iter_id, text, item_scores.get(item_id, 0.0)))
                
                opp_model_all_items_texts_scores: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
                for op_iter_id in opponent_top_iters:
                    items, item_scores = get_iteration_details(opponent_model_name, op_iter_id)
                    for item_id, text in items.items():
                        opp_model_all_items_texts_scores[item_id].append((op_iter_id, text, item_scores.get(item_id, 0.0)))

                # sort iterations numerically if possible, otherwise lexicographically
                _sorted = lambda it: sorted(it, key=lambda s: (int(s) if str(s).isdigit() else s))

                test_model_top_iters_ord = _sorted(test_model_top_iters)
                opponent_top_iters_ord   = _sorted(opponent_top_iters)

                for tm_iter_id in test_model_top_iters_ord:
                    if len(matchups_for_this_opponent) >= comparisons_budget_for_opponent:
                        break

                    tm_items, tm_item_scores = get_iteration_details(test_model, tm_iter_id)

                    # loop prompts in ascending order
                    for item_id in _sorted(tm_items.keys()):
                        if len(matchups_for_this_opponent) >= comparisons_budget_for_opponent:
                            break
                        if should_ignore_prompt_cw(item_id):
                            continue

                        # pick the earliest opponent iteration that also contains this prompt
                        op_iter_id = next(
                            (op_it for op_it in opponent_top_iters_ord
                            if item_id in existing_analyses
                                .get(opponent_model_name, {})
                                .get("iterations", {})
                                .get(op_it, {})
                                .get("items", {})),
                            None,
                        )
                        if op_iter_id is None:
                            continue

                        sig = create_matchup_signature_cw(
                            test_model, opponent_model_name,
                            item_id, tm_iter_id, op_iter_id
                        )
                        if sig in current_existing_matchups_this_run:
                            continue  # already judged

                        op_items, op_item_scores = get_iteration_details(opponent_model_name, op_iter_id)

                        matchups_for_this_opponent.append(
                            (
                                item_id,
                                tm_iter_id, tm_items[item_id], tm_item_scores.get(item_id, 0.0),
                                op_iter_id, op_items[item_id], op_item_scores.get(item_id, 0.0),
                            )
                        )

                        if len(matchups_for_this_opponent) >= comparisons_budget_for_opponent:
                            break

                if matchups_for_this_opponent:
                    logging.info(f"[ELO-CW] Generating {len(matchups_for_this_opponent)} comparison pairs for {test_model} vs {opponent_model_name}.")
                    # The `concurrency` parameter from the main function is used here for inner workers, like EQBench.
                    new_comps_for_opponent = _judge_item_iteration_pairs_in_parallel_cw(
                        test_model, opponent_model_name,
                        matchups_for_this_opponent,
                        pairwise_prompt_template, writing_prompts,
                        judge_model, api_clients,
                        max_workers=concurrency 
                    )
                    return new_comps_for_opponent
                return []

            # Parallel execution of opponent processing
            # The main `concurrency` parameter is used for the number of opponent tasks in parallel.
            outer_workers = min(len(opponents_to_process), concurrency)
            if opponents_to_process and outer_workers > 0:
                with ThreadPoolExecutor(max_workers=outer_workers) as executor:
                    future_to_opponent_details: Dict[Any, Tuple[str, int]] = {
                        executor.submit(_process_one_opponent, opp_details): opp_details
                        for opp_details in opponents_to_process
                    }
                    for future in as_completed(future_to_opponent_details):
                        opponent_details_completed = future_to_opponent_details[future]
                        try:
                            comps_from_opponent = future.result()
                            if comps_from_opponent:
                                round_comparisons_from_judging.extend(comps_from_opponent)
                        except Exception as e:
                            logging.error(f"[ELO-CW] Error processing opponent {opponent_details_completed[0]}: {e}", exc_info=True)
            else:
                logging.debug("[ELO-CW] No opponents to process in parallel this round or outer_workers is 0.")

            # Update the set of matchups judged in this run *after* all parallel tasks for the round are complete.
            if round_comparisons_from_judging:
                update_existing_matchups_from_comparisons_cw(round_comparisons_from_judging, current_existing_matchups_this_run)
            # END OF PARALLEL OPPONENT PROCESSING MODIFICATION

            # --- After judging all opponents in this loop/round ---
            newly_generated_in_this_loop: List[Dict[str, Any]] = []
            for comp in round_comparisons_from_judging: # This list now contains results from all parallel opponent tasks
                pair = comp.get("pair", {})
                item_id = comp.get("item_id")
                # This signature check is against what was known *before this ELO run started*
                sig = create_matchup_signature_cw(
                    pair.get("test_model",""), pair.get("neighbor_model",""), item_id,
                    pair.get("test_model_iteration_id",""), pair.get("neighbor_model_iteration_id","")
                )
                if sig not in initial_existing_matchups_global or "error" in comp: # Always include errors from this run
                    newly_generated_in_this_loop.append(comp)
            
            if newly_generated_in_this_loop:
                logging.info(f"[ELO-CW Stage {stage_idx} Loop {loops}] Generated {len(newly_generated_in_this_loop)} new comparison entries.")
                new_comparisons_generated_this_run_for_test_model.extend(newly_generated_in_this_loop)
                all_comparisons_global.extend(newly_generated_in_this_loop)
                # Deduplicate global list again after adding new ones
                all_comparisons_global = deduplicate_comparisons_cw(all_comparisons_global)


            # --- Re-solve ELO ratings ---
            # Use current rank window setting from config for interim solves
            comps_for_solver = get_solver_comparisons_cw(all_comparisons_global, elo_snapshot, RANK_WINDOW)
            
            if not comps_for_solver:
                logging.warning("[ELO-CW] No valid comparisons for interim ELO solve. Stability not checked.")
                # stable = True # Or let it run max_loops
            else:
                logging.info(f"[ELO-CW] Solving ELO with {len(comps_for_solver)} comparisons for stability check.")
                interim_mu_map = solve_with_trueskill_cw(
                    list(all_model_names_in_system), 
                    comps_for_solver,
                    initial_ratings=elo_snapshot, 
                    use_fixed_initial_ratings=True, 
                    debug=False 
                )
                
                new_elo_snapshot = elo_snapshot.copy()
                new_elo_snapshot.update({m: round(r,2) for m,r in interim_mu_map.items()})

                new_full_ladder = sorted(new_elo_snapshot.keys(), key=lambda m: new_elo_snapshot.get(m, DEFAULT_ELO))
                try:
                    rank_new_in_full_ladder = new_full_ladder.index(test_model)
                except ValueError:
                    logging.error(f"[ELO-CW] Test model '{test_model}' lost from ELO snapshot after interim solve!")
                    rank_new_in_full_ladder = -1 
                    elo_error_message = f"Test model '{test_model}' lost during ELO update."

                stable = (rank_new_in_full_ladder == rank_old_in_full_ladder) and (rank_new_in_full_ladder != -1)
                elo_snapshot = new_elo_snapshot 

            logging.info(f"[ELO-CW Stage {stage_idx} Loop {loops}] Test model rank: {rank_old_in_full_ladder} -> {rank_new_in_full_ladder if 'rank_new_in_full_ladder' in locals() else 'N/A'}. Stable: {stable}")
            if elo_error_message: break 

        # End of while loop for stage
        logging.info(f"[ELO-CW] Stage {stage_idx} finished. Reason: {'stable rank' if stable else ('max loops reached' if loops >= MAX_STAGE_LOOPS else 'error')}")
        if elo_error_message: break 

    # 7. Save newly generated comparisons for the current test_model to its entry in existing_analyses
    if new_comparisons_generated_this_run_for_test_model:
        logging.info(f"[ELO-CW] Appending {len(new_comparisons_generated_this_run_for_test_model)} new comparisons for '{test_model}'.")
        existing_analyses[test_model].setdefault("elo_analysis", {}).setdefault("pairwise_comparisons", [])
        
        existing_analyses[test_model]["elo_analysis"]["pairwise_comparisons"].extend(new_comparisons_generated_this_run_for_test_model)
        
        existing_analyses[test_model]["elo_analysis"]["pairwise_comparisons"] = deduplicate_comparisons_cw(
            existing_analyses[test_model]["elo_analysis"]["pairwise_comparisons"],
            model_name_filter=test_model 
        )
        logging.info(f"[ELO-CW] After deduplication, '{test_model}' has {len(existing_analyses[test_model]['elo_analysis']['pairwise_comparisons'])} comparisons.")
    else:
        logging.info(f"[ELO-CW] No new comparisons were generated for '{test_model}' in this run.")

    # 8. Final ELO Solve (using all global comparisons)
    logging.info("[ELO-CW] Performing final ELO calculation using all available comparisons.")
    final_elo_results_snapshot: Dict[str, Dict[str, Any]] = {} 

    models_for_final_solve = list(all_model_names_in_system)
    if not models_for_final_solve:
        logging.warning("[ELO-CW] No models available for final ELO solve.")
        if not elo_error_message: elo_error_message = "No models for final solve."
        save_json_file(existing_analyses, elo_results_file)
        return existing_analyses, elo_error_message


    final_comps_for_solver = get_solver_comparisons_cw(all_comparisons_global, elo_snapshot, RANK_WINDOW)

    if not final_comps_for_solver:
        logging.warning("[ELO-CW] No valid comparisons available for final ELO solve after filtering.")
        if not elo_error_message: elo_error_message = "No comparisons for final solve after filtering."
        for m_name in models_for_final_solve:
            current_elo = elo_snapshot.get(m_name, DEFAULT_ELO)
            final_elo_results_snapshot[m_name] = {
                "elo": round(current_elo, 2), "elo_norm": round(current_elo, 2), 
                "sigma": round(TS_SIGMA,2), "ci_low": round(current_elo - 1.96*TS_SIGMA,2), "ci_high": round(current_elo + 1.96*TS_SIGMA,2)
            }
    else:
        logging.info(f"[ELO-CW] Final ELO solve with {len(final_comps_for_solver)} comparisons for {len(models_for_final_solve)} models.")
        initial_ratings_final_solve = {m: DEFAULT_ELO for m in models_for_final_solve}

        final_mu_map, final_sigma_map_for_mu_bin = solve_with_trueskill_cw(
            models_for_final_solve,
            final_comps_for_solver,
            initial_ratings=initial_ratings_final_solve,
            use_fixed_initial_ratings=True, 
            bin_size_override=TRUESKILL_BIN_SIZE_FOR_WIN_EXPANSION, 
            return_sigma=True,
            debug=True 
        )

        _ , final_sigma_map_for_ci_bin = solve_with_trueskill_cw(
            models_for_final_solve,
            final_comps_for_solver,
            initial_ratings=final_mu_map, 
            use_fixed_initial_ratings=True, 
            bin_size_override=TRUESKILL_BIN_SIZE_FOR_CI_CALCULATION, 
            return_sigma=True,
            debug=False
        )
        
        normalized_mu_map = normalize_elo_scores_cw(final_mu_map) 

        for m_name in models_for_final_solve:
            mu = final_mu_map.get(m_name, DEFAULT_ELO)
            sigma = final_sigma_map_for_ci_bin.get(m_name, TS_SIGMA) 
            norm_mu = normalized_mu_map.get(m_name, mu) 

            ci_low_raw = mu - 1.96 * sigma
            ci_high_raw = mu + 1.96 * sigma
            
            final_elo_results_snapshot[m_name] = {
                "elo": round(mu, 2),
                "elo_norm": round(norm_mu, 2),
                "sigma": round(sigma, 2),
                "ci_low": round(ci_low_raw, 2),
                "ci_high": round(ci_high_raw, 2),
            }
        
        raw_plus_bounds = {}
        for m, data in final_elo_results_snapshot.items():
            raw_plus_bounds[m] = data["elo"] 
            raw_plus_bounds[f"{m}__ci_low"] = data["ci_low"]
            raw_plus_bounds[f"{m}__ci_high"] = data["ci_high"]
        
        norm_plus_bounds = normalize_elo_scores_cw(raw_plus_bounds)

        for m_name in final_elo_results_snapshot:
            norm_elo_fallback = final_elo_results_snapshot[m_name]["elo_norm"]
            final_elo_results_snapshot[m_name]["ci_low_norm"] = round(norm_plus_bounds.get(f"{m_name}__ci_low", norm_elo_fallback), 2)
            final_elo_results_snapshot[m_name]["ci_high_norm"] = round(norm_plus_bounds.get(f"{m_name}__ci_high", norm_elo_fallback), 2)


    # 9. Update existing_analyses with final ELOs and save
    for m_name, elo_data in final_elo_results_snapshot.items():
        if m_name not in existing_analyses: existing_analyses[m_name] = {} 
        existing_analyses[m_name].update(elo_data)
        existing_analyses[m_name].setdefault("elo_analysis", {}).setdefault("final_elo_ratings", {})
        existing_analyses[m_name]["elo_analysis"]["final_elo_ratings"][m_name] = elo_data["elo"] 

    existing_analyses["__metadata__"]["last_updated_elo_cw"] = datetime.now(timezone.utc).isoformat()

    save_success = save_json_file(existing_analyses, elo_results_file)
    if not save_success:
        msg = f"FAILED to save final ELO results to {elo_results_file}"
        logging.error(f"[ELO-CW] {msg}")
        if not elo_error_message: elo_error_message = msg
    else:
        logging.info(f"[ELO-CW] Successfully saved final ELO results to {elo_results_file}")
        logging.info(f"[ELO-CW] Test model '{test_model}' final ELO: {existing_analyses.get(test_model, {}).get('elo', 'N/A')}, Norm ELO: {existing_analyses.get(test_model, {}).get('elo_norm', 'N/A')}")

    return final_elo_results_snapshot, elo_error_message