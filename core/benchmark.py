# core/benchmark.py

import os
import re
import uuid
import time
import math
import logging
from datetime import datetime
import statistics as stats
import json
import random
import numpy as np
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils.file_io import load_json_file, update_run_data, save_json_file
from utils.api import APIClient
from core.conversation import CreativeWritingTask
from core.scoring import (
    compute_single_benchmark_score_creative,
    bootstrap_benchmark_stability_creative
)
from core.elo import run_elo_analysis_creative

# (compute_benchmark_results_creative and pick_best_iteration_for_each_prompt_model remain unchanged)
def compute_benchmark_results_creative(runs, run_key, runs_file, negative_criteria):
    """
    Gathers all creative tasks from the run, finds the completed/judged ones, aggregates results,
    does a final store into runs_file -> results -> benchmark_results.
    Then does a bootstrap stability check, storing that in "bootstrap_analysis."
    """
    run_data = runs.get(run_key, {})
    ctasks = run_data.get("creative_tasks", {})

    # Collect tasks that are done (nested iteration->prompt structure)
    completed_tasks = []
    for i_str, p_dict in ctasks.items():
        for prompt_id, t_info in p_dict.items():
            # Ensure status indicates completion and scores exist
            if t_info.get("status") in ["completed", "judged"]:
                 # Check if judge_scores actually exist for at least one modifier
                 has_scores = False
                 results_by_mod = t_info.get("results_by_modifier", {})
                 for mod, block in results_by_mod.items():
                     if block.get("judge_scores"):
                         has_scores = True
                         break
                 if has_scores:
                    completed_tasks.append(t_info)
                 else:
                    logging.warning(f"Task {run_key}/{i_str}/{prompt_id} has status '{t_info.get('status')}' but no judge_scores found. Excluding from final results.")


    if not completed_tasks:
        logging.warning(f"No completed tasks with scores found for run {run_key}. No final results computed.")
        # Still save empty/default results structure? Or just return? Let's return for now.
        # Ensure results structure exists though
        results_dict = run_data.get("results", {})
        bench_results = results_dict.get("benchmark_results", {})
        bench_results["creative_score_0_20"] = 0.0 # Or None/NaN?
        bench_results["eqbench_creative_score"] = 0.0 # Or None/NaN?
        bench_results["bootstrap_analysis"] = {"error": "No completed tasks with scores"}
        results_dict["benchmark_results"] = bench_results
        update_run_data(runs_file, run_key, {"results": results_dict})
        return

    # 1) Summarize
    summary_result = compute_single_benchmark_score_creative(completed_tasks, negative_criteria)
    creative_score_0_20 = summary_result["overall_score"]
    eqbench_creative_score = summary_result["eqbench_creative_score"]

    # 2) Bootstrap
    boot_stats = bootstrap_benchmark_stability_creative(completed_tasks, negative_criteria)

    # 3) Merge into run_data
    # Reload data just before final update to minimize overwriting concurrent changes
    current_runs = load_json_file(runs_file)
    run_data = current_runs.get(run_key, {}) # Get latest run_data
    results_dict = run_data.get("results", {})
    bench_results = results_dict.get("benchmark_results", {}) # Get latest results

    # Update with new calculations
    bench_results["creative_score_0_20"] = creative_score_0_20
    bench_results["eqbench_creative_score"] = eqbench_creative_score
    bench_results["bootstrap_analysis"] = boot_stats

    # Overwrite benchmark_results within results_dict
    results_dict["benchmark_results"] = bench_results

    # Update the run data file
    update_run_data(runs_file, run_key, {"results": results_dict})

    logging.info(f"Creative benchmark summary => Score(0-20)={creative_score_0_20:.2f}, eqbench_creative(0..100)={eqbench_creative_score:.2f}")
    if "error" not in boot_stats:
        logging.info(f"Bootstrap 95% CI: ({boot_stats['ci_lower']:.2f}, {boot_stats['ci_upper']:.2f})")


def pick_best_iteration_for_each_prompt_model(run_data, negative_criteria) -> Dict[str, Any]:
    """
    After we've generated multiple iterations for each (prompt, model),
    we pick the iteration that had the *best rubric score* for each (prompt, model).
    We'll produce a dictionary keyed by iteration_index -> { prompt_id -> data } only for those best items.
    """
    creative_tasks = run_data.get("creative_tasks", {})
    if not creative_tasks:
        return {}

    from core.scoring import invert_if_negative

    # We group tasks by (model_name, prompt_id) so we can find the best iteration for each prompt
    groups = {}
    for i_str, p_dict in creative_tasks.items():
        iteration_idx = int(i_str)
        for prompt_id, t_data in p_dict.items():
            if t_data.get("status") not in ["completed", "judged"]:
                continue
            # Ensure scores exist before considering
            has_scores = False
            results_by_mod = t_data.get("results_by_modifier", {})
            for mod, block in results_by_mod.items():
                if block.get("judge_scores"):
                    has_scores = True
                    break
            if not has_scores:
                continue # Skip if no scores found

            model_name = t_data.get("test_model", "unknown_model")
            key = (model_name, prompt_id)
            if key not in groups:
                groups[key] = []
            groups[key].append((iteration_idx, t_data))

    # For each (model, prompt_id), pick which iteration had the highest average rubric score
    best_map = {}
    for (model_name, prompt_id), items in groups.items():
        best_score = -float('inf') # Initialize with negative infinity
        best_iter = None
        best_data = None
        for (iteration_idx, t_data) in items:
            score_sum = 0.0
            count = 0
            results_by_mod = t_data.get("results_by_modifier", {})
            for seed_mod, block in results_by_mod.items():
                j_scores = block.get("judge_scores", {})
                for metric, val in j_scores.items():
                    if isinstance(val, (int, float)):
                        try:
                            new_val = invert_if_negative(metric, val, negative_criteria)
                            score_sum += new_val
                            count += 1
                        except ValueError: # Handle cases where score might be non-numeric string like "N/A"
                             logging.warning(f"Non-numeric score value '{val}' for metric '{metric}' in task {model_name}/{prompt_id}/iter{iteration_idx}. Skipping.")
                             continue

            # Handle division by zero if no valid scores found
            avg_score = (score_sum / count) if count > 0 else -float('inf')

            if avg_score > best_score:
                best_score = avg_score
                best_iter = iteration_idx
                best_data = t_data

        if best_data is not None:
            i_str = str(best_iter)
            if i_str not in best_map:
                best_map[i_str] = {}
            best_map[i_str][prompt_id] = best_data

    return best_map


def run_eq_bench_creative(
    test_model: str,
    judge_model: str,
    runs_file: str,
    num_threads: int = 4,
    run_id: Optional[str] = None,
    creative_prompts_file: str = "data/creative_writing_prompts_v3.json",
    creative_criteria_file: str = "data/creative_writing_criteria.txt",
    negative_criteria_file: str = "data/negative_criteria.txt",
    judge_prompt_file: str = "data/creative_writing_judging_prompt.txt",
    redo_judging: bool = False,
    save_interval: int = 2,
    iterations: int = 1,
    run_elo: bool = True
) -> str:
    """
    Main function to run the creative writing benchmark.
    """
    from utils.file_io import load_json_file, update_run_data
    from core.conversation import CreativeWritingTask

    def sanitize_model_name(name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]+', '_', name)

    sanitized_model = sanitize_model_name(test_model)
    base_id = run_id if run_id else str(uuid.uuid4())
    run_key = f"{base_id}__{sanitized_model}"

    # --- Init or resume ---
    runs = load_json_file(runs_file)
    if run_key not in runs:
        init_dict = {
            "test_model": test_model,
            "judge_model": judge_model,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "creative_prompts_file": creative_prompts_file,
            "creative_tasks": {},
            "results": {}
        }
        update_run_data(runs_file, run_key, init_dict)
        logging.info(f"Created new run: {run_key}")
    else:
        logging.info(f"Resuming run: {run_key}")
        if "start_time" not in runs[run_key]:
             update_run_data(runs_file, run_key, {"start_time": datetime.now().isoformat()})

    # --- Load criteria and prompts ---
    creative_writing_criteria = []
    if os.path.exists(creative_criteria_file):
        with open(creative_criteria_file, 'r', encoding='utf-8') as f:
            creative_writing_criteria = [line.strip() for line in f if line.strip()]

    negative_criteria = []
    if os.path.exists(negative_criteria_file):
        with open(negative_criteria_file, 'r', encoding='utf-8') as f:
            negative_criteria = [line.strip() for line in f if line.strip()]

    if not os.path.exists(judge_prompt_file):
        raise FileNotFoundError(f"Judge prompt file not found: {judge_prompt_file}")
    with open(judge_prompt_file, 'r', encoding='utf-8') as f:
        judge_prompt_template = f.read()

    if not os.path.exists(creative_prompts_file):
        raise FileNotFoundError(f"Creative prompts file not found: {creative_prompts_file}")
    creative_prompts = load_json_file(creative_prompts_file)

    # --- Build API clients ---
    api_clients = {
        "test": APIClient(model_type="test"),
        "judge": APIClient(model_type="judge")
    }

    # --- Handle redo_judging: Reset status and scores IN THE FILE ---
    if redo_judging:
        logging.info("Processing --redo-judging flag: Resetting saved judge data...")
        run_data_for_redo = load_json_file(runs_file).get(run_key, {}) # Load fresh data
        c_tasks_for_redo = run_data_for_redo.get("creative_tasks", {})
        tasks_updated = False
        tasks_to_update_in_file = {} # Accumulate changes

        for i_str, p_dict in c_tasks_for_redo.items():
            for prompt_id, c_dict in p_dict.items():
                if c_dict.get("status") in ["completed", "judged"]:
                    # Create a copy to modify
                    updated_c_dict = c_dict.copy()
                    updated_c_dict["status"] = "generated" # Mark as needing judging
                    results_by_mod = updated_c_dict.get("results_by_modifier", {})
                    # Ensure results_by_modifier exists and is a dict
                    if not isinstance(results_by_mod, dict):
                        results_by_mod = {}

                    updated_results_by_mod = {}
                    for seed_mod, block in results_by_mod.items():
                         # Create a copy of the block to modify
                         updated_block = block.copy()
                         updated_block.pop("judge_scores", None)
                         updated_block.pop("raw_judge_text", None)
                         updated_results_by_mod[seed_mod] = updated_block

                    updated_c_dict["results_by_modifier"] = updated_results_by_mod

                    # Add to batch update
                    if i_str not in tasks_to_update_in_file:
                        tasks_to_update_in_file[i_str] = {}
                    tasks_to_update_in_file[i_str][prompt_id] = updated_c_dict
                    tasks_updated = True
                    logging.debug(f"Marking task for reset: iteration={i_str}, prompt_id={prompt_id}")

        # Perform batch update to the file
        if tasks_updated:
            update_run_data(runs_file, run_key, {"creative_tasks": tasks_to_update_in_file})
            logging.info("Completed resetting judge data in file due to --redo-judging flag.")
        else:
            logging.info("No tasks found with 'completed' or 'judged' status to reset for --redo-judging.")


    # --- Figure out tasks: Create or load task objects into memory ---
    run_data = load_json_file(runs_file).get(run_key, {}) # Load potentially updated data
    existing_tasks = run_data.get("creative_tasks", {})
    tasks_to_run = [] # This list will hold the CreativeWritingTask objects

    for prompt_key, prompt_obj in creative_prompts.items():
        base_prompt = prompt_obj.get("writing_prompt", "")
        seed_mods = prompt_obj.get("seed_modifiers", [])
        if not seed_mods:
            logging.warning(f"No seed modifiers for prompt {prompt_key}; skipping.")
            continue

        for i in range(1, iterations+1):
            i_str = str(i)
            iteration_dict = existing_tasks.get(i_str, {})
            c_data = iteration_dict.get(str(prompt_key))

            if c_data and c_data.get("test_model") == test_model:
                # Load existing task data into an object
                try:
                    resumed_task = CreativeWritingTask.from_dict(c_data)
                    if resumed_task.status in ("completed", "judged"):
                        missing = any(
                            not blk.get("judge_scores")
                            for blk in resumed_task.results_by_modifier.values()
                        )
                        if missing:
                            logging.info(
                                f"Resetting status to 'generated' for "
                                f"task {prompt_key} (iter {i_str}) – "
                                "previous judge failed or incomplete."
                            )
                            resumed_task.status = "generated"
                    tasks_to_run.append(resumed_task)
                except Exception as e:
                     logging.error(f"Failed to load task from dict for iter={i_str}, prompt={prompt_key}: {e}. Skipping task.", exc_info=True)
                     # Add placeholder or skip? Skipping for now.
            else:
                # Create new task object
                iteration_seed = seed_mods[(i-1) % len(seed_mods)]
                new_task = CreativeWritingTask(
                    prompt_id=prompt_key,
                    base_prompt=base_prompt,
                    seed_modifiers=[iteration_seed],
                    iteration_index=i,
                    test_model=test_model,
                    judge_model=judge_model
                )
                tasks_to_run.append(new_task)

    logging.info(f"Total task objects prepared: {len(tasks_to_run)} (across {iterations} iteration(s))")

    # --- 1) Generate (if needed) ---
    tasks_needing_generation = []
    for task_obj in tasks_to_run:
        # Check the status of the object in memory
        if task_obj.status not in ["generated", "completed"]:
             tasks_needing_generation.append(task_obj)
        # If status is already 'generated', 'completed', 'judged', etc., skip generation

    if tasks_needing_generation:
        logging.info(f"Found {len(tasks_needing_generation)} tasks requiring generation.")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures_map = {
                executor.submit(
                    task_obj.generate_creative_piece, # This method should update task_obj.status
                    api_clients,
                    runs_file,
                    run_key,
                    save_interval # save_interval might not be needed if saving happens at end of method
                ): task_obj for task_obj in tasks_needing_generation
            }
            for fut in tqdm(list(futures_map.keys()), total=len(futures_map), desc="Generating creative pieces"):
                task = futures_map[fut]
                try:
                    _ = fut.result() # Wait for completion & potential status update on task object
                except Exception as e:
                    logging.error(f"Error generating for task {task.prompt_id} (iter {task.iteration_index}): {e}", exc_info=True)
                    # The generate_creative_piece method should ideally set task.status to an error state
    else:
        logging.info("No tasks require generation based on initial status.")


    # --- 2) Judge (if needed) ---
    tasks_needing_judging = []
    # ** Minimal Change Logic: Primarily check in-memory status, consult file only for redo **
    logging.info("Identifying tasks requiring judging...")
    # Load file data *once* specifically for the redo_judging check if needed
    existing_tasks_for_redo = {}
    if redo_judging:
         run_data_for_redo = load_json_file(runs_file).get(run_key, {})
         existing_tasks_for_redo = run_data_for_redo.get("creative_tasks", {})

    for task_obj in tasks_to_run: # Iterate through the objects potentially updated by generation
        needs_judging = False
        current_status = task_obj.status # Check in-memory status first

        if current_status == "generated":
            needs_judging = True

        # Check redo flag against *saved* status if necessary
        if redo_judging and not needs_judging: # Only check file if not already marked by 'generated' status
            i_str = str(task_obj.iteration_index)
            prompt_id = task_obj.prompt_id
            iteration_dict = existing_tasks_for_redo.get(i_str, {})
            c_data = iteration_dict.get(str(prompt_id), {})
            saved_status = c_data.get("status", None)
            if saved_status == "generated": # Check if file was reset correctly
                 needs_judging = True
                 logging.debug(f"Task {prompt_id} (Iter {i_str}) marked for judging via redo file check.")
                 # Ensure in-memory object is also ready for judging if file was reset
                 if task_obj.status != "generated":
                      task_obj.status = "generated"
                      # Clear potential stale scores from object if status was completed/judged
                      results_by_mod = getattr(task_obj, 'results_by_modifier', {})
                      for seed_mod, block in results_by_mod.items():
                          block.pop("judge_scores", None)
                          block.pop("raw_judge_text", None)


        if needs_judging:
            # Add the task object itself (potentially updated in memory)
            tasks_needing_judging.append(task_obj)

    if tasks_needing_judging:
        logging.info(f"Found {len(tasks_needing_judging)} tasks requiring judging.")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures_map = {
                 executor.submit(
                    task_obj.judge, # This method should update task_obj.status and save results
                    api_clients,
                    judge_prompt_template,
                    creative_writing_criteria,
                    negative_criteria,
                    runs_file,
                    run_key
                ): task_obj for task_obj in tasks_needing_judging
            }
            for fut in tqdm(list(futures_map.keys()), total=len(futures_map), desc="Judging creative pieces"):
                 task = futures_map[fut]
                 try:
                    _ = fut.result() # Wait for completion
                 except Exception as e:
                    logging.error(f"Error judging task {task.prompt_id} (iter {task.iteration_index}): {e}", exc_info=True)
                    # The judge method should ideally set task.status to an error state
    else:
        logging.info("No tasks require judging based on status and redo flag.")


    # --- 3) Compute final results (Rubric scores, Bootstrap) ---
    logging.info("Calculating final benchmark results...")
    # Load the absolute latest data from the file AFTER judging threads finished saving
    final_runs_data = load_json_file(runs_file)
    compute_benchmark_results_creative(final_runs_data, run_key, runs_file, negative_criteria)


    # --- 4) Run ELO analysis (conditionally) ---
    if run_elo:
        logging.info("Starting ELO analysis...")
        try:
            # ELO analysis likely needs the final saved data, so it reads runs_file internally or is passed data
            run_elo_analysis_creative(
                run_key=run_key,
                elo_results_file="elo_results.json",
                test_model=test_model,
                judge_model=judge_model,
                api_clients=api_clients,
                writing_prompts=creative_prompts, # Check if ELO needs more specific task data
                concurrency=num_threads,
                pairwise_prompt_file="data/pairwise_prompt.txt",
                negative_criteria=negative_criteria,
                creative_bench_runs_file=runs_file # Pass the path to the runs file
            )

            # Fetch and report the normalized ELO score
            elo_results = load_json_file("elo_results.json")
            elo_raw = "N/A"
            elo_norm = "N/A"
            if test_model in elo_results:
                elo_raw = elo_results[test_model].get("elo", "N/A")
                elo_norm = elo_results[test_model].get("elo_norm", "N/A")

            # Add ELO results to run data (load latest before updating)
            current_runs = load_json_file(runs_file)
            results_dict = current_runs.get(run_key, {}).get("results", {})
            bench_results = results_dict.get("benchmark_results", {}) # Get latest benchmark results
            bench_results["elo_raw"] = elo_raw
            bench_results["elo_normalized"] = elo_norm
            results_dict["benchmark_results"] = bench_results # Put updated benchmark results back
            update_run_data(runs_file, run_key, {"results": results_dict}) # Save updated results

            logging.info(f"ELO scores for {test_model}: Raw: {elo_raw}, Normalized: {elo_norm}")

        except FileNotFoundError:
             logging.error("ELO analysis skipped: elo_results.json not found. Was ELO run successful?")
             # Update run data to indicate ELO wasn't found/run
             current_runs = load_json_file(runs_file)
             results_dict = current_runs.get(run_key, {}).get("results", {})
             bench_results = results_dict.get("benchmark_results", {})
             bench_results["elo_raw"] = "Not Found"
             bench_results["elo_normalized"] = "Not Found"
             results_dict["benchmark_results"] = bench_results
             update_run_data(runs_file, run_key, {"results": results_dict})
        except Exception as e:
            logging.error(f"ELO analysis failed: {e}", exc_info=True)
            # Update run data to indicate ELO error
            current_runs = load_json_file(runs_file)
            results_dict = current_runs.get(run_key, {}).get("results", {})
            bench_results = results_dict.get("benchmark_results", {})
            bench_results["elo_raw"] = "Error"
            bench_results["elo_normalized"] = "Error"
            results_dict["benchmark_results"] = bench_results
            update_run_data(runs_file, run_key, {"results": results_dict})

    else:
        logging.info("Skipping ELO analysis as per --no-elo flag.")
        # Ensure ELO fields are marked as skipped if they don't exist (load latest first)
        current_runs = load_json_file(runs_file)
        results_dict = current_runs.get(run_key, {}).get("results", {})
        bench_results = results_dict.get("benchmark_results", {})
        if "elo_raw" not in bench_results:
             bench_results["elo_raw"] = "Skipped"
        if "elo_normalized" not in bench_results:
             bench_results["elo_normalized"] = "Skipped"
        results_dict["benchmark_results"] = bench_results
        update_run_data(runs_file, run_key, {"results": results_dict})


    # --- Mark status=completed and record end time ---
    update_run_data(
        runs_file,
        run_key,
        {
            "status": "completed",
            "end_time": datetime.now().isoformat()
        }
    )
    logging.info(f"Run {run_key} marked as completed.")

    return run_key