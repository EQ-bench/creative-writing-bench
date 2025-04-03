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

# (Keep compute_benchmark_results_creative and pick_best_iteration_for_each_prompt_model as they are)
# ... (previous functions remain unchanged) ...

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
            if t_info.get("status") in ["completed", "judged"]:
                completed_tasks.append(t_info)

    if not completed_tasks:
        logging.warning(f"No completed tasks found for run {run_key}. No final results computed.")
        return

    # 1) Summarize
    summary_result = compute_single_benchmark_score_creative(completed_tasks, negative_criteria)
    creative_score_0_20 = summary_result["overall_score"]
    eqbench_creative_score = summary_result["eqbench_creative_score"]

    # 2) Bootstrap
    boot_stats = bootstrap_benchmark_stability_creative(completed_tasks, negative_criteria)

    # 3) Merge into run_data
    results_dict = run_data.get("results", {})
    bench_results = results_dict.get("benchmark_results", {})
    bench_results["creative_score_0_20"] = creative_score_0_20
    bench_results["eqbench_creative_score"] = eqbench_creative_score
    bench_results["bootstrap_analysis"] = boot_stats

    # Overwrite
    results_dict["benchmark_results"] = bench_results

    update_run_data(runs_file, run_key, {"results": results_dict})

    logging.info(f"Creative benchmark summary => Score(0-20)={creative_score_0_20}, eqbench_creative(0..100)={eqbench_creative_score}")
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
            model_name = t_data.get("test_model", "unknown_model")
            key = (model_name, prompt_id)
            if key not in groups:
                groups[key] = []
            groups[key].append((iteration_idx, t_data))

    # For each (model, prompt_id), pick which iteration had the highest average rubric score
    best_map = {}
    for (model_name, prompt_id), items in groups.items():
        best_score = -999
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
                        new_val = invert_if_negative(metric, val, negative_criteria)
                        score_sum += new_val
                        count += 1
            avg_score = (score_sum / count) if count > 0 else 0.0
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
    run_elo: bool = True  # --- New parameter ---
) -> str:
    """
    Main function to run the creative writing benchmark. Similar structure to eqbench’s run_eq_bench_therapy.
    """
    from utils.file_io import load_json_file, update_run_data
    from core.conversation import CreativeWritingTask

    def sanitize_model_name(name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]+', '_', name)

    sanitized_model = sanitize_model_name(test_model)
    base_id = run_id if run_id else str(uuid.uuid4())
    run_key = f"{base_id}__{sanitized_model}"

    # init or resume
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
        # Ensure start time exists if resuming
        if "start_time" not in runs[run_key]:
             update_run_data(runs_file, run_key, {"start_time": datetime.now().isoformat()})


    creative_writing_criteria = []
    if os.path.exists(creative_criteria_file):
        with open(creative_criteria_file, 'r', encoding='utf-8') as f:
            creative_writing_criteria = [line.strip() for line in f if line.strip()]

    # load negative criteria
    negative_criteria = []
    if os.path.exists(negative_criteria_file):
        with open(negative_criteria_file, 'r', encoding='utf-8') as f:
            negative_criteria = [line.strip() for line in f if line.strip()]

    # load judge prompt
    if not os.path.exists(judge_prompt_file):
        raise FileNotFoundError(f"Judge prompt file not found: {judge_prompt_file}")
    with open(judge_prompt_file, 'r', encoding='utf-8') as f:
        judge_prompt_template = f.read()

    # load creative prompts
    if not os.path.exists(creative_prompts_file):
        raise FileNotFoundError(f"Creative prompts file not found: {creative_prompts_file}")
    creative_prompts = load_json_file(creative_prompts_file)

    # Build API clients
    api_clients = {
        "test": APIClient(model_type="test"),
        "judge": APIClient(model_type="judge")
    }

    # If redo_judging => remove existing judge scores
    if redo_judging:
        run_data = load_json_file(runs_file).get(run_key, {})
        c_tasks = run_data.get("creative_tasks", {})
        tasks_updated = False
        for i_str, p_dict in c_tasks.items():
            for prompt_id, c_dict in p_dict.items():
                # Reset status to allow re-judging
                if c_dict.get("status") in ["completed", "judged"]:
                    c_dict["status"] = "generated" # Mark as needing judging
                    results_by_mod = c_dict.get("results_by_modifier", {})
                    for seed_mod, block in results_by_mod.items():
                        block.pop("judge_scores", None)
                        block.pop("raw_judge_text", None)
                    # Update specific task data
                    update_run_data(runs_file, run_key, {
                        "creative_tasks": {
                            i_str: {
                                prompt_id: c_dict
                            }
                        }
                    })
                    tasks_updated = True
                    logging.info(f"Reset judge data and status for iteration={i_str}, prompt_id={prompt_id}")
        if tasks_updated:
             logging.info("Completed resetting judge data due to --redo-judging flag.")


    # figure out tasks
    run_data = load_json_file(runs_file).get(run_key, {})
    existing_tasks = run_data.get("creative_tasks", {})

    tasks_to_run = []
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
                # Resume existing
                resumed_task = CreativeWritingTask.from_dict(c_data)
                tasks_to_run.append(resumed_task)
            else:
                # Create new
                iteration_seed = seed_mods[(i-1) % len(seed_mods)]
                new_task = CreativeWritingTask(
                    prompt_id=prompt_key,
                    base_prompt=base_prompt,
                    seed_modifiers=[iteration_seed], # Only the one for this iteration
                    iteration_index=i,
                    test_model=test_model,
                    judge_model=judge_model
                )
                tasks_to_run.append(new_task)

    logging.info(f"Total tasks to process: {len(tasks_to_run)} (across {iterations} iteration(s))")

    # 1) Generate (if needed)
    tasks_needing_generation = []
    for task_obj in tasks_to_run:
        i_str = str(task_obj.iteration_index)
        prompt_id = task_obj.prompt_id
        iteration_dict = existing_tasks.get(i_str, {})
        c_data = iteration_dict.get(str(prompt_id), {})
        status = c_data.get("status", None)

        # Only generate if status is None (new) or explicitly needs generation
        if status is None or status == "pending_generation":
             tasks_needing_generation.append(task_obj)
        # If redo_judging is set, we don't need to regenerate, just rejudge later.
        # If status is 'generated', 'completed', or 'judged', generation is done.

    if tasks_needing_generation:
        logging.info(f"Found {len(tasks_needing_generation)} tasks requiring generation.")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures_map = {}
            for task_obj in tasks_needing_generation:
                fut = executor.submit(
                    task_obj.generate_creative_piece,
                    api_clients,
                    runs_file,
                    run_key,
                    save_interval
                )
                futures_map[fut] = task_obj
            # Use list(futures_map.keys()) for tqdm if futures_map can change size during iteration
            for fut in tqdm(list(futures_map.keys()), total=len(futures_map), desc="Generating creative pieces"):
                try:
                    _ = fut.result() # Wait for completion, handle exceptions if needed
                except Exception as e:
                    task = futures_map[fut]
                    logging.error(f"Error generating for task {task.prompt_id} (iter {task.iteration_index}): {e}", exc_info=True)
    else:
        logging.info("No tasks require generation.")


    # 2) Judge (if needed)
    tasks_needing_judging = []
    # Reload run_data to get latest status after generation step
    run_data = load_json_file(runs_file).get(run_key, {})
    existing_tasks = run_data.get("creative_tasks", {})

    for task_obj in tasks_to_run:
        i_str = str(task_obj.iteration_index)
        prompt_id = task_obj.prompt_id
        iteration_dict = existing_tasks.get(i_str, {})
        c_data = iteration_dict.get(str(prompt_id), {})
        status = c_data.get("status", None)

        # Judge if status is 'generated' OR if redo_judging is True and status is 'completed' or 'judged'
        needs_judging = (status == "generated")
        if redo_judging and status in ["completed", "judged"]:
             needs_judging = True # Force re-judging

        if needs_judging:
            # Ensure the task object has the latest data before judging
            task_obj = CreativeWritingTask.from_dict(c_data)
            tasks_needing_judging.append(task_obj)

    if tasks_needing_judging:
        logging.info(f"Found {len(tasks_needing_judging)} tasks requiring judging.")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures_map = {}
            for task_obj in tasks_needing_judging:
                fut = executor.submit(
                    task_obj.judge,
                    api_clients,
                    judge_prompt_template,
                    creative_writing_criteria,
                    negative_criteria,
                    runs_file,
                    run_key
                )
                futures_map[fut] = task_obj
            for fut in tqdm(list(futures_map.keys()), total=len(futures_map), desc="Judging creative pieces"):
                 try:
                    _ = fut.result() # Wait for completion, handle exceptions if needed
                 except Exception as e:
                    task = futures_map[fut]
                    logging.error(f"Error judging task {task.prompt_id} (iter {task.iteration_index}): {e}", exc_info=True)

    else:
        logging.info("No tasks require judging.")


    # 3) Compute final results (Rubric scores, Bootstrap)
    runs_after = load_json_file(runs_file)
    compute_benchmark_results_creative(runs_after, run_key, runs_file, negative_criteria)


    # 4) Run ELO analysis (conditionally)
    # --- Check the run_elo flag ---
    if run_elo:
        logging.info("Starting ELO analysis...")
        try:
            run_elo_analysis_creative(
                run_key=run_key,
                elo_results_file="elo_results.json",
                test_model=test_model,
                judge_model=judge_model,
                api_clients=api_clients,
                writing_prompts=creative_prompts,
                concurrency=num_threads,
                pairwise_prompt_file="data/pairwise_prompt.txt",
                negative_criteria=negative_criteria,
                creative_bench_runs_file=runs_file
            )

            # fetch and report the normalized ELO score
            elo_results = load_json_file("elo_results.json")
            elo_raw = "N/A"
            elo_norm = "N/A"

            # Extract the normalized ELO score if available
            if test_model in elo_results:
                elo_raw = elo_results[test_model].get("elo", "N/A")
                elo_norm = elo_results[test_model].get("elo_norm", "N/A")

            # Add to run results
            # Reload runs data to avoid overwriting other results potentially added
            current_runs = load_json_file(runs_file)
            results_dict = current_runs.get(run_key, {}).get("results", {})
            bench_results = results_dict.get("benchmark_results", {})
            bench_results["elo_raw"] = elo_raw
            bench_results["elo_normalized"] = elo_norm

            # Update the run data
            update_run_data(runs_file, run_key, {"results": {"benchmark_results": bench_results}})

            # Log the ELO scores
            logging.info(f"ELO scores for {test_model}: Raw: {elo_raw}, Normalized: {elo_norm}")
            # Print statement moved to main summary box

        except Exception as e:
            logging.error(f"ELO analysis failed: {e}", exc_info=True)
            # Optionally update run data to indicate ELO failure
            current_runs = load_json_file(runs_file)
            results_dict = current_runs.get(run_key, {}).get("results", {})
            bench_results = results_dict.get("benchmark_results", {})
            bench_results["elo_raw"] = "Error"
            bench_results["elo_normalized"] = "Error"
            update_run_data(runs_file, run_key, {"results": {"benchmark_results": bench_results}})

    else:
        logging.info("Skipping ELO analysis as per --no-elo flag.")
        # Ensure ELO fields are marked as skipped if they don't exist
        current_runs = load_json_file(runs_file)
        results_dict = current_runs.get(run_key, {}).get("results", {})
        bench_results = results_dict.get("benchmark_results", {})
        if "elo_raw" not in bench_results:
             bench_results["elo_raw"] = "Skipped"
        if "elo_normalized" not in bench_results:
             bench_results["elo_normalized"] = "Skipped"
        update_run_data(runs_file, run_key, {"results": {"benchmark_results": bench_results}})


    # Mark status=completed and record end time
    update_run_data(
        runs_file,
        run_key,
        {
            "status": "completed",
            "end_time": datetime.now().isoformat()
        }
    )

    return run_key