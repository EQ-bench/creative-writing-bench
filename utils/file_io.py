import os
import json
import logging
import threading
from typing import Dict, Any
import time
import tempfile
import fcntl
from contextlib import contextmanager

_file_locks = {}
_file_locks_lock = threading.Lock()

def _normalize_path(file_path: str) -> str:
    """Normalize paths so all callers share the same in-process lock key."""
    return os.path.abspath(file_path)

def get_file_lock(file_path: str) -> threading.Lock:
    """
    Acquire or create a per-file lock to avoid concurrent writes.
    """
    file_path = _normalize_path(file_path)
    with _file_locks_lock:
        if file_path not in _file_locks:
            _file_locks[file_path] = threading.Lock()
        return _file_locks[file_path]

@contextmanager
def _interprocess_file_lock(file_path: str):
    """
    Cross-process lock using a sidecar lock file.
    """
    lock_path = file_path + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

def load_json_file(file_path: str) -> dict:
    """
    Thread-safe read of a JSON file, returning an empty dict if not found or error.
    """
    file_path = _normalize_path(file_path)
    lock = get_file_lock(file_path)
    with lock:
        with _interprocess_file_lock(file_path):
            if not os.path.exists(file_path):
                return {}
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in {file_path}: {e}")
                return {}
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                return {}

def _atomic_write_json(data: Dict[str, Any], file_path: str):
    file_path = _normalize_path(file_path)
    target_dir = os.path.dirname(file_path) or "."
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(file_path) + ".",
        suffix=".tmp",
        dir=target_dir
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

def save_json_file(data: Dict[str, Any], file_path: str, max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    file_path = _normalize_path(file_path)
    lock = get_file_lock(file_path)
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(retry_delay)
        with lock:
            try:
                with _interprocess_file_lock(file_path):
                    _atomic_write_json(data, file_path)
                logging.debug(f"Successfully wrote JSON to {file_path}")
                return True
            except Exception as e:
                logging.error(f"save_json_file() attempt {attempt+1} failed: {e}")
    logging.error(f"Failed to save JSON to {file_path} after {max_retries} attempts.")
    return False

def update_run_data(runs_file: str, run_key: str, update_dict: Dict[str, Any],
                    max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    """
    Thread-safe function to MERGE partial run data into the existing run file.
    
    *** For "creative_tasks", we do a nested merge so that
    we do not overwrite entire iterations or entire prompt dictionaries. ***
    """
    runs_file = _normalize_path(runs_file)
    lock = get_file_lock(runs_file)
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(retry_delay)

        with lock:
            with _interprocess_file_lock(runs_file):
                if not os.path.exists(runs_file):
                    current_runs = {}
                else:
                    try:
                        with open(runs_file, 'r', encoding='utf-8') as f:
                            current_runs = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        msg = (
                            f"Error reading/parsing run file {runs_file}: {e}. "
                            "Aborting to avoid overwriting valid data."
                        )
                        logging.error(msg)
                        return False

                if not isinstance(current_runs, dict):
                    current_runs = {}

                if run_key not in current_runs:
                    current_runs[run_key] = {}

                # Merge top-level keys
                for top_key, new_val in update_dict.items():
                    if top_key in ["conversations", "results", "elo_analysis"]:
                        # same as original approach: shallow merge
                        if top_key not in current_runs[run_key]:
                            current_runs[run_key][top_key] = {}
                        if not isinstance(new_val, dict) or not isinstance(current_runs[run_key][top_key], dict):
                            current_runs[run_key][top_key] = new_val
                        else:
                            for item_id, item_data in new_val.items():
                                current_runs[run_key][top_key][item_id] = item_data

                    elif top_key == "creative_tasks":
                        if top_key not in current_runs[run_key]:
                            current_runs[run_key][top_key] = {}
                        # new_val should be: { iteration_idx => { prompt_id => {...} } }
                        for iteration_idx, prompt_map in new_val.items():
                            if iteration_idx not in current_runs[run_key][top_key]:
                                current_runs[run_key][top_key][iteration_idx] = {}
                            # Now we do a second-level merge on prompt_id
                            if not isinstance(prompt_map, dict):
                                # if user overwrote iteration data with something non-dict, just set
                                current_runs[run_key][top_key][iteration_idx] = prompt_map
                            else:
                                for p_id, p_data in prompt_map.items():
                                    current_runs[run_key][top_key][iteration_idx][p_id] = p_data

                    else:
                        # Overwrite any other key
                        current_runs[run_key][top_key] = new_val

                # Now write updated data
                try:
                    _atomic_write_json(current_runs, runs_file)
                    logging.debug(f"Successfully updated run_key={run_key} in {runs_file}")
                    return True
                except Exception as e:
                    logging.error(f"Error saving merged run data on attempt {attempt+1}: {e}")

    logging.error(f"update_run_data() failed after {max_retries} attempts on {runs_file}")
    return False
