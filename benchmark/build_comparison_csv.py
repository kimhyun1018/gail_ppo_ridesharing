import os
import json
import csv
from statistics import mean
from typing import Dict, List, Optional


# -----------------------------
# User inputs
# -----------------------------
OFFLINE_DIR = "/tf/rl_project/data/offline_benchmark"
ONLINE_NEAREST_DIR = "/tf/rl_project/data/online_nearest_benchmark"
ONLINE_WAIT_AWARE_DIR = "/tf/rl_project/data/online_wait_t_aware_benchmark"
RL_DIR = "/tf/rl_project/data/rl_performance_eval"

OUTPUT_CSV = "/tf/rl_project/data/benchmark_episode_comparison.csv"


# -----------------------------
# Helpers
# -----------------------------
def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return mean(vals)


def load_json_files_from_dir(directory: str) -> List[dict]:
    data = []
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            path = os.path.join(directory, fname)
            with open(path, "r") as f:
                obj = json.load(f)

            # Handle either:
            # 1) one episode per file (dict)
            # 2) combined results file (list of dicts)
            if isinstance(obj, dict):
                data.append(obj)
            elif isinstance(obj, list):
                data.extend(obj)

    return data


def build_episode_map(records: List[dict]) -> Dict[int, dict]:
    episode_map = {}
    for rec in records:
        episode_id = rec.get("episode_id")
        if episode_id is None:
            continue
        episode_map[int(episode_id)] = rec
    return episode_map


def summarize_episode(record: Optional[dict]) -> Dict[str, Optional[float]]:
    if record is None:
        return {
            "avg_wait_time": None,
            "avg_in_vehicle_time": None,
            "avg_service_time": None,
            "num_passengers_generated": None,
            "num_passengers_served": None,
            "service_rate": None,
        }

    passengers = record.get("passengers", [])
    wait_times = [p.get("waiting_time") for p in passengers]
    in_vehicle_times = [p.get("in_vehicle_time") for p in passengers]
    service_times = [p.get("service_time") for p in passengers]

    generated = record.get("num_passengers_generated")
    served = record.get("num_passengers_served")

    service_rate = None
    if generated not in (None, 0) and served is not None:
        service_rate = served / generated

    return {
        "avg_wait_time": safe_mean(wait_times),
        "avg_in_vehicle_time": safe_mean(in_vehicle_times),
        "avg_service_time": safe_mean(service_times),
        "num_passengers_generated": generated,
        "num_passengers_served": served,
        "service_rate": service_rate,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    offline_records = load_json_files_from_dir(OFFLINE_DIR)
    online_nearest_records = load_json_files_from_dir(ONLINE_NEAREST_DIR)
    online_wait_aware_records = load_json_files_from_dir(ONLINE_WAIT_AWARE_DIR)
    rl_records = load_json_files_from_dir(RL_DIR)

    offline_map = build_episode_map(offline_records)
    online_nearest_map = build_episode_map(online_nearest_records)
    online_wait_aware_map = build_episode_map(online_wait_aware_records)
    rl_map = build_episode_map(rl_records)

    all_episode_ids = sorted(
        set(offline_map.keys())
        | set(online_nearest_map.keys())
        | set(online_wait_aware_map.keys())
        | set(rl_map.keys())
    )

    fieldnames = [
        "episode_id",

        "offline_avg_wait_time",
        "offline_avg_in_vehicle_time",
        "offline_avg_service_time",
        "offline_num_passengers_generated",
        "offline_num_passengers_served",
        "offline_service_rate",

        "online_nearest_avg_wait_time",
        "online_nearest_avg_in_vehicle_time",
        "online_nearest_avg_service_time",
        "online_nearest_num_passengers_generated",
        "online_nearest_num_passengers_served",
        "online_nearest_service_rate",

        "online_wait_aware_avg_wait_time",
        "online_wait_aware_avg_in_vehicle_time",
        "online_wait_aware_avg_service_time",
        "online_wait_aware_num_passengers_generated",
        "online_wait_aware_num_passengers_served",
        "online_wait_aware_service_rate",

        "rl_avg_wait_time",
        "rl_avg_in_vehicle_time",
        "rl_avg_service_time",
        "rl_num_passengers_generated",
        "rl_num_passengers_served",
        "rl_service_rate",
    ]

    rows = []

    for episode_id in all_episode_ids:
        offline_summary = summarize_episode(offline_map.get(episode_id))
        online_nearest_summary = summarize_episode(online_nearest_map.get(episode_id))
        online_wait_aware_summary = summarize_episode(online_wait_aware_map.get(episode_id))
        rl_summary = summarize_episode(rl_map.get(episode_id))

        row = {
            "episode_id": episode_id,

            "offline_avg_wait_time": offline_summary["avg_wait_time"],
            "offline_avg_in_vehicle_time": offline_summary["avg_in_vehicle_time"],
            "offline_avg_service_time": offline_summary["avg_service_time"],
            "offline_num_passengers_generated": offline_summary["num_passengers_generated"],
            "offline_num_passengers_served": offline_summary["num_passengers_served"],
            "offline_service_rate": offline_summary["service_rate"],

            "online_nearest_avg_wait_time": online_nearest_summary["avg_wait_time"],
            "online_nearest_avg_in_vehicle_time": online_nearest_summary["avg_in_vehicle_time"],
            "online_nearest_avg_service_time": online_nearest_summary["avg_service_time"],
            "online_nearest_num_passengers_generated": online_nearest_summary["num_passengers_generated"],
            "online_nearest_num_passengers_served": online_nearest_summary["num_passengers_served"],
            "online_nearest_service_rate": online_nearest_summary["service_rate"],

            "online_wait_aware_avg_wait_time": online_wait_aware_summary["avg_wait_time"],
            "online_wait_aware_avg_in_vehicle_time": online_wait_aware_summary["avg_in_vehicle_time"],
            "online_wait_aware_avg_service_time": online_wait_aware_summary["avg_service_time"],
            "online_wait_aware_num_passengers_generated": online_wait_aware_summary["num_passengers_generated"],
            "online_wait_aware_num_passengers_served": online_wait_aware_summary["num_passengers_served"],
            "online_wait_aware_service_rate": online_wait_aware_summary["service_rate"],

            "rl_avg_wait_time": rl_summary["avg_wait_time"],
            "rl_avg_in_vehicle_time": rl_summary["avg_in_vehicle_time"],
            "rl_avg_service_time": rl_summary["avg_service_time"],
            "rl_num_passengers_generated": rl_summary["num_passengers_generated"],
            "rl_num_passengers_served": rl_summary["num_passengers_served"],
            "rl_service_rate": rl_summary["service_rate"],
        }

        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to: {OUTPUT_CSV}")
    print(f"Total episodes written: {len(rows)}")


if __name__ == "__main__":
    main()