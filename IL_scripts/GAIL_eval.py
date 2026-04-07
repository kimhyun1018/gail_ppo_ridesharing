import os
import sys
import json
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force matplotlib to use 'TkAgg' backend only if rendering is used
matplotlib.use('TkAgg')

# Import ShuttleEnv and add the module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_GAIL import ShuttleEnv

# Register the environment
register(
    id='ShuttleEnv-v0',
    entry_point='env.env_GAIL:ShuttleEnv',
)


def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def make_env(render_mode=None):
    env = ShuttleEnv(env_config={}, render_mode=render_mode)
    return env


def serialize_passenger(passenger: dict) -> dict:
    """
    Convert passenger dict into a JSON-safe structure for benchmark replay.
    """
    return {
        "name": passenger.get("name"),
        "origin": list(passenger.get("origin")) if passenger.get("origin") is not None else None,
        "destination": list(passenger.get("destination")) if passenger.get("destination") is not None else None,
        "request_time": passenger.get("request_time"),
        "pickup_time": passenger.get("pickup_time"),
        "dropoff_time": passenger.get("dropoff_time"),
        "picked_up": passenger.get("picked_up"),
        "status": passenger.get("status"),
        "matd": passenger.get("matd"),
    }


def save_episode_realized_demand(
    output_dir: str,
    episode_idx: int,
    env,
    total_reward: float,
    steps: int,
    model_path: str,
):
    """
    Save the realized demand for a single evaluation episode.
    """
    os.makedirs(output_dir, exist_ok=True)

    episode_payload = {
        "episode_id": episode_idx + 1,
        "model_path": model_path,
        "grid_size": env.grid_size,
        "num_shuttles": env.num_shuttles,
        "total_reward": total_reward,
        "steps": steps,
        "env_time": env.time,
        "initial_shuttle_state": {
            "position": [env.grid_size // 2, env.grid_size // 2],
            "orientation": "north",
        },
        "passengers": [serialize_passenger(p) for p in env.all_passengers],
    }

    file_path = os.path.join(output_dir, f"episode_{episode_idx + 1:04d}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(episode_payload, f, indent=2, default=convert_numpy)


def save_episode_performance(
    output_dir: str,
    episode_idx: int,
    env,
    total_reward: float,
    steps: int,
    model_path: str,
):
    """
    Save passenger-level RL performance metrics for one episode.
    """
    os.makedirs(output_dir, exist_ok=True)

    episode_metrics = {
        "episode_id": episode_idx + 1,
        "model_path": model_path,
        "total_reward": total_reward,
        "steps": steps,
        "env_time": env.time,
        "num_passengers_generated": len(env.all_passengers),
        "num_passengers_served": len(env.passenger_log),
        "passengers": []
    }

    for p in env.passenger_log:
        episode_metrics["passengers"].append({
            "name": p["name"],
            "origin": list(p["origin"]) if p.get("origin") is not None else None,
            "destination": list(p["destination"]) if p.get("destination") is not None else None,
            "request_time": p.get("request_time"),
            "pickup_time": p.get("pickup_time"),
            "dropoff_time": p.get("dropoff_time"),
            "waiting_time": p.get("wait_time"),
            "in_vehicle_time": p.get("travel_time"),
            "service_time": p.get("total_time"),  # waiting + in-vehicle
        })

    file_path = os.path.join(
        output_dir,
        f"episode_{episode_idx + 1:04d}_performance.json"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(episode_metrics, f, indent=2, default=convert_numpy)


def evaluate_gail_model(
    saved_model_path,
    episodes=10,
    render_mode=None,
    deterministic=True,
    realized_demand_dir="/tf/rl_project/data/realized_demand_eval",
    performance_dir="/tf/rl_project/data/rl_performance_eval",
):
    # Load the trained PPO model
    model = PPO.load(saved_model_path)

    # Create environment
    env = make_env(render_mode=render_mode)

    # Evaluate the agent for the given number of episodes
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            if render_mode is not None:
                env.render()

            # Get the action from the trained model
            action, _ = model.predict(obs, deterministic=deterministic)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if the episode has finished
            done = terminated or truncated

            total_reward += reward
            steps += 1

        # Save realized demand for this episode
        save_episode_realized_demand(
            output_dir=realized_demand_dir,
            episode_idx=episode,
            env=env,
            total_reward=total_reward,
            steps=steps,
            model_path=saved_model_path,
        )

        # Save RL performance for this episode
        save_episode_performance(
            output_dir=performance_dir,
            episode_idx=episode,
            env=env,
            total_reward=total_reward,
            steps=steps,
            model_path=saved_model_path,
        )

        print(
            f"Episode {episode + 1}: "
            f"Total reward: {total_reward}, "
            f"Steps taken: {steps}, "
            f"Passengers generated: {len(env.all_passengers)}, "
            f"Passengers served: {len(env.passenger_log)}\n"
            f"  Demand saved to: {os.path.join(realized_demand_dir, f'episode_{episode + 1:04d}.json')}\n"
            f"  Performance saved to: {os.path.join(performance_dir, f'episode_{episode + 1:04d}_performance.json')}"
        )

    env.close()


if __name__ == "__main__":
    saved_model_path = "/tf/rl_project/RL_models/11_17/RL3.zip"

    evaluate_gail_model(
        saved_model_path=saved_model_path,
        episodes=1000,
        render_mode=None,
        deterministic=True,
        realized_demand_dir="/tf/rl_project/data/realized_demand_eval",
        performance_dir="/tf/rl_project/data/rl_performance_eval",
    )