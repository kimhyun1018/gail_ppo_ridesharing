import os
import sys
import torch
import gymnasium as gym
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.envs.registration import register
import random

# Import ShuttleEnv and add the module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_GAIL import ShuttleEnv

register(
    id='ShuttleEnv-v0',
    entry_point='env.env_GAIL:ShuttleEnv',  # Path to your environment class
)

# General settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

def make_env():
    # No fixed seed to allow stochastic behavior
    env = ShuttleEnv(env_config={'grid_size': 10, 'num_shuttles': 1, 'total_time': 500, 'max_passengers': 6, 'max_active_passengers': 10}, render_mode=None)
    env = RolloutInfoWrapper(env)
    return env

# Function to load the policy learned from GAIL and continue training with PPO
def continue_training_with_ppo(save_dir, total_timesteps=200000, tensorboard_log_dir="tensorboard_logs"):
    # Create the environment with multiple independent workers, each with a unique seed for randomness
    env = make_vec_env(lambda: make_env(), n_envs=1, seed=random.randint(0, 10000))

    # Load the saved PPO policy from GAIL training
    ppo_model_path = os.path.join(save_dir, 'RL2.zip')
    if os.path.exists(ppo_model_path):
        logging.info(f"Loading PPO policy learned from GAIL at: {ppo_model_path}")
        
        # Load the model and set the TensorBoard log path
        model = PPO.load(ppo_model_path, env=env, device=device, tensorboard_log=tensorboard_log_dir)
        
        model.learning_rate = 1e-4  
        model.n_steps = 1024
        model.batch_size = 4096
        model.clip_range = lambda _: 0.2  # PPO clipping range
        model.ent_coef = 0.001  # Entropy coefficient to encourage exploration
        model.gamma = 0.99  # Discount factor
    else:
        raise FileNotFoundError(f"Could not find the PPO model file from GAIL training at {ppo_model_path}")

    # Set up TensorBoard logging directory
    tensorboard_log_path = os.path.join(save_dir, tensorboard_log_dir)
    if not os.path.exists(tensorboard_log_path):
        os.makedirs(tensorboard_log_path)

    # Set up checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=os.path.join(save_dir, "ppo_continued_run"), 
        name_prefix='ppo_checkpoint'
    )

    # Continue training and log metrics to TensorBoard
    logging.info(f"Continuing training using PPO for {total_timesteps} timesteps with TensorBoard logging...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, log_interval=10)

    # Save the final model after continued training
    final_model_path = os.path.join(save_dir, 'RL3')
    model.save(final_model_path)
    logging.info(f"Continued PPO training complete. Model saved at {final_model_path}")

# Main function to continue training
if __name__ == "__main__":
    save_dir = '/tf/rl_project/RL_models/11_17'
    total_timesteps = 200000  # Continue training for 100,000 timesteps (adjust as needed)
    tensorboard_log_dir = "tensorboard_logs"
    continue_training_with_ppo(save_dir, total_timesteps, tensorboard_log_dir)
