import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.gail import GAIL
import numpy as np
import sys
from gymnasium.envs.registration import register
from mpi4py import MPI
import torch.optim as optim




# Import ShuttleEnv and add the module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_GAIL import ShuttleEnv

register(
    id='ShuttleEnv-v0',
    entry_point='env.env_GAIL:ShuttleEnv',  # Path to your environment class
)

SEED = None  # Set your seed here

def make_env():
    env = ShuttleEnv(env_config={}, render_mode=None)
    env = RolloutInfoWrapper(env)
    env.reset(seed=SEED)  # Set the seed here
    return env

class SB3CheckpointWrapper:
    def __init__(self, sb3_callback, model):
        """Wrap a stable_baselines3 CheckpointCallback to be used in imitation library."""
        self.sb3_callback = sb3_callback
        self.sb3_callback.model = model  # Manually assign the model to the callback

    def __call__(self, rollout):
        # Pass the rollout data to the original callback
        return self.sb3_callback.on_step()  # This will handle checkpoint saving logic

# Modify your train_gail function
def train_gail(env_id, expert_data_path, save_dir, total_timesteps=500000, demo_batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load enriched expert data
    expert_data = np.load(expert_data_path)
    observations = expert_data["observations"]
    actions = expert_data["actions"]
    rewards = expert_data["rewards"]
    next_observations = expert_data["next_observations"]
    dones = expert_data["dones"]

    # Prepare expert trajectories using the full enriched data
    expert_trajectories = types.Transitions(
        obs=observations,
        acts=actions,
        next_obs=next_observations,
        dones=dones,
        infos=[{'reward': rew} for rew in rewards]  # You can pass rewards through the `infos` field
    )

    # Create vectorized environments
    env = make_vec_env(make_env, n_envs=8)

    # Set hyperparameters manually here
    learning_rate = 1e-4
    n_steps = 1024
    batch_size = 4096
    clip_range = 0.2
    ent_coef = 0.02

    # Initialize the PPO model for the policy network
    learner = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard/"),
        device=device,
        seed=SEED,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        gamma=0.99,  # Keeping gamma constant
        ent_coef=ent_coef,
    )

    # Define reward network for GAIL
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    
    # Initialize GAIL algorithm
    trainer = GAIL(
        venv=env,
        demonstrations=expert_trajectories,
        demo_batch_size=demo_batch_size,
        reward_net=reward_net,
        gen_algo=learner,
        allow_variable_horizon=True
    )

    # Set the learning rate of the discriminator manually
    trainer._disc_opt.param_groups[0]['lr'] = 1e-4  # Manually adjust learning rate for discriminator

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=os.path.join(save_dir, "gail_run"), name_prefix='gail_checkpoint')

    # Wrap CheckpointCallback with SB3CheckpointWrapper and pass the PPO model (learner)
    wrapped_callback = SB3CheckpointWrapper(checkpoint_callback, learner)

    # Attach the wrapped callback during training
    trainer.train(total_timesteps=total_timesteps, callback=wrapped_callback)

    # Save the final model
    learner.save(os.path.join(save_dir, 'gail_final_test_3'))
    print("Training completed.\n")



if __name__ == "__main__":
    env_id = 'ShuttleEnv-v0'  # Registered environment
    expert_data_path = '/tf/rl_project/data/expert_demonstration/session_10_12/expert_demonstration.npz'
    save_dir = '/tf/rl_project/IL_models/GAIL_models/session_10_15'

    train_gail(env_id, expert_data_path, save_dir)
