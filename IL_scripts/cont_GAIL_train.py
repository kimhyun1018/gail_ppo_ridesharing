import os
import torch
from stable_baselines3 import PPO
from imitation.data import types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.env_util import make_vec_env
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
import numpy as np
import sys
from gymnasium.envs.registration import register
import logging

# Import ShuttleEnv and add the module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_GAIL import ShuttleEnv

register(
    id='ShuttleEnv-v0',
    entry_point='env.env_GAIL:ShuttleEnv',  # Path to your environment class
)

SEED = 42  # Set your seed here

def make_env():
    env = ShuttleEnv(env_config={}, render_mode=None)
    print("Action space:", env.action_space)
    print("Action space shape:", env.action_space.shape)
    env = RolloutInfoWrapper(env)
    env.reset(seed=SEED)  # Set the seed here
    return env

# Custom logger callback to save metrics in a log file
class CustomGAILLogger:
    def __init__(self, log_file_path, trainer, verbose=0):
        self.rewards = []
        self.policy_losses = []
        self.disc_losses = []
        self.adversary_accuracies = []
        self.steps = []
        self.step_count = 0
        self.trainer = trainer
        self.log_file_path = log_file_path

        # Set up the logger
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filemode='w'  # 'w' to overwrite the file each time the script is run
        )
        self.logger = logging.getLogger()

    def __call__(self, *args, **kwargs):
        self._on_step()

    def _on_step(self):
        self.step_count += 1

        # Fetch policy loss and other metrics directly from the trainer's logger
        policy_loss = self._get_policy_loss()
        disc_loss, disc_acc = self._get_disc_metrics()
        reward = self._get_reward()

        self.policy_losses.append(policy_loss)
        self.disc_losses.append(disc_loss)
        self.adversary_accuracies.append(disc_acc)
        self.rewards.append(reward)

        self.steps.append(self.step_count)

        # Log metrics to the log file
        self.log_metrics()

    def _get_policy_loss(self):
        try:
            log_data = self.trainer.gen_algo.logger.name_to_value
            policy_loss = log_data.get("train/policy_loss", 0)
            return policy_loss
        except Exception as e:
            print(f"Error in _get_policy_loss: {e}")
            return 0

    def _get_disc_metrics(self):
        try:
            log_data = self.trainer.logger.name_to_value
            disc_loss = log_data.get("disc/disc_loss", 0)
            disc_acc = log_data.get("disc/disc_acc", 0)
            return disc_loss, disc_acc
        except Exception as e:
            print(f"Error in _get_disc_metrics: {e}")
            return 0, 0

    def _get_reward(self):
        try:
            log_data = self.trainer.gen_algo.logger.name_to_value
            reward = log_data.get("rollout/ep_rew_mean", 0)
            return reward
        except Exception as e:
            print(f"Error in _get_reward: {e}")
            return 0

    def log_metrics(self):
        self.logger.info(
            f"Step {self.step_count}: Reward = {self.rewards[-1]}, "
            f"Policy Loss = {self.policy_losses[-1]}, "
            f"Disc Loss = {self.disc_losses[-1]}, "
            f"Disc Acc = {self.adversary_accuracies[-1]}"
        )

def continue_training_gail(env_id, expert_data_path, save_dir, continue_from, total_timesteps=100000, demo_batch_size=256, use_logger=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Continuing training on {device}")

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
        infos=[{'reward': rew} for rew in rewards]
    )

    # Create vectorized environments
    env = make_vec_env(make_env, n_envs=8)


    # Set hyperparameters manually here
    learning_rate = 5e-4
    n_steps = 512
    batch_size = 8192
    clip_range = 0.1
    ent_coef = 0.02


    # Initialize the PPO model for the policy network, and load the previous checkpoint model
    if os.path.exists(continue_from):
        print(f"Loading model from {continue_from}")
        learner = PPO.load(continue_from, env=env, device=device, tensorboard_log="./ppo_tensorboard/")
    else:
        print(f"No existing model found at {continue_from}. Exiting...")
        return

    # Reinitialize the optimizer to apply the learning rate properly
    learner.policy.optimizer = torch.optim.Adam(learner.policy.parameters(), lr=learning_rate)


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

    trainer._disc_opt.param_groups[0]['lr'] = 1e-4

    learner.policy.optimizer.param_groups[0]['lr'] = learning_rate
    learner.n_steps = n_steps
    #learner.rollout_buffer = RolloutBuffer(n_steps, learner.observation_space, learner.action_space, device=device, gamma=0.99)
    learner.batch_size = batch_size
    learner.clip_range = lambda _: clip_range  # Wrap the float in a lambda to avoid the 'callable' error
    learner.ent_coef = ent_coef

    # Optional custom logger
    if use_logger:
        log_file_path = os.path.join(save_dir, 'gail_training_log.txt')
        custom_logging_callback = CustomGAILLogger(log_file_path, trainer)
    else:
        custom_logging_callback = None  # Disable the custom logger

    # Continue training the GAIL model from the saved checkpoint
    trainer.train(total_timesteps=total_timesteps, callback=custom_logging_callback)

    # Save the new model after continuing training
    learner.save(os.path.join(save_dir, 'gail_final_3'))
    print("Continuing training completed.")


if __name__ == "__main__":
    env_id = 'ShuttleEnv-v0'  # Registered environment
    expert_data_path = '/tf/rl_project/data/expert_demonstration/session_10_12/expert_demonstration.npz'
    save_dir = '/tf/rl_project/IL_models/GAIL_models/session_10_15'
    continue_from = os.path.join(save_dir, 'gail_final_2.zip')  # Path to the saved model to continue from

    continue_training_gail(env_id, expert_data_path, save_dir, continue_from)