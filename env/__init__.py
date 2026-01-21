# envs/__init__.py

from gymnasium.envs.registration import register

register(
    id='ShuttleEnv-v0',
    entry_point='envs.shuttle_env:ShuttleEnv',
)
