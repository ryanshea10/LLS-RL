import gymnasium as gym
from noisyenv.wrappers import RandomUniformScaleReward

base_env = gym.make('LunarLander-v3')
env = RandomUniformScaleReward(env=base_env, noise_rate=0.01, low=0.9, high=1.1)

# And just use as you would normally
observation, info = env.reset(seed=333)

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()