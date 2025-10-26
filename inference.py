"""
    This script runs inference on a trained Lunar Lander policy and saves a GIF of the episode.
"""

import gymnasium as gym
import torch
import numpy as np
from PIL import Image
import os
from network import FeedForwardNN
import argparse

def run_inference(actor_model_path, num_episodes=1, save_gif=True, env_params=None):
    """
        Runs inference on a trained policy and saves a GIF of the episode.

        Parameters:
            actor_model_path - path to the trained actor model
            num_episodes - number of episodes to run
            save_gif - whether to save a GIF of the episode
            env_params - dictionary of environment parameters

        Return:
            None
    """
    # Create environment with specified parameters
    env_params = env_params or {}
    env = gym.make('LunarLander-v3', 
                   continuous=False,
                   enable_wind=True,
                   wind_power=5,
                   render_mode='rgb_array',
                   turbulence_power=1.5,
                   )

    # Extract dimensions from environment
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Build policy network
    policy = FeedForwardNN(obs_dim, act_dim, is_actor=True)
    
    # Load trained policy
    policy.load_state_dict(torch.load(actor_model_path))
    policy.eval()  # Set to evaluation mode

    # Lists to store rewards and frames
    all_rewards = []
    frame_list = []

    # Run episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        frames = []

        while not done:
            # Get action from policy
            with torch.no_grad():
                action_probs = policy(torch.tensor(obs, dtype=torch.float))
                action = torch.argmax(action_probs).item()

            # Take action in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Save frame if we're making a GIF
            if save_gif:
                frame = env.render()
                frames.append(frame)

        # Save frames for this episode
        if save_gif:
            frame_list.extend(frames)

        all_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")

    env.close()

    # Print statistics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"\nStatistics over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Save GIF if requested
    if save_gif and frame_list:
        # Create saved_models directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frame_list]
        
        # Save as GIF
        output_gif = 'saved_models/lunar_lander_inference.gif'
        pil_frames[0].save(
            output_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=50,  # 50ms between frames
            loop=0
        )
        print(f"Animation saved to {output_gif}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on trained Lunar Lander policy")
    parser.add_argument('--actor-model', type=str, default='saved_models/ppo_actor.pt',
                      help='Path to the trained actor model')
    parser.add_argument('--num-episodes', type=int, default=1,
                      help='Number of episodes to run')
    parser.add_argument('--no-save-gif', action='store_true',
                      help='Disable saving GIF of the episode')
    parser.add_argument('--wind-power', type=float, default=5.0,
                      help='Wind power in the environment')
    parser.add_argument('--gravity', type=float, default=-10.0,
                      help='Gravity value in the environment')

    args = parser.parse_args()

    env_params = {
        'wind_power': args.wind_power,
        'gravity': args.gravity
    }

    run_inference(
        actor_model_path=args.actor_model,
        num_episodes=args.num_episodes,
        save_gif=not args.no_save_gif,
        env_params=env_params
    )

if __name__ == '__main__':
    main() 