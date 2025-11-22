"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch

from ppo import PPO
from ppo_lls import PPO_LLS
from network import FeedForwardNN
from lls_model import LLS_Model
from eval_policy import eval_policy

import argparse

def get_args():
    """
        Description:
        Parses arguments at command line.

        Parameters:
            None

        Return:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train_lls')              # can be 'train', 'train_lls', or 'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename
    parser.add_argument('--action_space', dest='action_space', type=str, 
        choices=["discrete", "continuous"], default="discrete")

    args = parser.parse_args()

    return args

def train(env, hyperparameters, actor_model, critic_model):
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training

        Return:
            None
    """ 
    print(f"Training", flush=True)

    # Create a model for PPO.
    if args.mode == 'train_lls':
        model = PPO_LLS(policy_class=LLS_Model, env=env, **hyperparameters)
    elif args.mode == 'train':
        model = PPO(policy_class=LLS_Model, env=env, **hyperparameters)
    elif args.mode == 'train_ffnn':
        model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
    else:
        print(f"Invalid mode: {args.mode}", flush=True)
        sys.exit(0)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)

def test(env, actor_model, action_space: str = "discrete"):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Load the saved model checkpoint
    checkpoint = torch.load(actor_model)
    
    # Check if this checkpoint contains metadata or just state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # New format with metadata
        saved_action_space = checkpoint['action_space']
        saved_obs_dim = checkpoint['obs_dim']
        saved_act_dim = checkpoint['act_dim']
        state_dict = checkpoint['state_dict']
        
        # Validate action space matches
        if saved_action_space != action_space:
            print(f"WARNING: Model was trained with action_space='{saved_action_space}' "
                  f"but you specified action_space='{action_space}'", flush=True)
            print(f"Using the model's action_space: '{saved_action_space}'", flush=True)
            action_space = saved_action_space
        
        obs_dim = saved_obs_dim
        act_dim = saved_act_dim
        
        print(f"Loaded model metadata: action_space={action_space}, obs_dim={obs_dim}, act_dim={act_dim}", flush=True)
    else:
        # If model saved as only state dict (no metadata), use provided action_space
        print(f"WARNING: Model was saved without metadata. Assuming action_space='{action_space}'", flush=True)
        state_dict = checkpoint
        
        # Extract dimensions from environment
        obs_dim = env.observation_space.shape[0]
        if action_space == 'discrete':  
            act_dim = env.action_space.n
        elif action_space == 'continuous':
            act_dim = env.action_space.shape[0]
        else:
            raise ValueError(
                f"Unrecognized action_space type in test(): '{action_space}'. "
                "action_space must be one of {'discrete', 'continuous'}."
            )

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim, is_actor=True, action_space=action_space)

    # Load in the actor model weights
    policy.load_state_dict(state_dict)

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True, action_space=action_space)

def main(args):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
                'timesteps_per_batch': 2048, 
                'max_timesteps_per_episode': 2048,  # LunarLander can have longer episodes
                'gamma': 0.99, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2,
                'render': True,
                'render_every_i': 10,
                'action_space': args.action_space,
              }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    continuous_flag = (args.action_space == "continuous")
    env = gym.make('LunarLander-v3', 
                   continuous=continuous_flag,
                   enable_wind=True,
                   wind_power=5,
                   render_mode='human' if args.mode == 'test' else 'rgb_array')

    # Train or test, depending on the mode specified
    if 'train' in args.mode:
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model, action_space=args.action_space)

if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)