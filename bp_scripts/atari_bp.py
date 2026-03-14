import os
import argparse
import wandb
import torch
from functools import partial
from pettingzoo.mpe import simple_spread_v3
import gymnasium as gym

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

# from lls_model import LLS_RAY


def simple_spread_env_creator(args):
    env = simple_spread_v3.parallel_env(
        N=3,
        continuous_actions=False,
        max_cycles=25,
    )
    return env

def lunar_lander_env_creator(args):
    return gym.make("LunarLander-v3",
                   continuous=args.continuous,
                   gravity=args.gravity,
                   enable_wind=args.enable_wind,
                   wind_power=args.wind_power,
                   turbulence_power=args.turbulence_power)

def robot_tank_env_creator(args):
    return gym.make("ALE/Robotank-v5")

class GradientLoggingCallback(DefaultCallbacks):
    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        """Called after RLlib has computed the gradients for a given training batch.

        Args:
            policy: Reference to the current Policy object.
            train_batch: The training data batch dict.
            result: A dict where you can store custom metrics for this batch.
            **kwargs: Forward-compatibility placeholder.
        """
        # Make sure this is a Torch policy
        if not hasattr(policy, "model") or not hasattr(policy.model, "parameters"):
            return

        total_grad_norm_sq = 0.0
        # Loop through each named parameter, compute per-layer gradient norm
        with torch.no_grad():
            for name, param in policy.model.named_parameters():
                if param.grad is not None:
                    layer_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm_sq += layer_grad_norm ** 2

                    # Store in result (to appear in TensorBoard logs)
                    result[f"grad_norm/{name}"] = layer_grad_norm

                    # Log to Weights & Biases (if active)
                    if wandb.run is not None:
                        wandb.log({f"grad_norm/{name}": layer_grad_norm})

        total_grad_norm = total_grad_norm_sq ** 0.5
        result["grad_norm"] = total_grad_norm
        if wandb.run is not None:
            wandb.log({"grad_norm": total_grad_norm})

def trial_dirname_creator(trial, alg, nn_type, lr, gamma, surrogate, identifier):
    return f"{alg}_{nn_type}_{lr}_{gamma}_{surrogate}_{identifier}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO model with rllib')
    parser.add_argument('--max_steps', type=int, default=5e6, help='Total number of train timesteps.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=.999, help='discount factor')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu', 'swish'], help='surrogate gradient fn to use')
    parser.add_argument('--id', type=str, default='1', help='unique id for saving results')
    parser.add_argument('--env', type=str, default='simple_spread', choices=['simple_spread', 'lunar_lander', 'robot_tank'], help='environment to train on')
    
    # Lunar Lander specific parameters
    parser.add_argument('--continuous', action='store_true', help='Use continuous actions for Lunar Lander')
    parser.add_argument('--gravity', type=float, default=-10.0, help='Gravity value for Lunar Lander')
    parser.add_argument('--enable-wind', action='store_true', help='Enable wind in Lunar Lander')
    parser.add_argument('--wind-power', type=float, default=15.0, help='Wind power for Lunar Lander')
    parser.add_argument('--turbulence-power', type=float, default=1.5, help='Turbulence power for Lunar Lander')
    
    args = parser.parse_args()

    ray.init()

    # Set up environment-specific configurations
    if args.env == 'simple_spread':
        env_name = "simple_spread_v3_experiments"
        env_creator_fn = simple_spread_env_creator
        env_wrapper = ParallelPettingZooEnv
        project_name = "simple_spread_experiments"
    elif args.env == 'lunar_lander':  # lunar_lander
        env_name = "LunarLander-v3"
        env_creator_fn = lunar_lander_env_creator
        env_wrapper = lambda env: env  # No wrapper needed for Lunar Lander
        project_name = "lunar_lander_experiments"
    elif args.env == 'robot_tank':
        env_name = "ALE/Robotank-v5"
        env_creator_fn = robot_tank_env_creator
        env_wrapper = lambda env: env  # No wrapper needed for Robot Tank
        project_name = "robot_tank_experiments"

    register_env(env_name, lambda config: env_wrapper(env_creator_fn(args)))
    # ModelCatalog.register_custom_model("LLS_RAY", LLS_RAY)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .env_runners(num_env_runners=8, rollout_fragment_length=2048)
        .training(
            train_batch_size=8 * 2048,
            lr=args.lr,
            lr_schedule=[[0, args.lr], [args.max_steps, args.lr * 0.1]],  # Linear decay from lr to 0.1*lr
            gamma=args.gamma,
            lambda_=0.97,
            # use_gae=True,
            clip_param=0.2,
            # grad_clip=None,
            entropy_coeff=0.02,
            vf_loss_coeff=0.5,
            minibatch_size=128,
            num_sgd_iter=4,
            grad_clip=0.5,
            # model={"custom_model": "LLS_RAY"}
        )
        .debugging(log_level="ERROR")
        .resources(num_gpus=2)
        .callbacks(GradientLoggingCallback)
    )

    # Add environment parameters to the config for logging
    if args.env == 'lunar_lander':
        env_config = {
            "continuous": args.continuous,
            "gravity": args.gravity,
            "enable_wind": args.enable_wind,
            "wind_power": args.wind_power,
            "turbulence_power": args.turbulence_power
        }
        config = config.environment(env=env_name, clip_actions=True, env_config=env_config)

    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

    # Include environment parameters in the run name if using Lunar Lander
    if args.env == 'lunar_lander':
        run_name = f"PPO_ann_{args.lr}_{args.gamma}_{args.activation}_{args.id}_g{args.gravity}_w{args.wind_power}_t{args.turbulence_power}"
        if args.continuous:
            run_name += "_cont"
        if args.enable_wind:
            run_name += "_wind"
    else:
        run_name = f"PPO_ann_{args.lr}_{args.gamma}_{args.activation}_{args.id}"

    wandb_callback = WandbLoggerCallback(
        project=project_name,
        log_config=True,            # logs the entire config to W&B
        name=run_name
    )

    partial_trial_dirname_creator = partial(trial_dirname_creator, alg='PPO', nn_type='ann', lr=args.lr, 
                                            gamma=args.gamma, surrogate=args.activation, identifier=args.id)

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": args.max_steps},
        checkpoint_freq=15,
        storage_path="~/marl/ray_results/" + env_name,
        trial_dirname_creator=partial_trial_dirname_creator,
        config=config.to_dict(),
        callbacks=[wandb_callback]
    )