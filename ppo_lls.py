import gymnasium as gym
import time
import wandb
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, Normal

class PPO_LLS:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, policy_class, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """

        self.action_space = hyperparameters.get('action_space')
        if self.action_space is None:
            raise ValueError(
                "action_space must be included in hyperparameters used to initialize PPO_LLS()"
            )

        # Make sure the environment is compatible with our code
        assert(type(env.observation_space) == gym.spaces.Box)
        if self.action_space == "discrete":
            assert(type(env.action_space) == gym.spaces.Discrete)
        else:
            assert(type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        if self.action_space == "discrete":
            self.act_dim = env.action_space.n  # Number of discrete actions
        else:
            self.act_dim = env.action_space.shape[0] # Continuous actions

        # Initialize actor and critic networks
        self.actor = policy_class(
            self.obs_dim, self.act_dim, is_actor=True, 
            action_space=self.action_space,
            training_mode="PPO_LLS_MxM", optimizer="AdamWSF")  # ALG STEP 1
        self.critic = policy_class(
            self.obs_dim, 1, is_actor=False, 
            action_space=self.action_space, training_mode="LLS_MxM", 
            optimizer="AdamWSF", loss_type='mse', lr=hyperparameters['lr'])

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],    # losses of critic network in current iteration
            'mean_values': [],      # mean value predictions
            'mean_advantages': [],  # mean advantages
            'mean_ratios': [],      # mean policy ratios
        }

        # Initialize wandb
        wandb.init(
            project="ppo-training",
            config={
                "timesteps_per_batch": self.timesteps_per_batch,
                "max_timesteps_per_episode": self.max_timesteps_per_episode,
                "n_updates_per_iteration": self.n_updates_per_iteration,
                "learning_rate": self.lr,
                "gamma": self.gamma,
                "clip": self.clip,
                "seed": self.seed,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            }
        )

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_layer_log_probs = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = V.cpu().detach() + A_k.cpu()

            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                # Update learning rate with linear decay
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0)

                np.random.shuffle(inds) # Shuffle indices to split into minibatches
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]
                    mini_layer_log_probs = batch_layer_log_probs[idx]

                    self.actor.ppo_update(mini_obs, mini_acts, mini_layer_log_probs, mini_advantage, mini_log_prob, 
                                          self.clip, self.ent_coef, self.max_grad_norm)
                    # V = self.critic(mini_obs, labels=mini_rtgs)
                    V = self.critic(mini_obs, labels=mini_rtgs.view(-1, 1))

                    # Log losses and metrics
                    # self.logger['actor_losses'].append(actor_loss.detach())
                    # self.logger['critic_losses'].append(critic_loss.detach())
                    self.logger['mean_values'].append(V.mean().cpu().detach())
                    self.logger['mean_advantages'].append(A_k.mean().cpu().detach())
                    # self.logger['mean_ratios'].append(ratios.mean().detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './saved_models/ppo_actor.pt')
                torch.save(self.critic.state_dict(), './saved_models/ppo_critic.pt')

        # Close wandb run
        wandb.finish()

    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.

            Parameters:
                None

            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []
        batch_layer_log_probs = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # values collected per episode
            ep_dones = [] # dones collected per episode

            # Reset the environment. sNote that obs is short for observation. 
            obs, _ = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()
                ep_dones.append(done)

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob, layer_log_probs = self.get_action(obs)
                val = self.critic(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)

                # Don't really care about the difference between terminated or truncated in this, so just combine them
                done = terminated | truncated

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(val)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                batch_layer_log_probs.append(layer_log_probs)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)

        if self.action_space == "discrete":
            batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long)  # Convert to numpy array first
        else:
            batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)

        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        batch_layer_log_probs = torch.tensor(np.array(batch_layer_log_probs), dtype=torch.float)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_layer_log_probs
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """

        if self.action_space == "discrete":
            # Query the actor network for action probabilities
            action_probs, hidden_states = self.actor(obs)
            layer_preds = [Categorical(h) for h in hidden_states]

            # Create a categorical distribution with the action probabilities
            dist = Categorical(action_probs)

            # Sample an action from the distribution
            action = dist.sample()

            # Calculate the log probability for that action
            log_prob = dist.log_prob(action)
            layer_log_probs = [lp.log_prob(action).cpu().detach() for lp in layer_preds]

            # Return the sampled action and the log probability of that action in our distribution
            return action.cpu().detach().numpy(), log_prob.cpu().detach(), layer_log_probs

        else: # Continuous env
            (means, log_stds), hidden_states = self.actor(obs)
            stds = log_stds.exp()
            dist = Normal(means, stds)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # For continuous env, hidden states are regression outputs
            # FIXME: Layer log probs need different computation (TBD based on LLS method)
            layer_log_probs = []  # Placeholder
            
            return action.cpu().detach().numpy(), log_prob.cpu().detach(), layer_log_probs

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs
        V = self.critic(batch_obs).squeeze()

        if self.action_space == "discrete":
            # Calculate the log probabilities of batch actions using most recent actor network
            action_probs = self.actor(batch_obs)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(batch_acts)
            entropy = dist.entropy()
        else:
            (means, log_stds), _ = self.actor(batch_obs)
            stds = log_stds.exp()
            dist = Normal(means, stds)
            log_probs = dist.log_prob(batch_acts).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, entropy

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
        self.num_minibatches = 4                        # Number of minibatches to split the batch into
        self.ent_coef = 0.01                            # Entropy regularization coefficient
        self.max_grad_norm = 0.5                        # Maximum gradient norm to clip gradients at
        self.lam = 0.97                                  # Lambda for GAE

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])
        avg_value = np.mean([v.float().mean() for v in self.logger['mean_values']])
        avg_advantage = np.mean([a.float().mean() for a in self.logger['mean_advantages']])
        avg_ratio = np.mean([r.float().mean() for r in self.logger['mean_ratios']])

        # Log to wandb
        wandb.log({
            "iteration": i_so_far,
            "timesteps_so_far": t_so_far,
            "average_episode_length": avg_ep_lens,
            "average_episode_return": avg_ep_rews,
            "average_actor_loss": avg_actor_loss,
            "average_critic_loss": avg_critic_loss,
            "average_value": avg_value,
            "average_advantage": avg_advantage,
            "average_ratio": avg_ratio,
            "iteration_time": float(delta_t),
            # "learning_rate": self.actor_optim.param_groups[0]['lr']
        })

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['mean_values'] = []
        self.logger['mean_advantages'] = []
        self.logger['mean_ratios'] = []