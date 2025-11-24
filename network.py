import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
		For actor network: outputs action probabilities for discrete actions
		For critic network: outputs a single value
	"""
	def __init__(self, in_dim, out_dim, is_actor=False, action_space: str = "discrete"):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
				is_actor - whether this is the actor network (True) or critic network (False)

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.is_actor = is_actor
		self.action_space = action_space
		self.out_dim = out_dim

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)

		# Final layer output size must change based on action space type AND whether this is an actor
		# For continuous action actor: output means and log_stds (double the action_dim)
		# For discrete action actor: output logits for each action
		# For critic: always output single value regardless of action space
		if is_actor and action_space == "continuous":
			# Continuous actor outputs both means and log_stds
			self.layer3 = nn.Linear(64, out_dim * 2)
		else:
			# Discrete actor, or any critic
			self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
				        For actor: action probabilities (softmax)
				        For critic: value estimate
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		# For continuous actor, return means and log stddevs separately
		if self.is_actor and self.action_space == "continuous":
			means = output[..., :self.out_dim]
			log_stds = output[..., self.out_dim:]
			# Clamp for numerical stability (prevents NaN/Inf)
			means = torch.clamp(means, -2, 2)
			log_stds = torch.clamp(log_stds, -20, 2)
			return means, log_stds
		# For discrete actor, apply softmax for actor network to get action probabilities
		if self.is_actor:
			output = F.softmax(output, dim=-1)

		return output