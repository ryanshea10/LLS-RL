import torch
from torch import nn
from lls_utils import AdamWScheduleFree
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Categorical, Normal
from lls_layers import LLS_layer, LinearBlock, layer_pred_LLS
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import logging
import numpy as np

__all__ = ["LLS_RAY"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLS_RAY(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        
        self.actor = LLS_Model(obs_space.shape[0], num_outputs, is_actor=False, **kwargs) # the model is the actor but we don't want softmax applied
        self.critic = LLS_Model(obs_space.shape[0], 1, is_actor=False, **kwargs)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = self.actor(input_dict["obs"])
        v = self.critic(input_dict["obs"])
        self._value_out = v
        return x, state
    
    def value_function(self):
        return self._value_out.flatten()


class LLS_Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, is_actor, action_space: str = "discrete", *args, **kwargs):
        nn.Module.__init__(self)
        
        # Extract parameters from kwargs with defaults
        bias = kwargs.get('bias', True)
        lr = kwargs.get('lr', 1e-1)
        n_classes = kwargs.get('n_classes', 4)
        momentum = kwargs.get('momentum', 0)
        weight_decay = kwargs.get('weight_decay', 0)
        nesterov = kwargs.get('nesterov', False)
        optimizer = kwargs.get('optimizer', "SGD")
        milestones = kwargs.get('milestones', [10, 30, 50])
        gamma = kwargs.get('gamma', 0.1)
        training_mode = kwargs.get('training_mode', "BP")
        lr_scheduler = kwargs.get('lr_scheduler', "MultiStepLR")
        patience = kwargs.get('patience', 20)
        temperature = kwargs.get('temperature', 1)
        label_smoothing = kwargs.get('label_smoothing', 0.0)
        dropout = kwargs.get('dropout', 0.0)
        waveform = kwargs.get('waveform', "cosine")
        loss_type = kwargs.get('loss_type', "cross_entropy")

        self.hidden = 64 # number of hidden neurons
        self.in_features = num_inputs
        self.out_features = num_outputs
        self.n_classes = num_outputs
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.reduced_set = 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None
        self.is_actor = is_actor # if true, the model is the actor and we want softmax applied to outputs
        self.lr = lr
        self.action_space = action_space

        linear_block1 = LinearBlock(self.in_features, self.hidden)
        self.linear_block1 = LLS_layer(block=linear_block1, lr=lr, n_classes=self.n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=self.hidden, reduced_set=self.reduced_set, pooling_size=self.hidden,
                                     loss_type=self.loss_type)
        
        linear_block2 = LinearBlock(self.hidden, self.hidden)
        self.linear_block2 = LLS_layer(block=linear_block2, lr=lr, n_classes=self.n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=self.hidden, reduced_set=self.reduced_set, pooling_size=self.hidden,
                                     loss_type=self.loss_type)

        if is_actor and action_space == "continuous":
            self.linear_out = nn.Linear(self.hidden, num_outputs * 2, bias=bias)
        else:
            self.linear_out = nn.Linear(self.hidden, num_outputs, bias=bias)

        # If using continuous action space with PPO training, need projection heads from hidden layers to action space
        # Not necessary for standard backprop (--mode train)
        if is_actor and action_space == "continuous" and "PPO" in training_mode:
            # Project hidden layer 1 output to (means, log_stds)
            self.layer1_to_action = nn.Linear(self.hidden, num_outputs * 2, bias=bias)
            # Project hidden layer 2 output to (means, log_stds)
            self.layer2_to_action = nn.Linear(self.hidden, num_outputs * 2, bias=bias)

            # Verify initialization
            if torch.isnan(self.layer1_to_action.weight).any():
                print("ERROR: layer1_to_action weights contain NaN at initialization!")
            if torch.isnan(self.layer2_to_action.weight).any():
                print("ERROR: layer2_to_action weights contain NaN at initialization!")
        
        self.to(device)

        # Feedback matrix
        self.feedback = None

        if self.training_mode != "BP":
            if self.feedback is not None and self.training_mode != "SharedClassifier":
                params_dict = [{"params": self.linear_out.parameters()}, {"params": self.feedback}]
            else:
                params_dict = [{"params": self.linear_out.parameters()}]

            # For continuous action space, create separate optimizers for projection heads
            if is_actor and action_space == "continuous" and "PPO" in training_mode:

                # # DEBUG
                # print(f"DEBUG: Creating projection head optimizers for continuous action space")

                if optimizer == "SGD":
                    self.layer1_proj_optimizer = optim.SGD(
                        self.layer1_to_action.parameters(),
                        lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=nesterov)
                    self.layer2_proj_optimizer = optim.SGD(
                        self.layer2_to_action.parameters(),
                        lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=nesterov)

                elif optimizer == "Adam":
                    # Use lower learning rate for projection head optimizers
                    proj_lr = 0.00005 # 0.0001
                    self.layer1_proj_optimizer = optim.Adam(self.layer1_to_action.parameters(),
                        lr=proj_lr, weight_decay=weight_decay)
                    self.layer2_proj_optimizer = optim.Adam(self.layer2_to_action.parameters(),
                        lr=proj_lr, weight_decay=weight_decay)

                elif optimizer == "AdamWSF":
                    self.layer1_proj_optimizer = AdamWScheduleFree(
                        self.layer1_to_action.parameters(),
                        lr=lr, weight_decay=weight_decay)
                    self.layer2_proj_optimizer = AdamWScheduleFree(
                        self.layer2_to_action.parameters(),
                        lr=lr, weight_decay=weight_decay)
                
            if optimizer == "SGD":
                self.optimizer = optim.SGD(params_dict, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                           nesterov=nesterov)
            elif optimizer == "Adam":
                self.optimizer = optim.Adam(params_dict, lr=lr, weight_decay=weight_decay)
            elif optimizer == "AdamWSF":
                self.optimizer = AdamWScheduleFree(params_dict, lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"{optimizer} is not supported")

            if lr_scheduler == "MultiStepLR":
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            elif lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience, verbose = True)
        
        if self.loss_type == "cross_entropy":
            self.loss_fn = torch.nn.functional.cross_entropy
        elif self.loss_type == "mse":
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"{self.loss_type} is not supported")

    def lr_scheduler_step(self, loss_avg=None):
        if loss_avg is None:
            self.lr_scheduler.step()
            self.policy_linear_block1.lr_scheduler.step()
            self.value_linear_block1.lr_scheduler.step()
        else:
            self.lr_scheduler.step(loss_avg)
            self.policy_linear_block1.lr_scheduler.step(loss_avg)
            self.value_linear_block1.lr_scheduler.step(loss_avg)

    def reset_statistics(self):
        self.policy_linear_block1.reset_statistics()
        self.value_linear_block1.reset_statistics()

    def print_stats(self):
        logging.info(
            "Losses - L1: {0:.4f} - L2: {1:.4f}".format(
                self.policy_linear_block1.loss_avg, self.value_linear_block1.loss_avg))

    def optimizer_eval(self):
        if isinstance(self.optimizer, AdamWScheduleFree):
            self.optimizer.eval()
            self.policy_linear_block1.optimizer.eval()
            self.value_linear_block1.optimizer.eval()

    def optimizer_train(self):
        if isinstance(self.optimizer, AdamWScheduleFree):
            self.optimizer.train()
            self.policy_linear_block1.optimizer.train()
            self.value_linear_block1.optimizer.train()
    
    def ppo_update(self, obs, acts, old_layer_log_probs, advantage, old_log_probs, clip, ent_coef, max_grad_norm):
        x, hidden_states = self.online_forward(obs)

        if self.action_space == "discrete":
            layer_preds = [Categorical(h.to(device)) for h in hidden_states]
            layer_log_probs = [lp.log_prob(act.to(device)) for lp, act in zip(layer_preds, acts)]
            
            for i in range(len(layer_log_probs)): # calculate ppo loss for each hidden layer
                log_ratio = layer_log_probs[i] - old_layer_log_probs[:, i].to(device)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantage[i]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage[i]
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = layer_preds[i].entropy().mean()
                actor_loss = actor_loss - ent_coef * entropy_loss
                self.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()

            dist = Categorical(x) # calculate ppo loss for the output layer
            log_ratio = dist.log_prob(acts.to(device)) - old_log_probs.to(device)
            ratio = torch.exp(log_ratio)
            # Use same advantage as layers (they're all the same, just use first one)
            surr1 = ratio * advantage[0].to(device)
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage[0].to(device)
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = dist.entropy().mean()
            actor_loss = actor_loss - ent_coef * entropy_loss
            self.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()

        else: # Continuous env
            # Update projection heads layer-wise (similar to discrete case)
            for i in range(len(hidden_states)):
                # Split hidden_states[i] into means and log_stds
                action_dim = acts.shape[-1]
                means = hidden_states[i][..., :action_dim]
                log_stds = hidden_states[i][..., action_dim:]

                # Debug: Check for NaN before clamping
                # if torch.isnan(hidden_states[i]).any():
                #     print(f"ERROR: NaN in hidden_states[{i}] BEFORE clamping!")
                #     print(f"Range: [{hidden_states[i].min()}, {hidden_states[i].max()}]")
                
                # Clamp for stability
                means = torch.clamp(means, -2, 2)
                # Clamp log_stds to enforce minimum std of 0.1 (exp(-2.3) ≈ 0.1)
                # to prevent policy collapse
                log_stds = torch.clamp(log_stds, -2.3, 2)

                # # Debug: Check for NaN after clamping
                # if torch.isnan(means).any():
                #     print(f"ERROR: NaN in means[{i}] AFTER clamping!")
                # if torch.isnan(log_stds).any():
                #     print(f"ERROR: NaN in log_stds[{i}] AFTER clamping!")

                stds = log_stds.exp()

                # # Debug: Check for NaN/inf in stds
                # if torch.isnan(stds).any():
                #     print(f"ERROR: NaN in stds[{i}]! log_stds range: [{log_stds.min()}, {log_stds.max()}]")
                # if torch.isinf(stds).any():
                #     print(f"ERROR: Inf in stds[{i}]! log_stds range: [{log_stds.min()}, {log_stds.max()}]")
                # # Debug: Check if stds are collapsing (too small)
                # if (stds < 0.01).any():
                #     print(f"WARNING: Very small stds in layer[{i}]! Min std={stds.min():.6f}, this causes huge negative log_probs and gradient explosion")
                
                # Create distribution and compute log prob
                dist = Normal(means, stds)
                log_prob = dist.log_prob(acts.to(device)).sum(dim=-1)
                
                # PPO loss
                log_ratio = log_prob - old_layer_log_probs[:, i].to(device)
                # # Debug: Warn about large log_ratios
                # if log_ratio.abs().max() > 10:
                #     print(f"WARNING: Large log_ratio in layer[{i}]! Range: [{log_ratio.min():.2f}, {log_ratio.max():.2f}]")
                # Clip log_ratio
                # Clip log_ratio to prevent exp overflow (exp(20) = 4e8, exp(50) = 5e21 = inf)
                log_ratio = torch.clamp(log_ratio, -20, 20)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantage[i]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage[i]
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = dist.entropy().sum(dim=-1).mean()
                actor_loss = actor_loss - ent_coef * entropy_loss
                
                # Update corresponding projection head
                if i == 0:
                    self.layer1_proj_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Zero LLS layer gradients to prevent accumulation
                    for param in self.linear_block1.parameters():
                        if param.grad is not None:
                            param.grad.zero_()

                    # # Debug: Check gradient magnitudes before clipping
                    # total_norm = 0
                    # for p in self.layer1_to_action.parameters():
                    #     if p.grad is not None:
                    #         param_norm = p.grad.data.norm(2)
                    #         total_norm += param_norm.item() ** 2
                    #         total_norm = total_norm ** 0.5
                    # if total_norm > 100:
                    #     print(f"WARNING: Large gradients in layer1 projection! Norm={total_norm}")

                    nn.utils.clip_grad_norm_(self.layer1_to_action.parameters(), max_grad_norm)
                    self.layer1_proj_optimizer.step()

                    # # Debug: Check if parameters became NaN after step
                    # if torch.isnan(self.layer1_to_action.weight).any():
                    #     print(f"ERROR: layer1_to_action weights became NaN after optimizer step!")

                elif i == 1:
                    self.layer2_proj_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Zero LLS layer gradients to prevent accumulation
                    for param in self.linear_block2.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                    nn.utils.clip_grad_norm_(self.layer2_to_action.parameters(), max_grad_norm)
                    self.layer2_proj_optimizer.step()
            
            # Output layer (continuous)
            means, log_stds = x
            stds = log_stds.exp()
            dist = Normal(means, stds)
            log_probs = dist.log_prob(acts.to(device)).sum(dim=-1)
            log_ratio = log_probs - old_log_probs.to(device)
            # Clip log_ratio to prevent exp overflow
            log_ratio = torch.clamp(log_ratio, -20, 20)
            ratio = torch.exp(log_ratio)
            # Use same advantage as layers (they're all the same, just use first one)
            surr1 = ratio * advantage[0].to(device)
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage[0].to(device)
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = dist.entropy().sum(dim=-1).mean()
            actor_loss = actor_loss - ent_coef * entropy_loss
            
            self.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
    
    def online_forward(self, x):
        hidden_states = []
        x = x.to(device)

        x = self.linear_block1(x)
        layer_pred = layer_pred_LLS(x, act_size=self.hidden, n_classes=self.n_classes, 
                                    modulation_term=self.linear_block1.feedback, 
                                    modulation=self.linear_block1.modulation_mode, 
                                    freq=None, waveform=self.waveform)
        
        if self.is_actor and self.action_space == "continuous":
            # Project hidden activations (not LLS predictions) to action space (means + log_stds)
            # Use x (the hidden layer output) which has shape [batch, 64]
            # # Debug: Check for NaN
            # if torch.isnan(x).any():
            #     print(f"ERROR: NaN in LLS layer1 output x! min={x.min()}, max={x.max()}, mean={x.mean()}")

            action_pred = self.layer1_to_action(x.detach())

            # if torch.isnan(action_pred).any():
            #     print(f"ERROR: NaN in layer1 projection output! Input x range: [{x.min()}, {x.max()}]")
            #     print(f"Projection weight range: [{self.layer1_to_action.weight.min()}, {self.layer1_to_action.weight.max()}]")
            
            # DEBUG
            # if self.layer1_to_action.bias is not None:
                # print(f"Projection bias range: [{self.layer1_to_action.bias.min()}, {self.layer1_to_action.bias.max()}]")

            hidden_states.append(action_pred)
        else:
            hidden_states.append(layer_pred[0].clone())
        
        x = self.linear_block2(x) 
        layer_pred = layer_pred_LLS(x, act_size=self.hidden, n_classes=self.n_classes, 
                                    modulation_term=self.linear_block2.feedback, 
                                    modulation=self.linear_block2.modulation_mode, 
                                    freq=None, waveform=self.waveform)
        
        if self.is_actor and self.action_space == "continuous":
            # Project hidden activations (not LLS predictions) to action space (means + log_stds)
            # Use x (the hidden layer output) which has shape [batch, 64]
            action_pred = self.layer2_to_action(x.detach())
            hidden_states.append(action_pred)
        else:
            hidden_states.append(layer_pred[0].clone())
        
        x = self.linear_out(x.detach())

        if self.is_actor and self.action_space == "discrete":
            hidden_states = [F.softmax(h, dim=-1) for h in hidden_states]
            return F.softmax(x, dim=-1), hidden_states
        elif self.is_actor:
            # Output layer: split into means and log_stds
            means = x[..., :self.out_features]
            log_stds = x[..., self.out_features:]
            means = torch.clamp(means, -2, 2)
            # Clamp to enforce minimum std of 0.1 to prevent policy collapse
            log_stds = torch.clamp(log_stds, -2.3, 2)
            
            # Hidden states already contain raw action parameters (means + log_stds concatenated)
            # No need to process them here - they'll be processed in layer_update_continuous_ppo
            return (means, log_stds), hidden_states

        return x, hidden_states

    def forward(self, x, labels=None):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = x.to(device)
        
        if "PPO" in self.training_mode:
            return self.online_forward(x)
        
        training = self.training
        x = self.linear_block1(x, labels=labels, feedback=self.feedback)
        x = self.linear_block2(x, labels=labels, feedback=self.feedback)
        x = self.linear_out(x)

        if self.training_mode != "BP" and training and labels is not None:
            loss = self.loss_fn(x, labels.to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.is_actor and self.action_space == "discrete":
            return F.softmax(x, dim=-1)
        elif self.is_actor:
            means = x[..., :self.out_features]
            log_stds = x[..., self.out_features:]
            means = torch.clamp(means, -2, 2)
            # Clamp to enforce minimum std of 0.1 to prevent policy collapse
            log_stds = torch.clamp(log_stds, -2.3, 2)
            return means, log_stds

        return x