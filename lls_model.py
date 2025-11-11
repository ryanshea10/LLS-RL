import torch
from torch import nn
from lls_utils import AdamWScheduleFree
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Categorical
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
    def __init__(self, num_inputs, num_outputs, is_actor, *args, **kwargs):
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

        self.linear_out = nn.Linear(self.hidden, num_outputs, bias=bias)
        self.to(device)

        # Feedback matrix
        self.feedback = None

        if self.training_mode != "BP":
            if self.feedback is not None and self.training_mode != "SharedClassifier":
                params_dict = [{"params": self.linear_out.parameters()}, {"params": self.feedback}]
            else:
                params_dict = [{"params": self.linear_out.parameters()}]

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
        layer_preds = [Categorical(h.to(device)) for h in hidden_states]
        layer_log_probs = [lp.log_prob(act.to(device)) for lp, act in zip(layer_preds, acts)]
        
        for i in range(len(layer_log_probs)): # calculate ppo loss for each hidden layer
            log_ratio = layer_log_probs[i] - old_layer_log_probs[i].to(device)
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
        surr1 = ratio * advantage.to(device)
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage.to(device)
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = dist.entropy().mean()
        actor_loss = actor_loss - ent_coef * entropy_loss
        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()
    
    def online_forward(self, x):
        hidden_states = []
        x = x.to(device)

        x = self.linear_block1(x)
        # Use feedback_linear if available, otherwise fall back to feedback parameter
        linear_layer1 = self.linear_block1.feedback_linear if self.linear_block1.feedback_linear is not None else None
        layer_pred = layer_pred_LLS(x, act_size=self.hidden, n_classes=self.n_classes, 
                                    modulation_term=self.linear_block1.feedback, 
                                    modulation=self.linear_block1.modulation_mode, 
                                    freq=None, waveform=self.waveform, 
                                    linear_layer=linear_layer1)
        hidden_states.append(layer_pred[0].clone())
        
        x = self.linear_block2(x.detach())
        # Use feedback_linear if available, otherwise fall back to feedback parameter
        linear_layer2 = self.linear_block2.feedback_linear if self.linear_block2.feedback_linear is not None else None
        layer_pred = layer_pred_LLS(x, act_size=self.hidden, n_classes=self.n_classes, 
                                    modulation_term=self.linear_block2.feedback, 
                                    modulation=self.linear_block2.modulation_mode, 
                                    freq=None, waveform=self.waveform, 
                                    linear_layer=linear_layer2)
        hidden_states.append(layer_pred[0].clone())
        
        x = self.linear_out(x.detach())

        if self.is_actor:
            hidden_states = [F.softmax(h, dim=-1) for h in hidden_states]
            return F.softmax(x, dim=-1), hidden_states

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
        
        if self.is_actor:
            return F.softmax(x, dim=-1)
        return x
