import torch
from torch import nn
from lls_utils import AdamWScheduleFree, SGDScheduleFree
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

__all__ = ["LLS_layer", "LinearBlock", "ConvBlock", "ConvDWBlock"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_frequency_matrix(num_rows, num_cols, min_freq=50, max_freq=2000, freq=None):
    if freq is None:
        frequencies = torch.linspace(min_freq, max_freq, num_rows).unsqueeze(1).to(device)
    else:
        frequencies = freq
    # phases = torch.randn(num_rows, 1) * 2 * 3.14159
    t = torch.arange(num_cols).float().unsqueeze(0).to(device)
    sinusoids = torch.sin(frequencies * t )
    return sinusoids

# def generate_frequency_matrix(num_rows, num_cols, min_freq=100, max_freq=2000, freq=None):
#     frequencies = torch.linspace(min_freq, max_freq, num_rows).unsqueeze(1)
#     # phases = torch.randn(num_rows, 1) * 2 * 3.14159
#     t = torch.arange(num_cols).float().unsqueeze(0)
#     sinusoids = torch.cos(np.pi*frequencies * (t + 0.5)/num_cols)
#     return sinusoids

def compute_LocalLosses(activation, labels, local_classifier, temperature=1, label_smoothing=0.0, act_size=8):
    batch_size = activation.size(0)
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation, act_size).view(batch_size, -1)
    local_classifier_red = F.adaptive_avg_pool1d(local_classifier, latents.size(1))
    layer_pred = torch.matmul(latents, local_classifier_red.T)
    loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    return loss

def layer_pred_LLS(activation, act_size=1, n_classes=10, modulation_term=None, modulation=False, freq=None, waveform="cosine"):
    batch_size = activation.size(0) if activation.dim() > 1 else 1
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation.view(batch_size, -1), act_size).view(batch_size, -1)
    basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=512, freq=freq).to(device)
    # basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=latents.size(1) - 50).to(device)
    if waveform == "square":
        basis = torch.sign(basis)

    latents = F.normalize(latents, dim=1)
    layer_pred = torch.matmul(latents, basis.T)
    if modulation == 1:
        layer_pred = modulation_term*layer_pred
    if modulation == 2:
        layer_pred = torch.matmul(layer_pred, modulation_term)

    return layer_pred

def compute_LLS(activation, labels, temperature=1, label_smoothing=0.0, act_size=1, n_classes=10,
                modulation_term=None, modulation=False, freq=None, waveform="cosine", loss_type="cross_entropy"):
    layer_pred = layer_pred_LLS(activation, act_size, n_classes, modulation_term, modulation, freq, waveform)
    if loss_type == "cross_entropy":
        loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    elif loss_type == "mse":
        loss = torch.nn.functional.mse_loss(layer_pred, labels.to(device))
    else:
        raise ValueError(f"{loss_type} is not supported")
    return loss


def compute_LLS_Random(activation, labels, random_basis, temperature=1, label_smoothing=0.0, act_size=8):
    batch_size = activation.size(0)
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation, act_size).view(batch_size, -1)
    random_basis_red = F.adaptive_avg_pool1d(random_basis, latents.size(1))
    layer_pred = torch.matmul(latents, random_basis_red.T)
    loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    return loss


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, training_mode=None):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        if training_mode == "DRTP":
            self.nonlinearity = nn.Tanh()
        else:
            self.nonlinearity = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinearity(x)
        return x


class LLS_layer(nn.Module):
    def __init__(self, block:nn.Module, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="LLS",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0,
                 waveform="cosine", hidden_dim = 2048, reduced_set=20, pooling_size = 4, scaler = False, loss_type="cross_entropy"):
        super(LLS_layer, self).__init__()
        self.block = block
        self.lr = lr
        self.n_classes = n_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.training_mode = training_mode
        self.patience = patience
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.waveform = waveform
        self.milestones = milestones
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.reduced_set = reduced_set
        self.pooling_size = pooling_size
        self.scaler = None
        self.loss_type = loss_type
        self.feedback = None
        self.modulation = None
        self.modulation_mode = None

        if  "LocalLosses" in self.training_mode:
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])), requires_grad=True)
            self.training_mode = "LocalLosses"

        elif "LLS_Random" in self.training_mode:
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])),
                                         requires_grad=False)
        elif "LLS_M_Random" in self.training_mode:
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])),
                                         requires_grad=False)
            self.modulation = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 1])), requires_grad=True)
        elif "LLS_MxM_Random" in self.training_mode:
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])),
                                         requires_grad=False)
            self.modulation = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, n_classes])), requires_grad=True)
        elif "LLS_M" in self.training_mode:
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes])), requires_grad=True)
            self.modulation_mode = 1
        elif "LLS_MxM" in self.training_mode:
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, n_classes])), requires_grad=True)
            self.modulation_mode = 2
        elif "LLS_MxM_reduced" in self.training_mode:
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([self.reduced_set, n_classes])), requires_grad=True)
            self.modulation_mode = 2

        # Optimizer
        if training_mode != "BP":
            if isinstance(optimizer, (tuple, list)):
                optimizer = optimizer[0]  # Take first element if it's a tuple/list
                
            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                           nesterov=nesterov)
            elif optimizer == "Adam":
                self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer == "AdamWSF":
                self.optimizer = AdamWScheduleFree(self.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer == "SGDSF":
                self.optimizer = SGDScheduleFree(self.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"{optimizer} is not supported")

            if lr_scheduler == "MultiStepLR":
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            elif lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience, verbose = True)

        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0

    def record_statistics(self, loss, batch_size):
        self.loss_hist += loss.item() * batch_size
        self.samples += batch_size
        self.loss_avg = self.loss_hist / self.samples if self.samples > 0 else 0

    def reset_statistics(self):
        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0
    
    def layer_update(self, out, labels, feedback):
        if self.training_mode == "LLS":
                temperature = self.temperature
                label_smoothing = self.label_smoothing
                loss = compute_LLS(out, labels, temperature, label_smoothing, self.pooling_size,
                                   self.n_classes, waveform=self.waveform, loss_type=self.loss_type)

        elif self.training_mode == "LLS_M" or self.training_mode == "LLS_MxM" or self.training_mode == "LLS_MxM_reduced":
            temperature = self.temperature
            label_smoothing = self.label_smoothing
            loss = compute_LLS(out, labels, temperature, label_smoothing, self.pooling_size,
                                self.n_classes if self.training_mode != "LLS_MxM_reduced" else self.reduced_set,
                                modulation=self.modulation_mode, modulation_term=self.feedback,
                                waveform=self.waveform, loss_type=self.loss_type)

        elif self.training_mode == "LLS_Random" or self.training_mode == "LLS_M_Random" or self.training_mode == "LLS_MxM_Random":
            temperature = self.temperature
            label_smoothing = self.label_smoothing
            if self.training_mode == "LLS_Random":
                feedback = self.feedback
            elif self.training_mode == "LLS_M_Random":
                feedback = self.modulation * self.feedback
            else:
                feedback = torch.matmul(self.modulation, self.feedback)
            loss = compute_LocalLosses(out, labels, feedback, temperature, label_smoothing, act_size=self.pooling_size)

        elif self.training_mode == "LocalLosses":
            temperature = self.temperature
            label_smoothing = self.label_smoothing
            loss = compute_LocalLosses(out, labels, self.feedback, temperature, label_smoothing, act_size=self.pooling_size)
        else:
            raise NotImplementedError(f"Unknown training mode: {self.training_mode}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.record_statistics(loss.detach(), out.size(0))

    def layer_update_continuous_ppo(self, action_params, actions, old_log_probs, advantage, clip, ent_coef, max_grad_norm):
        """
        Layer-wise PPO update for continuous actions using Gaussian distributions.
        
        Parameters:
            action_params: Tensor of shape [batch, action_dim * 2] containing means and log_stds
            actions: Actual actions taken [batch, action_dim]
            old_log_probs: Log probabilities from previous policy [batch]
            advantage: Advantage estimates [batch]
            clip: PPO clip parameter
            ent_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        
        # Split into means and log_stds
        action_dim = actions.shape[-1]
        means = action_params[..., :action_dim]
        log_stds = action_params[..., action_dim:]
        
        # Clamp for stability
        means = torch.clamp(means, -2, 2)
        # Clamp to enforce minimum std of 0.1 to prevent policy collapse
        log_stds = torch.clamp(log_stds, -2.3, 2)
        stds = log_stds.exp()
        
        # Create Normal distribution
        dist = Normal(means, stds)
        
        # Compute log probability of the taken actions
        log_prob = dist.log_prob(actions).sum(dim=-1)
        
        # PPO loss calculation
        log_ratio = log_prob - old_log_probs
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy regularization
        entropy_loss = dist.entropy().sum(dim=-1).mean()
        actor_loss = actor_loss - ent_coef * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.block.parameters(), max_grad_norm)
        self.optimizer.step()
        
        # Record statistics
        self.record_statistics(actor_loss.detach(), actions.size(0))
        
        return actor_loss

    def forward(self, x, labels=None, feedback=None, x_err=None):
        training = self.training

        if self.training_mode == "BP" or not training or labels is None:
            return self.block(x)
        else:
            out = self.block(x.detach())
            self.layer_update(out, labels, feedback)

            return out.detach()