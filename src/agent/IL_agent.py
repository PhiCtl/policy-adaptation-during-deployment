import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent.encoder import make_encoder

LOG_FREQ = 10000

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def make_il_agent(obs_shape, action_shape, args, dynamics_input_shape):
    return SacSSAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        dynamics_input_shape=dynamics_input_shape,
        dynamics_output_shape=args.dynamics_output_shape,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        ss_lr=args.ss_lr,
        ss_update_freq=args.ss_update_freq,
        num_layers=args.num_layers,
        num_shared_layers=args.num_shared_layers,
        num_filters=args.num_filters,
    )


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


#IL agent is a regresson that want to predict the following actions
class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, dynamics_output_shape, hidden_dim,
        encoder_feature_dim, num_layers, num_filters, num_shared_layers
    ):
        super().__init__()

        self.encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )

        # Concatenate dynamics and obs
        self.input_feat_dim = self.encoder.feature_dim + dynamics_output_shape
        self.trunk = nn.Sequential(
            nn.Linear(self.input_feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0]), nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, obs, dyn_feat, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        joint_input = torch.cat([obs, dyn_feat], dim=1)
        mu = self.trunk(joint_input)

        return mu

    def tie_actor_from(self, source):
        """Tie actor parameters to another actor"""
        # Both objects should be actor models
        assert type(self) == type(source)

        # We copy actor encoder
        self.encoder.tie_encoder_from(source.encoder)
        # Copy linear layers
        for tgt, src in zip(self.trunk, source.trunk):
            if isinstance(tgt, nn.Linear) and isinstance(src, nn.Linear):
                tie_weights(tgt, src)


class DomainSpecific(nn.Module):
    """MLP specific domain network."""

    def __init__(self, dynamics_input_shape, dynamics_output_shape, hidden_dim=20):
        super().__init__()

        self.specific = nn.Sequential(nn.Linear(dynamics_input_shape, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, dynamics_output_shape))

    def forward(self, gt):
        res = self.specific(gt)
        return res

class InvFunction(nn.Module):
    """MLP for inverse dynamics model."""
    def __init__(self, obs_dim, action_dim, dynamics_output_shape, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2*obs_dim + dynamics_output_shape, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, h, h_next, dyn_feat):
        joint_input = torch.cat([h, h_next, dyn_feat], dim=1)
        return self.trunk(joint_input)

    def tie_inv_from(self, source):
        assert type(self) == type(source)
        # Copy linear layers
        for tgt, src in zip(self.trunk, source.trunk):
            if isinstance(tgt, nn.Linear) and isinstance(src, nn.Linear):
                tie_weights(tgt, src)


class SacSSAgent(object):
    """
    SAC with an auxiliary self-supervised task.
    Based on https://github.com/denisyarats/pytorch_sac_ae
    """
    def __init__(
        self,
        obs_shape,
        action_shape,
        dynamics_input_shape,
        dynamics_output_shape,
        hidden_dim=256,
        actor_lr=1e-3,
        actor_update_freq=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        ss_lr=1e-3,
        ss_update_freq=1,
        num_layers=4,
        num_shared_layers=4,
        num_filters=32,
    ):
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.ss_update_freq = ss_update_freq

        assert num_layers >= num_shared_layers, 'num shared layers cannot exceed total amount'

        # Actor
        self.actor = Actor(
            obs_shape, action_shape, dynamics_output_shape, hidden_dim,
            encoder_feature_dim,
            num_layers, num_filters, num_layers
        ).cuda()

        # Domain specific part
        self.domain_spe = DomainSpecific(dynamics_input_shape, dynamics_output_shape).cuda()
        # Initialize feat_vect
        self.dynamics_output_shape = dynamics_output_shape
        self.feat_vect = None
        
        # Self-supervision
        self.ss_encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        ).cuda()

        self.ss_encoder.copy_conv_weights_from(self.actor.encoder, num_shared_layers)
        self.inv = InvFunction(encoder_feature_dim, action_shape[0], dynamics_output_shape, hidden_dim).cuda()
        self.inv.apply(weight_init)

        # actor optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )

        # domain specific optimizer
        self.domain_spe_optimizer = torch.optim.Adam(
            self.domain_spe.parameters(), lr=ss_lr
        )
            
        # ss optimizers
        self.init_ss_optimizers(encoder_lr, ss_lr)

        self.train()

    def init_feat_vect(self, init_value):
        self.feat_vect = torch.tensor(init_value).unsqueeze(0).float().cuda()
        self.feat_vect.requires_grad = True
        self.feat_vect_optimizer = torch.optim.Adam(
            [self.feat_vect], lr=1e-3
        )

    def init_ss_optimizers(self, encoder_lr=1e-3, ss_lr=1e-3):
        
        self.encoder_optimizer =  torch.optim.Adam(
                self.ss_encoder.parameters(), lr=encoder_lr
        )
        self.inv_optimizer =  torch.optim.Adam(
                self.inv.parameters(), lr=ss_lr
        )
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.domain_spe.train(training)
        if self.ss_encoder is not None:
            self.ss_encoder.train(training)
        if self.inv is not None:
            self.inv.train(training)

    def select_action(self, obs, mass=None):
        with torch.no_grad():
                if isinstance(obs, np.ndarray):
                    obs = torch.FloatTensor(obs).cuda()
                obs = obs.unsqueeze(0)
                if mass is not None: # If we're in the training phase
                    mass = torch.FloatTensor(mass).cuda()
                    mass = mass.unsqueeze(0)
                    dyn_feat = self.domain_spe(mass)
                else : # If we're at test phase
                    dyn_feat = self.feat_vect
                mu  = self.actor(obs, dyn_feat)
                return mu.cpu().data.numpy().flatten()

    def predict_action(self, obs, next_obs, mass, gt, L=None, step=None):
        """Make the forward pass for actor, domain specific and ss head"""

        # 1. Reset gradients
        self.actor_optimizer.zero_grad()
        self.domain_spe_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()

        # 2 . Do the forward pass
        if obs.dim() < 3 :
            obs = obs.unsqueeze(0)
        # TODO should we move obs to cuda ?

        mass = torch.FloatTensor(mass).cuda()
        mass = mass.repeat(obs.shape[0], 1) # create a batch of masses

        dyn_feat = self.domain_spe(mass) # compute dynamics features

        # Make actor prediction
        mu = self.actor(obs, dyn_feat)

        # Make SS prediction
        h = self.ss_encoder(obs)
        h_next = self.ss_encoder(next_obs)
        pred_action = self.inv(h, h_next, dyn_feat)

        # 3. Compute losses
        actor_loss = F.mse_loss(mu, gt)
        inv_loss = F.mse_loss(pred_action, gt)
        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_inv/inv_loss', inv_loss, step)

        return mu, pred_action, actor_loss + inv_loss


    def update_inv(self, obs, next_obs, action, L=None, step=None):

        assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84
        assert self.feat_vect is not None # Assert we can use this member

        h = self.ss_encoder(obs)
        h_next = self.ss_encoder(next_obs)
        feats = self.feat_vect.repeat(obs.shape[0], 1)
        pred_action = self.inv(h, h_next, feats)

        inv_loss = F.mse_loss(pred_action, action)

        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()
        self.feat_vect_optimizer.zero_grad()

        inv_loss.backward()

        self.encoder_optimizer.step()
        self.inv_optimizer.step()
        self.feat_vect_optimizer.step()

        if L is not None:
            L.log('train_inv/inv_loss', inv_loss, step)

        return inv_loss.item()

    
    def update(self):

        self.actor_optimizer.step()
        self.encoder_optimizer.step()
        self.inv_optimizer.step()
        self.domain_spe_optimizer.step()

            
    def tie_agent_from(self, source):
        """Tie all domain generic part between self and source"""
        # Tie full agents
        assert(isinstance(source, SacSSAgent))

        # Tie SS encoder
        self.ss_encoder.tie_encoder_from(source.ss_encoder)
        # Tie actor
        self.actor.tie_actor_from(source.actor)
        # Tie inv
        self.inv.tie_inv_from(source.inv)

    def extract_feat_vect(self, mass):
        """Extract dynamics feature vector to check if stays constant"""
        mass = torch.as_tensor(mass).float().unsqueeze(0).cuda()
        return self.domain_spe(mass).cpu().data.numpy().flatten()

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
    
        if self.inv is not None:
            torch.save(
                self.inv.state_dict(),
                '%s/inv_%s.pt' % (model_dir, step)
            )
     
        if self.ss_encoder is not None:
            torch.save(
                self.ss_encoder.state_dict(),
                '%s/ss_encoder_%s.pt' % (model_dir, step)
            )
        
        if self.domain_spe is not None:
            torch.save(
                self.domain_spe.state_dict(),
                '%s/domain_specific_%s.pt' %(model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )

        self.inv.load_state_dict(
            torch.load('%s/inv_%s.pt' % (model_dir, step))
        )

        self.ss_encoder.load_state_dict(
            torch.load('%s/ss_encoder_%s.pt' % (model_dir, step))
        )

        self.domain_spe.load_state_dict(
            torch.load('%s/domain_specific_%s.pt' % (model_dir, step))
        )
