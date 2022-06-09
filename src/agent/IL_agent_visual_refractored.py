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

def make_il_agent(obs_shape, action_shape, args, visual=False, dynamics_input_shape=None):
    if not visual :
        assert(dynamics_input_shape is not None), "Needs input shape for domain specific module"
        return ILGTAgent(obs_shape=obs_shape,
        action_shape=action_shape,
        dynamics_input_shape=dynamics_input_shape,
        dynamics_output_shape=args.dynamics_output_shape,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        il_lr=args.il_lr,
        ss_lr=args.ss_lr,
        ss_update_freq=args.ss_update_freq,
        num_layers=args.num_layers,
        num_shared_layers=args.num_shared_layers,
        num_filters=args.num_filters,
    )

    else :
        return ILVisualAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            dynamics_output_shape=args.dynamics_output_shape,
            hidden_dim=args.hidden_dim,
            actor_lr=args.actor_lr,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            ss_lr=args.ss_lr,
            il_lr=args.il_lr,
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
                utils.tie_weights(src=src, trg=tgt)

    def verify_weights_from(self, source):
        """Check if weights are the same between two actors"""
        # Both objects should be actor models
        assert type(self) == type(source)

        is_equal = self.encoder.verify_weights_from(source.encoder)
        print("Encoders in actors are the same : ", is_equal)
        for tgt, src in zip(self.trunk, source.trunk):
            if isinstance(tgt, nn.Linear) and isinstance(src, nn.Linear):
                print("lin layer actor")
                if not utils.verify_weights(src=src, trg=tgt):
                    is_equal = False

        return is_equal

class DomainSpecific(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

class DomainSpecificGT(DomainSpecific):
    """MLP specific domain network."""

    def __init__(self, dynamics_input_shape, dynamics_output_shape, hidden_dim=20):
        super().__init__()

        self.specific = nn.Sequential(nn.Linear(dynamics_input_shape, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, dynamics_output_shape))

    def forward(self, gt):
        res = self.specific(gt)
        return res

    def verify_weights(self, source):
        # Both objects should be actor models
        assert type(self) == type(source)

        is_equal = True
        for tgt, src in zip(self.specific, source.specific):
            if isinstance(tgt, nn.Linear) and isinstance(src, nn.Linear):
                print("lin layer domain spe")
                if not utils.verify_weights(src=src, trg=tgt):
                    is_equal = False

        return is_equal

class DomainSpecificVisual(DomainSpecific):
    """MLP specific domain network."""

    def __init__(self, obs_shape, action_shape, encoder_feature_dim,
                 num_layers, num_filters, num_shared_layers,
                 dynamics_output_shape, hidden_dim=20):

        super().__init__()

        self.encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )
        self.num_shared = num_shared_layers

        input_feature_dim = 3 * encoder_feature_dim + 2 * action_shape[0]

        self.specific = nn.Sequential(nn.Linear(input_feature_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, dynamics_output_shape))

    def forward(self, obs1, act1, obs2, act2, obs3):
        obs1 = self.encoder(obs1)
        obs2 = self.encoder(obs2)
        obs3 = self.encoder(obs3)
        joint_input = torch.cat([obs1, act1, obs2, act2, obs3], dim=1)
        res = self.specific(joint_input)
        return res

    def verify_weights_from(self, source):

        """Check if weights are the same between two domain specific"""

        # Both objects should be actor models
        assert type(self) == type(source)

        is_diff = not self.encoder.verify_weights_from(source.encoder)
        print("Encoders in domain are not the same : ", is_diff)
        for tgt, src in zip(self.specific, source.specific):
            if isinstance(tgt, nn.Linear) and isinstance(src, nn.Linear):
                if utils.verify_weights(src=src, trg=tgt):
                    is_diff = False

        return is_diff


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
                utils.tie_weights(src=src, trg=tgt)

    def verify_weights_from(self, source):
        """Check if weights are the same between two inv"""
        is_equal = True
        assert type(self) == type(source)
        # Copy linear layers
        for tgt, src in zip(self.trunk, source.trunk):
            if isinstance(tgt, nn.Linear) and isinstance(src, nn.Linear):
                print("lin layer inv")
                if not utils.verify_weights(src=src, trg=tgt):
                    is_equal = False
        return is_equal


class ILSSAgent(object):
    """
    IL agent classifier with an auxiliary self-supervised task.
    """
    def __init__(
            self,
            obs_shape,
            action_shape,
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
        self.num_shared_layers = num_shared_layers

        # Actor
        self.actor = Actor(
            obs_shape, action_shape, dynamics_output_shape, hidden_dim,
            encoder_feature_dim,
            num_layers, num_filters, num_layers
        ).cuda()

        # Abstract domain specific
        self.domain_spe = DomainSpecific()

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

        # ss optimizers
        self.init_ss_optimizers(encoder_lr, ss_lr)

        self.train()

    def init_feat_vect(self, init_value, batch_size):
        self.feat_vect = torch.tensor(init_value).unsqueeze(0).float().cuda()
        self.feat_vect.requires_grad = True
        self.feat_vect_optimizer = torch.optim.Adam(
            [self.feat_vect], lr=self.il_lr
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

    def select_action(self, obs):
        raise NotImplementedError

    def predict_action(self, obs, next_obs, dynamics):
        raise NotImplementedError

    def update_inv(self):
        raise NotImplementedError

    def update(self):

        raise NotImplementedError

    def tie_agent_from(self, source):
        """Tie all domain generic part between self and source"""
        # Tie domain generic part of agents
        assert (isinstance(source, ILSSAgent))

        # Tie SS encoder
        self.ss_encoder.tie_encoder_from(source.ss_encoder)
        # Tie actor
        self.actor.tie_actor_from(source.actor)
        # Tie inv
        self.inv.tie_inv_from(source.inv)

    def verify_weights_from(self, source):

        assert (isinstance(source, ILSSAgent))

        print("shared Encoders are the same :", self.ss_encoder.verify_weights_from(source.ss_encoder))
        print("Actors are the same : ", self.actor.verify_weights_from(source.actor))
        print("Inv are the same: ", self.inv.verify_weights_from(source.inv))
        print("Actors and shared encoders share the same layers : ", self.actor.encoder.verify_weights_from(self.ss_encoder, num=self.num_shared_layers))


    def extract_feat_vect(self, dyn):
        raise NotImplementedError

    def save_full(self, model_dir, step):
        """Save full model behaves better"""
        path = '%s/full_model%s.pt' % (model_dir, step)
        torch.save(self, path)

    def load_full(self, model_dir, step):
        path = '%s/full_model%s.pt' % (model_dir, step)
        model = torch.load(path)
        return model


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
                '%s/domain_specific_%s.pt' % (model_dir, step)
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

class ILVisualAgent(ILSSAgent):

    def __init__(
            self,
            obs_shape,
            action_shape,
            dynamics_output_shape,
            hidden_dim=256,
            actor_lr=1e-3,
            actor_update_freq=2,
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            il_lr=1e-3,
            ss_lr=1e-3,
            ss_update_freq=1,
            num_layers=4,
            num_shared_layers=4,
            num_filters=32,
    ):

        super().__init__(obs_shape,
            action_shape,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            actor_update_freq=actor_update_freq,
            encoder_feature_dim=encoder_feature_dim,
            encoder_lr=encoder_lr,
            encoder_tau=encoder_tau,
            ss_lr=ss_lr,
            ss_update_freq=ss_update_freq,
            num_layers=num_layers,
            num_shared_layers=num_shared_layers,
            num_filters=num_filters)

        # Domain specific part
        self.domain_spe = DomainSpecificVisual(obs_shape, action_shape, encoder_feature_dim,
                                               num_layers, num_filters, num_shared_layers, dynamics_output_shape).cuda()
        # self.domain_spe.encoder.copy_conv_weights_from(self.actor.encoder, num_shared_layers)

        # Domain specific optimizer
        self.domain_spe_optimizer = torch.optim.Adam(
            self.domain_spe.parameters(), lr=il_lr
        )


    def select_action(self, obs, traj):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).cuda()
            obs = obs.unsqueeze(0)

            dyn_feat = self.domain_spe(*traj) # *traj is equivalent to obs1, act1, obs2, act2, obs3
            mu = self.actor(obs, dyn_feat)
            return mu.cpu().data.numpy().flatten()


    def predict_action(self, obs, next_obs, traj, gt, L=None, step=None):
        """Make the forward pass for actor, domain specific and ss head"""

        # 1. Reset gradients
        self.actor_optimizer.zero_grad()
        self.domain_spe_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()

        # 2 . Do the forward pass
        if obs.dim() < 3:
            obs = obs.unsqueeze(0)

        dyn_feat = self.domain_spe(*traj)  # compute dynamics features

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


    def update_inv(self, obs, next_obs, action, traj):

        assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84

        h = self.ss_encoder(obs)
        h_next = self.ss_encoder(next_obs)
        dyn_feat = self.domain_spe(*traj)
        pred_action = self.inv(h, h_next, dyn_feat)

        inv_loss = F.mse_loss(pred_action, action)

        self.domain_spe_optimizer.zero_grad()
        inv_loss.backward()
        self.domain_spe_optimizer.step()

    def update(self):

        self.actor_optimizer.step()
        self.encoder_optimizer.step()
        self.inv_optimizer.step()
        self.domain_spe_optimizer.step()


    def extract_feat_vect(self, traj):
        """Extract dynamics feature vector"""
        return self.domain_spe(*traj).cpu().data.numpy().flatten()

    def verify_weights_from(self, source):

        super().verify_weights_from()
        print("Domain spe and actors shared encoders are the same : ", self.actor.encoder.verify_weights_from(
            self.domain_spe.encoder, num=self.num_shared_layers))
        print("Domain spe are not the same", self.domain_spe.verify_weights_from(source.domain_spe))


class ILGTAgent(ILSSAgent):

    def __init__(self,
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
                 il_lr=1e-3,
                 ss_lr=1e-3,
                 ss_update_freq=1,
                 num_layers=4,
                 num_shared_layers=4,
                 num_filters=32,
                 ):
        super().__init__(obs_shape,
                 action_shape,
                 dynamics_output_shape,
                 hidden_dim=hidden_dim,
                 actor_lr=actor_lr,
                 actor_update_freq=actor_update_freq,
                 encoder_feature_dim=encoder_feature_dim,
                 encoder_lr=encoder_lr,
                 encoder_tau=encoder_tau,
                 ss_lr=ss_lr,
                 ss_update_freq=ss_update_freq,
                 num_layers=num_layers,
                 num_shared_layers=num_shared_layers,
                 num_filters=num_filters)

        self.il_lr = il_lr

        # Domain specific part
        self.domain_spe = DomainSpecificGT(dynamics_input_shape, dynamics_output_shape).cuda()
        # Initialize feat_vect
        self.dynamics_output_shape = dynamics_output_shape
        self.feat_vect = None

        # domain specific optimizer
        self.domain_spe_optimizer = torch.optim.Adam(
            self.domain_spe.parameters(), lr=ss_lr
        )

    def init_feat_vect(self, init_value):
        self.feat_vect = torch.tensor(init_value).unsqueeze(0).float().cuda()
        self.feat_vect.requires_grad = True
        self.feat_vect_optimizer = torch.optim.Adam(
            [self.feat_vect], lr=self.il_lr
        )

    def select_action(self, obs, mass=None, force = None):
        with torch.no_grad():
                if isinstance(obs, np.ndarray):
                    obs = torch.FloatTensor(obs).cuda()
                obs = obs.unsqueeze(0)
                if mass is not None: # If we're in the training phase
                    mass = torch.FloatTensor(mass).cuda()
                    mass = mass.unsqueeze(0)
                    dyn_feat = self.domain_spe(mass)
                elif force is not None:
                    force = torch.FloatTensor(force).cuda()
                    force = force.unsqueeze(0)
                    dyn_feat = self.domain_spe(force)
                else: # If we're at test phase
                    dyn_feat = self.feat_vect
                mu  = self.actor(obs, dyn_feat)
                return mu.cpu().data.numpy().flatten()

    def predict_action(self, obs, next_obs, gt, force=None, mass=None):
        """Make the forward pass for actor, domain specific and ss head"""

        # 1. Reset gradients
        self.actor_optimizer.zero_grad()
        self.domain_spe_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()

        # 2 . Do the forward pass
        if obs.dim() < 3:
            obs = obs.unsqueeze(0)

        if mass is not None:
            mass = torch.FloatTensor(mass).cuda()
            mass = mass.repeat(obs.shape[0], 1)  # create a batch of masses

            dyn_feat = self.domain_spe(mass)  # compute dynamics features

        elif force is not None:
            force = torch.FloatTensor(force).cuda()
            force = force.repeat(obs.shape[0], 1)  # create a batch of forces

            dyn_feat = self.domain_spe(force)  # compute dynamics features

        else :
            raise NotImplementedError

        # Make actor prediction
        mu = self.actor(obs, dyn_feat)

        # Make SS prediction
        h = self.ss_encoder(obs)
        h_next = self.ss_encoder(next_obs)
        pred_action = self.inv(h, h_next, dyn_feat)

        # 3. Compute losses
        actor_loss = F.mse_loss(mu, gt)
        inv_loss = F.mse_loss(pred_action, gt)

        return mu, pred_action, actor_loss + inv_loss

    def update_inv(self, obs, next_obs, action):

        assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84
        assert self.feat_vect is not None  # Assert we can use this member

        h = self.ss_encoder(obs)
        h_next = self.ss_encoder(next_obs)

        inv_loss = 0
        for o, n_o, act in zip(h, h_next, action):
            o = o.unsqueeze(0)
            n_o = n_o.unsqueeze(0)
            act = act.unsqueeze(0)
            pred_action = self.inv(o, n_o, self.feat_vect)
            inv_loss += F.mse_loss(pred_action, act)

        self.feat_vect_optimizer.zero_grad()
        inv_loss.backward()
        self.feat_vect_optimizer.step()

        return inv_loss.item()

    def update(self):

        self.actor_optimizer.step()
        self.encoder_optimizer.step()
        self.inv_optimizer.step()
        self.domain_spe_optimizer.step()

    def verify_weights_from(self, source):
        super().verify_weights_from()
        print("Domain specifics should not be the same", not self.domain_spe.verify_weights(source.domain_spe))

    def extract_feat_vect(self, mass= None, force = None):
        """Extract dynamics feature vector to check if stays constant"""
        if mass is not None:
            mass = torch.as_tensor(mass).float().unsqueeze(0).cuda()
            return self.domain_spe(mass).cpu().data.numpy().flatten()
        else:
            force = torch.as_tensor(force).float().unsqueeze(0).cuda()
            return self.domain_spe(force).cpu().data.numpy().flatten()




