import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.encoder import make_encoder

LOG_FREQ = 10000


def make_il_agent(obs_shape, action_shape, args):
    return SacSSAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        #critic_lr=args.critic_lr,
        #critic_beta=args.critic_beta,
        #critic_tau=args.critic_tau,
        #critic_target_update_freq=args.critic_target_update_freq,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        use_rot=args.use_rot,
        use_inv=args.use_inv,
        #use_curl=args.use_curl,
        ss_lr=args.ss_lr,
        ss_update_freq=args.ss_update_freq,
        num_layers=args.num_layers,
        num_shared_layers=args.num_shared_layers,
        num_filters=args.num_filters,
        #curl_latent_dim=args.curl_latent_dim,
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
        self, obs_shape, action_shape, hidden_dim,
        encoder_feature_dim, num_layers, num_filters, num_shared_layers
    ):
        super().__init__()

        self.encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu = self.trunk(obs) 

        return mu 


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class RotFunction(nn.Module):
    """MLP for rotation prediction."""
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, h):
        return self.trunk(h)


class InvFunction(nn.Module):
    """MLP for inverse dynamics model."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2*obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, h, h_next):
        joint_h = torch.cat([h, h_next], dim=1)
        return self.trunk(joint_h)


class SacSSAgent(object):
    """
    SAC with an auxiliary self-supervised task.
    Based on https://github.com/denisyarats/pytorch_sac_ae
    """
    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        #critic_lr=1e-3,
        #critic_beta=0.9,
        #critic_tau=0.005,
        #critic_target_update_freq=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        use_rot=False,
        use_inv=False,
        #use_curl=False,
        ss_lr=1e-3,
        ss_update_freq=1,
        num_layers=4,
        num_shared_layers=4,
        num_filters=32,
        #curl_latent_dim=128,
    ):
        self.discount = discount
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.ss_update_freq = ss_update_freq
        self.use_rot = use_rot
        self.use_inv = use_inv

        assert num_layers >= num_shared_layers, 'num shared layers cannot exceed total amount'

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, num_layers
        ).cuda()


        self.log_alpha = torch.tensor(np.log(init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # self-supervision
        self.rot = None
        self.inv = None
        self.ss_encoder = None

        if use_rot or use_inv:
            self.ss_encoder = make_encoder(
                obs_shape, encoder_feature_dim, num_layers,
                num_filters, num_shared_layers
            ).cuda()
            #self.ss_encoder.copy_conv_weights_from(self.critic.encoder, num_shared_layers)
            self.ss_encoder.copy_conv_weights_from(self.actor.encoder, num_shared_layers)
            
            # rotation
            if use_rot:
                self.rot = RotFunction(encoder_feature_dim, hidden_dim).cuda()
                self.rot.apply(weight_init)

            # inverse dynamics
            if use_inv:
                self.inv = InvFunction(encoder_feature_dim, action_shape[0], hidden_dim).cuda()
                self.inv.apply(weight_init)
            
        # ss optimizers
        self.init_ss_optimizers(encoder_lr, ss_lr)

        # sac optimizers
        self.actor_optimizer = torch.optim.Adam(
            #self.actor.parameters(), lr=actor_lr, weight_decay=1e-3, betas=(actor_beta, 0.999)
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        #self.critic_optimizer = torch.optim.Adam(
        #self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        # ) 

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        #self.critic_target.train()
        self.actor.train()

    def init_ss_optimizers(self, encoder_lr=1e-3, ss_lr=1e-3):
        
        if self.ss_encoder is not None:
            self.encoder_optimizer =  torch.optim.Adam(
                self.ss_encoder.parameters(), lr=encoder_lr
            )
        if self.use_rot:
            self.rot_optimizer =  torch.optim.Adam(
                self.rot.parameters(), lr=ss_lr
            )
        if self.use_inv:
            self.inv_optimizer =  torch.optim.Adam(
                self.inv.parameters(), lr=ss_lr
            )
        # if self.use_curl:
        #     # self.encoder_optimizer =  torch.optim.Adam(
        #     #     self.critic.encoder.parameters(), lr=encoder_lr
        #     # )
        #     self.encoder_optimizer =  torch.optim.Adam(
        #          self.actor.encoder.parameters(), lr=encoder_lr
        #     )
        #     self.curl_optimizer =  torch.optim.Adam(
        #         self.curl.parameters(), lr=ss_lr
        #     )
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        #self.critic.train(training)
        if self.ss_encoder is not None:
            self.ss_encoder.train(training)
        if self.rot is not None:
            self.rot.train(training)
        if self.inv is not None:
            self.inv.train(training)
        # if self.curl is not None:
        #     self.curl.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).cuda()
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).cuda()
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()


    def update_actor(self, pred, gt, step = None, L=None):
        # detach encoder, so we don't update it with the actor loss

        # if bca_loss : # If we're in the test phase and want to adapt

        #     assert buffer is not None, "Need a buffer to use behavioural cloning adaptation"
        #     assert clone is not None, "Need  clone to use behavioural cloning adaptation"

        #     # Sample from source buffer
        #     obses_src, _, _, _, _ = buffer.sample()

        #     # Evaluate clone agent
        #     with utils.eval_mode(clone) and torch.no_grad():
        #         _, pi_target, _, _ = clone.actor(obses_src, compute_log_pi=False)

        #     # Compute KL divergence loss
        #     _, pi, log_pi, _  = self.actor(obses_src, detach_encoder=True)
        #     #kl_loss = nn.KLDivLoss(reduction='batchmean')
        #     #actor_loss = kl_loss(log_pi, pi_target)
        #     mse_loss = nn.MSELoss()
        #     actor_loss = mse_loss(gt, pred)

        # else : # We're in the training phase
        #_, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        mse_loss = nn.MSELoss()
        actor_loss = mse_loss(pred, gt)

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def update_rot(self, obs, L=None, step=None):
        assert obs.shape[-1] == 84

        obs, label = utils.rotate(obs)
        h = self.ss_encoder(obs)
        pred_rotation = self.rot(h)
        rot_loss = F.cross_entropy(pred_rotation, label)

        self.encoder_optimizer.zero_grad()
        self.rot_optimizer.zero_grad()
        rot_loss.backward()

        self.encoder_optimizer.step()
        self.rot_optimizer.step()

        if L is not None:
            L.log('train_rot/rot_loss', rot_loss, step)

        return rot_loss.item()

    def update_inv(self, obs, next_obs, action, L=None, step=None):
        assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84

        h = self.ss_encoder(obs)
        h_next = self.ss_encoder(next_obs)

        pred_action = self.inv(h, h_next)
        inv_loss = F.mse_loss(pred_action, action)

        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()
        inv_loss.backward()

        self.encoder_optimizer.step()
        self.inv_optimizer.step()

        if L is not None:
            L.log('train_inv/inv_loss', inv_loss, step)

        return inv_loss.item()

    # def update_curl(self, obs_anchor, obs_pos, L=None, step=None, ema=False):
    #     assert obs_anchor.shape[-1] == 84 and obs_pos.shape[-1] == 84

    #     z_a = self.curl.encode(obs_anchor)
    #     z_pos = self.curl.encode(obs_pos, ema=True)
        
    #     logits = self.curl.compute_logits(z_a, z_pos)
    #     labels = torch.arange(logits.shape[0]).long().cuda()
    #     curl_loss = F.cross_entropy(logits, labels)
        
    #     self.encoder_optimizer.zero_grad()
    #     self.curl_optimizer.zero_grad()
    #     curl_loss.backward()

    #     self.encoder_optimizer.step()
    #     self.curl_optimizer.step()
    #     if L is not None:
    #         L.log('train/curl_loss', curl_loss, step)

    #     if ema:
    #         # utils.soft_update_params(
    #         #     self.critic.encoder, self.critic_target.encoder,
    #         #     self.encoder_tau
    #         # )
    #         utils.soft_update_params(
    #             self.actor.encoder,
    #             self.encoder_tau
    #         )

    #     return curl_loss.item()

    def update(self, obs, next_obs, L, step, pred, gt):
        # if self.use_curl:
        #     obs, action, reward, next_obs, not_done, curl_kwargs = replay_buffer.sample_curl()
        # else:
        #obs, action, reward, next_obs, not_done = replay_buffer.sample()
        
        L.log('train/batch_reward', reward.mean(), step)

        #self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor(pred, gt)

        # if step % self.critic_target_update_freq == 0:
        #     utils.soft_update_params(
        #         self.critic.Q1, self.critic_target.Q1, self.critic_tau
        #     )
        #     utils.soft_update_params(
        #         self.critic.Q2, self.critic_target.Q2, self.critic_tau
        #     )
        #     utils.soft_update_params(
        #         self.critic.encoder, self.critic_target.encoder,
        #         self.encoder_tau
        #     )
        
        if self.rot is not None and step % self.ss_update_freq == 0:
            self.update_rot(obs, L, step)

        if self.inv is not None and step % self.ss_update_freq == 0:
            self.update_inv(obs, next_obs, action, L, step)

        # if self.curl is not None and step % self.ss_update_freq == 0:
        #     obs_anchor, obs_pos = curl_kwargs["obs_anchor"], curl_kwargs["obs_pos"]
        #     self.update_curl(obs_anchor, obs_pos, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        # torch.save(
        #     self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        # )
        if self.rot is not None:
            torch.save(
                self.rot.state_dict(),
                '%s/rot_%s.pt' % (model_dir, step)
            )
        if self.inv is not None:
            torch.save(
                self.inv.state_dict(),
                '%s/inv_%s.pt' % (model_dir, step)
            )
        # if self.curl is not None:
        #     torch.save(
        #         self.curl.state_dict(),
        #         '%s/curl_%s.pt' % (model_dir, step)
        #     )
        if self.ss_encoder is not None:
            torch.save(
                self.ss_encoder.state_dict(),
                '%s/ss_encoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        # self.critic.load_state_dict(
        #     torch.load('%s/critic_%s.pt' % (model_dir, step))
        # )
        if self.rot is not None:
            self.rot.load_state_dict(
                torch.load('%s/rot_%s.pt' % (model_dir, step))
            )
        if self.inv is not None:
            self.inv.load_state_dict(
                torch.load('%s/inv_%s.pt' % (model_dir, step))
            )
        # if self.curl is not None:
        #     self.curl.load_state_dict(
        #         torch.load('%s/curl_%s.pt' % (model_dir, step))
        #     )
        if self.ss_encoder is not None:
            self.ss_encoder.load_state_dict(
                torch.load('%s/ss_encoder_%s.pt' % (model_dir, step))
            )