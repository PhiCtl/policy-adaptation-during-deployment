import pandas as pd
import torch
import numpy as np
from scipy.ndimage import convolve1d
import cv2
import os
from datetime import datetime
import random

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def verify_weights(src, trg):
    return torch.equal(trg.weight,src.weight) and torch.equal(trg.bias, src.bias)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def moving_average_reward(rewards, current_ep=None, wind_lgth=15):
    # Causal convolutional filter
    w = np.concatenate((np.zeros(wind_lgth + 1), np.ones(wind_lgth))).astype(np.float64) / (wind_lgth)
    avg = convolve1d(rewards, w, mode='nearest')
    if current_ep is None:
        return avg
    else:
        assert current_ep >= 0
        return avg[current_ep]

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

class SimpleBuffer(object):

    def __init__(self, obs_shape, action_shape, capacity, batch_size, label=None):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)


        # Save label, ie. domain specificity
        self.label = label
        self.idx = 0
        self.full = False

    def add(self, obs, action, next_obs):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.next_obses[self.idx], next_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_path(self, obses, actions):
        # TODO ugly...
        obs1 = obses[:-1]
        act1 = actions[:]
        obs2 = obses[1:]

        for obs, action, next_obs in zip(obs1, act1, obs2):
            self.add(obs, action, next_obs)

    def sample(self, idxs=None):

        if idxs is None :
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )

        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).float().cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)

        return obses, actions, next_obses


class TrajectoryBuffer(SimpleBuffer):
    """Stores data from an environment in 5 arrays
       we want at some point to retrieve a successive sequence of observations and actions
       that's why observations and actions are stored this way
    """

    def __init__(self, obs_shape, action_shape, capacity, batch_size, label=None):
        self.capacity = capacity
        self.batch_size = batch_size

        super().__init__(obs_shape, action_shape, capacity, batch_size, label=label)

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.actions_1 = np.empty((capacity, *action_shape), dtype=np.float32)
        self.obses_2 = np.empty((capacity, *obs_shape), dtype=obs_dtype)


    def add(self, obs_0, action_0, obs_1, action_1, obs_2):

        np.copyto(self.actions_1[self.idx], action_1)
        np.copyto(self.obses_2[self.idx], obs_2)

        super().add(obs_0, action_0, obs_1)  # Take care of increments

    def add_path(self, obses, actions):
        obs0s = obses[:-2]
        act0s = actions[:-1]
        obs1s = obses[1:-1]
        act1s = actions[1:]
        obs2s = obses[2:]
        for obs_0, action_0, obs_1, action_1, obs_2 in zip(obs0s, act0s, obs1s, act1s, obs2s):
            self.add(obs_0, action_0, obs_1, action_1, obs_2)

    def sample(self, idxs=None):

        if idxs is None :
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )
        obses_0, actions_0, obses_1 = super().sample(idxs = idxs)
        actions_1 = torch.as_tensor(self.actions_1[idxs]).float().cuda()
        obses_2 = torch.as_tensor(self.obses_2[idxs]).float().cuda()

        obses_2 = random_crop(obses_2)

        return [obses_0, actions_0, obses_1, actions_1, obses_2]

    def sample_traj(self):
        """Sample single trajectory"""

        ix = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        obs_0, action_0, obs_1 = super().sample(idxs=ix)
        action_1 = torch.as_tensor(self.actions_1[ix]).float().cuda()
        obs_2 = torch.as_tensor(self.obses_2[ix]).float().cuda()

        obs_2 = random_crop(obs_2)

        return [obs_0, action_0, obs_1, action_1, obs_2]

class ExtendedTrajectoryBuffer(TrajectoryBuffer):
    """Stores data from an environment in 7 arrays
           we want at some point to retrieve a successive sequence of observations and actions
           that's why observations and actions are stored this way
        """

    def __init__(self, obs_shape, action_shape, capacity, batch_size, label=None):
        self.capacity = capacity
        self.batch_size = batch_size

        super().__init__(obs_shape, action_shape, capacity, batch_size, label=label)

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8


        self.actions_2 = np.empty((capacity, *action_shape), dtype=np.float32)
        self.obses_3 = np.empty((capacity, *obs_shape), dtype=obs_dtype)

    def add(self, obs_0, action_0, obs_1, action_1, obs_2, action_2, obs_3):

        np.copyto(self.actions_2[self.idx], action_2)
        np.copyto(self.obses_3[self.idx], obs_3)
        super().add(obs_0, action_0, obs_1, action_1, obs_2)  # Take care of increments

    def add_path(self, obses, actions):
        obs0s = obses[:-3]
        act0s = actions[:-2]
        obs1s = obses[1:-2]
        act1s = actions[1:-1]
        obs2s = obses[2:-1]
        act2s = actions[2:]
        obs3s = obses[3:]
        for obs_0, action_0, obs_1, action_1, obs_2, action_2, obs_3 in zip(obs0s, act0s, obs1s, act1s, obs2s, act2s,
                                                                            obs3s):
            self.add(obs_0, action_0, obs_1, action_1, obs_2, action_2, obs_3)

    def sample(self, idxs=None):

        if idxs is None :
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )

        obses_0, actions_0, obses_1, actions_1, obses_2 = super().sample(idxs = idxs)
        actions_2 = torch.as_tensor(self.actions_2[idxs]).float().cuda()
        obses_3 = torch.as_tensor(self.obses_3[idxs]).float().cuda()

        obses_3 = random_crop(obses_3)

        return obses_0, actions_0, obses_1, [obses_0, actions_0, obses_1, actions_1, obses_2, actions_2, obses_3 ]

    def sample_traj(self):
        """Sample single trajectory"""

        ix = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        obs_0, action_0, obs_1, action_1, obs_2 = super().sample(idxs=ix)
        action_2 = torch.as_tensor(self.actions_2[ix]).float().cuda()
        obs_3 = torch.as_tensor(self.obses_3[ix]).float().cuda()

        obs_3 = random_crop(obs_3)

        return [obs_0, action_0, obs_1, action_1, obs_2, action_2, obs_3]

class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, label=None):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()
        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)

        return obses, actions, rewards, next_obses, not_dones


    def sample_curl(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        pos = obses.clone()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)
        pos = random_crop(pos)

        curl_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                           time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, curl_kwargs



def get_curl_pos_neg(obs, replay_buffer):
    """Returns one positive pair + batch of negative samples from buffer"""
    obs = torch.as_tensor(obs).cuda().float().unsqueeze(0)
    pos = obs.clone()

    obs = random_crop(obs)
    pos = random_crop(pos)

    # Sample negatives and insert positive sample
    obs_pos = replay_buffer.sample_curl()[-1]['obs_pos']
    obs_pos[0] = pos

    return obs, obs_pos


def batch_from_obs(obs, batch_size=32):
    """Converts a pixel obs (C,H,W) to a batch (B,C,H,W) of given size"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)


def _rotate_single_with_label(x, label):
    """Rotate an image"""
    if label == 1:
        return x.flip(2).transpose(1, 2)
    elif label == 2:
        return x.flip(2).flip(1)
    elif label == 3:
        return x.transpose(1, 2).flip(2)
    return x


def rotate(x):
    """Randomly rotate a batch of images and return labels"""
    images = []
    labels = torch.randint(4, (x.size(0),), dtype=torch.long).to(x.device)
    for img, label in zip(x, labels):
        img = _rotate_single_with_label(img, label)
        images.append(img.unsqueeze(0))

    return torch.cat(images), labels


def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop"""
    assert isinstance(x, torch.Tensor) and x.is_cuda, \
        'input must be CUDA tensor'

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
        'window_shape must be a tuple with same number of dimensions as x'

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3)
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def random_crop(imgs, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
        'must either specify both w1 and h1 or neither of them'

    is_tensor = isinstance(imgs, torch.Tensor)
    if is_tensor:
        assert imgs.is_cuda, 'input images are tensors but not cuda!'
        return random_crop_cuda(imgs, size=size, w1=w1, h1=h1, return_w1_h1=return_w1_h1)

    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return imgs, None, None
        return imgs

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    if w1 is None:
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)

    windows = view_as_windows(imgs, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[np.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped
