import pandas as pd
import torch
import numpy as np
from scipy.ndimage import convolve1d
import cv2
import os
from datetime import datetime
import random


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


class Recorder(object):

    def __init__(self, save_dir, type):
        self._save_dir = save_dir
        self._type = type

    def reset(self):
        raise NotImplementedError

    def update(self, change, reward):
        raise NotImplementedError

    def end_episode(self):
        self.reset()

    def save(self, file_name, adapt):
        raise NotImplementedError


class AdaptRecorder(Recorder):

    def __init__(self, save_dir, type):
        super().__init__(save_dir, type)
        self.changes_tot, self.changes = [], []
        self.rewards_tot, self.rewards = [], []

    def reset(self):
        self.changes, self.rewards = [], []

    def update(self, change, reward):
        self.changes.append(change)
        self.rewards.append(reward)

    def end_episode(self):
        self.changes_tot.append(self.changes)
        self.rewards_tot.append(self.rewards)

        self.reset()

    def save(self, file_name, adapt):
        self.rewards_tot = np.array(self.rewards_tot).transpose()
        self.changes_tot = np.array(self.changes_tot).transpose()
        df_r = pd.DataFrame(self.rewards_tot, columns=[f'episode_{i}_reward' for i in range(self.rewards_tot.shape[1])])
        df_s = pd.DataFrame(self.changes_tot,
                            columns=[f'episode_{i}_{self._type}' for i in range(self.changes_tot.shape[1])])
        df_tot = df_r.join(df_s)
        # Rename file and folders
        file_name += datetime.now().strftime("%H-%M-%S")
        file_name += self._type
        file_name += "_pad.csv" if adapt else "_eval.csv"
        df_tot.to_csv(os.path.join(self._save_dir, file_name))
        self.changes_tot, self.rewards_tot = [], []


class EnvtRecorder(Recorder):

    def __init__(self, save_dir, type):
        super().__init__(save_dir, type)
        self.rewards_cumul, self.reward = [], 0
        self.df = []
        self.params, self.bg_name = None, None

    def reset(self):
        self.reward = 0

    def update(self, change, reward):
        self.reward += reward

    def end_episode(self):
        self.rewards_cumul.append(self.reward)
        self.reset()

    def save(self, file_name, adapt):
        self.df.append({"background": self.bg_name,
                        "params": list(self.params.values()),
                        "mean cumulative": np.mean(self.rewards_cumul),
                        "std cumulative": np.std(self.rewards_cumul)})
        self.rewards_cumul = []

    def load_background(self, bg):
        self.bg_name = bg

    def load_change(self, params):
        self.params = params

    def close(self):
        self.df = pd.DataFrame(self.df)
        file_name = datetime.now().strftime("%H-%M-%S") + "_change_"
        file_name += self._type
        file_name += "_eval.csv"
        self.df.to_csv(os.path.join(self._save_dir, file_name))


def moving_average_reward(rewards, current_ep=None, wind_lgth=15):
    # Causal convolutional filter
    w = np.concatenate((np.zeros(wind_lgth + 1), np.ones(wind_lgth))).astype(np.float64) / (wind_lgth)
    avg = convolve1d(rewards, w, mode='nearest')
    if current_ep is None:
        return avg
    else:
        assert current_ep >= 0
        return avg[current_ep]


def compute_distance(img1, img2):
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[20:40, 20:40, :]
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[20:40, 20:40, :]

    x1 = np.cos(img1[:, :, 0] * np.pi / 180) * img1[:, :, 1]
    y1 = np.sin(img1[:, :, 0] * np.pi / 180) * img1[:, :, 1]
    z1 = img1[:, :, 2]

    x2 = np.cos(img2[:, :, 0] * np.pi / 180) * img2[:, :, 1]
    y2 = np.sin(img2[:, :, 0] * np.pi / 180) * img2[:, :, 1]
    z2 = img2[:, :, 2]

    return np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2).sum())


def compute_speed(avg_reward, max_speed, coef=0.1, max_reward=8):
    return (max_speed * np.exp(coef * (avg_reward - max_reward))).astype(int)


def wrap_speed(speed, max):
    sp = speed % max
    if sp > max / 2:
        sp -= max
    return sp


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


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

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
