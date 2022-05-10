import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import moving_average_reward

class Predictor(nn.Module) :

    def __init__(self):
        self.buffer = None
        self.step = 0

    def forward(self):
        raise NotImplementedError

    def add_buffer(self, experience_buffer):
        self.buffer = experience_buffer

    def sample_latent(self):
        raise NotImplementedError

    def set_step(self, step):
        self.step = step


class HardcodedPredictor(Predictor):

    """
    HardCoded Predictor module which outputs the true current and future values
    Of the given dynamics
    Dynamics it can handle are allowed to change in a time-dependent way only
    """

    def __init__(self, window, latent_shape, init_dynamics):
        super().__init__()
        self.window = window
        self.dynamics = init_dynamics
        self.forecasting_steps = latent_shape

    def forward(self):
        return self.dynamics.future(self.forecasting_steps)

    def update_dynamics(self, step):
        avg_reward = moving_average_reward(self.buffer.rewards, wind_lgth=self.avg_wind)
        if avg_reward > self.threshold and step % self.avg_wind == 0 and step > 1:
            self.dynamics.update()

    def sample_latent(self):
        return self.dynamics.init_value()

def build_predictor(predictor, args) :

    if predictor == 'cart_mass':
        return HardcodedPredictor(args.window,
                                  args.latent_shape,
                                  CartMass())
    else:
        raise NotImplementedError(f'{predictor} is not handled yet')

class Dynamics(object) :

    def __init__(self):
        pass

    def set_time_param(self, avg_wind):
        pass

    def init_value(self):
        raise NotImplementedError

    def update(self):
        pass

    def value(self):
        raise NotImplementedError

class CartMass(Dynamics) :

    def __init__(self, init_value = 1, step = -0.1):
        super().__init__()
        self.init_value = init_value
        self.step = step
        self.mass = init_value

    def update(self):
        self.mass += self.step
        if self.mass <= 0 :
            self.mass = self.init_value

    # value
    def value(self):
        return self.mass.copy()

    def init_value(self):
        return self.init_value


