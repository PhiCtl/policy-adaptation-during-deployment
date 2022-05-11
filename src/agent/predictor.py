import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import moving_average_reward

class Predictor(nn.Module) :

    def __init__(self):
        self.step = None

    def forward(self):
        raise NotImplementedError

    def sample_dynamics(self):
        raise NotImplementedError

    def init_step(self, step):
        self.step = step


class HardcodedPredictor(Predictor):

    """
    HardCoded Predictor module which outputs the true current and future values
    Of the given dynamics
    Dynamics it can handle are allowed to change in a time-dependent way only
    """

    def __init__(self, dynamics_shape, init_dynamics):
        super().__init__()
        self.dynamics = init_dynamics
        self.forecasting_steps = dynamics_shape

    def forward(self):
        return self.dynamics.sample_window(self.step, self.forecasting_steps)

    def sample_dynamics(self):
        return self.dynamics.init_value()

def build_predictor(predictor, args) :

    if predictor == 'cart_mass' and args.domain_name == 'cartpole':
        return HardcodedPredictor( args.latent_shape,
                                  CartMass(args.window))
    else:
        raise NotImplementedError(f'{predictor} for {args.domain_name} is not handled yet')

class Dynamics(object) :

    def __init__(self):
        pass

    def init_value(self):
        raise NotImplementedError

    def _create_values(self):
        raise NotImplementedError

    def sample_window(self):
        raise NotImplementedError

class CartMass(Dynamics) :

    def __init__(self, time_change, allowed_values=[1,0.1], step = -0.1):
        super().__init__()
        self.time_change = time_change
        self.step = step
        self.values = None
        self._create_values()

    def _create_values(self, allowed_values):
        # TODO implement
        pass

    def init_value(self):
        if self.values :
            return self.values[0]

    def sample_window(self, start_point, length):
        # TODO implement
        pass


