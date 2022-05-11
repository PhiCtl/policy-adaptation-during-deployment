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
        self.forecasting_steps = dynamics_shape
        self.dynamics = init_dynamics

    def forward(self):
        return self.dynamics.sample_window(self.step, self.forecasting_steps)

    def sample_dynamics(self):
        return self.dynamics.init_value()

def build_predictor(predictor, args) :

    if predictor == 'cart_mass' and args.domain_name == 'cartpole':
        return HardcodedPredictor(args.dynamics_shape,
                                  CartMass(args.window))
    else:
        raise NotImplementedError(f'{predictor} for {args.domain_name} is not handled yet')

class Dynamics(object) :

    def __init__(self):
        pass

    def init_value(self):
        raise NotImplementedError

    def _create_values(self, allowed_values):
        raise NotImplementedError

    def sample_window(self, start_point, length):
        raise NotImplementedError

class CartMass(Dynamics) :

    def __init__(self, time_of_invariance, allowed_values=[0.1, 1], step=-0.1):
        super().__init__()
        assert allowed_values[0] < allowed_values[1], "Allowed values should be of the type [min_value, max_value]"
        self.time_of_invariance = time_of_invariance
        self.step = step
        self.values = None
        self._create_values(allowed_values)

    def _create_values(self, allowed_values):
        if self.step < 0:
            end, start = allowed_values
        else :
            start, end = allowed_values
        values, current = [], start

        while current*np.sign(self.step) >= end*np.sign(self.step):
            values.extend([current]*self.time_of_invariance)
            current += self.step

        self.values = np.array(values)

    def init_value(self):
        if self.values :
            return self.values[0]

    def sample_window(self, start_point, length):
        tot_length = len(self.values)
        i = start_point %  tot_length
        if i + length >= tot_length:
            res = np.concatenate([self.values[i:], self.values[:tot_length-length]])
            return res
        return self.values[i: i+length]


