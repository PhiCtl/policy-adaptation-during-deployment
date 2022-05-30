import numpy as np
import pandas as pd
from datetime import datetime
import os

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

    def __init__(self, save_dir, type, action=False):
        super().__init__(save_dir, type)
        self.changes_tot, self.changes = [], []
        self.rewards_tot, self.rewards = [], []
        self.actions_tot, self.actions = None, None
        if action:
            self.actions_tot, self.actions = [], []

    def reset(self):
        self.changes, self.rewards, self.actions = [], [], []

    def update(self, change, reward):
        self.changes.append(change)
        self.rewards.append(reward)

    def end_episode(self):
        self.changes_tot.append(self.changes)
        self.rewards_tot.append(self.rewards)
        if self.actions_tot:
            self.actions_tot.append(self.actions)

        self.reset()

    def save(self, file_name, adapt):
        self.rewards_tot = np.array(self.rewards_tot).transpose()
        self.changes_tot = np.array(self.changes_tot).transpose()
        df_r = pd.DataFrame(self.rewards_tot, columns=[f'episode_{i}_reward' for i in range(self.rewards_tot.shape[1])])
        df_s = pd.DataFrame(self.changes_tot,
                            columns=[f'episode_{i}_{self._type}' for i in range(self.changes_tot.shape[1])])

        if self.actions_tot:
            self.actions_tot = np.array(self.actions_tot).transpose()
            df_a = pd.DataFrame(self.actions_tot,
                                columns=[f'episode_{i}_action' for i in range(self.actions_tot.shape[1])])
            df_r = df_r.join(df_a)
        df_tot = df_r.join(df_s)

        # Rename file and folders
        file_name += datetime.now().strftime("%H-%M-%S")
        file_name += self._type
        file_name += "_pad.csv" if adapt else "_eval.csv"
        df_tot.to_csv(os.path.join(self._save_dir, file_name))
        self.changes_tot, self.rewards_tot, self.actions_tot = [], [], []