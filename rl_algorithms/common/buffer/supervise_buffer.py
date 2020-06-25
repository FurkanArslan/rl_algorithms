import os
import pickle

from PIL import Image
import numpy as np

from rl_algorithms.common.abstract.buffer import BaseBuffer


class SuperviseBuffer(BaseBuffer):
    def __init__(
        self, experiences_dir: str, batch_size: int = 32,
    ):
        self.batch_size = batch_size
        self.state_dir = experiences_dir + "/state/"
        self.next_state_dir = experiences_dir + "/next_state/"
        self.ard_list = pickle.load(open(experiences_dir + "/ard.pkl", "rb"))
        self.ard_list = np.asarray(self.ard_list)
        self.framestack_num = len(os.listdir(self.state_dir)) // len(self.ard_list)

    def add(self):
        pass

    def sample(self):
        indices = np.random.choice(len(self), size=self.batch_size, replace=False)
        batch_state = None
        batch_next_state = None
        for i, index in enumerate(indices):
            tmp_state = []
            tmp_next_state = []
            for j in range(self.framestack_num):
                s = Image.open(self.state_dir + "%d-%d.png" % (index, j))
                tmp_state.append(np.asarray(s))

                n_s = Image.open(self.next_state_dir + "%d-%d.png" % (index, j))
                tmp_next_state.append(np.asarray(n_s))
            tmp_state = np.stack(tmp_state, 0)
            tmp_next_state = np.stack(tmp_next_state, 0)
            if i == 0:
                batch_state = np.zeros((self.batch_size, *tmp_state.shape))
                batch_next_state = np.zeros((self.batch_size, *tmp_next_state.shape))
            batch_state[i] = tmp_state
            batch_next_state[i] = tmp_next_state
        batch_ard = self.ard_list[indices]
        return batch_state, batch_next_state, batch_ard

    def __len__(self) -> int:
        return len(self.ard_list)


if __name__ == "__main__":
    test = SuperviseBuffer("data/experience/pong/200625_102934", 32)
    test.sample()
    print()
