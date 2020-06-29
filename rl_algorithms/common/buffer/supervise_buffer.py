import os
import pickle

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rl_algorithms.common.helper_functions import numpy2floattensor


def image_post_process(batch_state, batch_next_state, batch_ard):
    n, n_s = numpy2floattensor((batch_state, batch_next_state))
    a, r, d = numpy2floattensor(
        (np.array([batch_ard[0]]), np.array([batch_ard[1]]), np.array([batch_ard[2]]),)
    )
    a = a.reshape(-1)
    r = r.reshape(-1, 1)
    d = d.reshape(-1, 1)
    return n, a, r, n_s, d


def vector_post_process(experience):
    n, n_s = numpy2floattensor((experience[0], experience[3]))
    a, r, d = numpy2floattensor(
        (
            np.array([experience[1]]),
            np.array([experience[2]]),
            np.array([np.uint8(experience[4])]),
        )
    )
    return n, a, r, n_s, d


class SuperviseBuffer(torch.utils.data.Dataset):
    def __init__(self, experiences_dir: str, from_disk: bool = True):
        self.from_disk = from_disk
        if os.path.isdir(experiences_dir + "/state/"):
            self.is_image = True
            self.state_dir = experiences_dir + "/state/"
            self.next_state_dir = experiences_dir + "/next_state/"
            self.ard_list = pickle.load(open(experiences_dir + "/ard.pkl", "rb"))
            self.ard_list = np.asarray(self.ard_list)
            self.framestack_num = len(os.listdir(self.state_dir)) // len(self.ard_list)
        else:
            self.is_image = False
            self.experience_list = pickle.load(
                open(experiences_dir + "/experience.pkl", "rb")
            )

        if not from_disk:
            if self.is_image:
                self.state_list = None
                self.next_state_list = None
                for i in tqdm(range(len(self.ard_list))):
                    tmp_state = []
                    tmp_next_state = []
                    for j in range(self.framestack_num):
                        s = Image.open(self.state_dir + "%d-%d.png" % (i, j))
                        tmp_state.append(np.asarray(s))

                        n_s = Image.open(self.next_state_dir + "%d-%d.png" % (i, j))
                        tmp_next_state.append(np.asarray(n_s))
                    tmp_state = np.stack(tmp_state, 0)
                    tmp_next_state = np.stack(tmp_next_state, 0)
                    if i == 0:
                        self.state_list = np.ndarray(
                            [len(self.ard_list), *tmp_state.shape]
                        )
                        self.next_state_list = np.ndarray(
                            [len(self.ard_list), *tmp_next_state.shape]
                        )
                    self.state_list[i] = tmp_state
                    self.next_state_list[i] = tmp_next_state

    def __getitem__(self, idx):
        if self.is_image:
            if self.from_disk:
                tmp_state = []
                tmp_next_state = []
                for j in range(self.framestack_num):
                    s = Image.open(self.state_dir + "%d-%d.png" % (idx, j))
                    tmp_state.append(np.asarray(s))

                    n_s = Image.open(self.next_state_dir + "%d-%d.png" % (idx, j))
                    tmp_next_state.append(np.asarray(n_s))
                tmp_state = np.stack(tmp_state, 0)
                tmp_next_state = np.stack(tmp_next_state, 0)

            else:
                tmp_state = self.state_list[idx]
                tmp_next_state = self.next_state_list[idx]
            batch_ard = self.ard_list[idx]

            return image_post_process(tmp_state, tmp_next_state, batch_ard)
        else:
            return vector_post_process(self.experience_list[idx])

    def __len__(self) -> int:
        if self.is_image:
            return len(self.ard_list)
        else:
            return len(self.experience_list)


if __name__ == "__main__":
    test = SuperviseBuffer("data/experience/pong/200625_155241")
    dataloader = DataLoader(test, batch_size=32, shuffle=True)
    for epoch in range(20):
        for batch_idx, samples in enumerate(dataloader):
            print()
