import os
from imageio import imwrite as imsave
import random
from glob import glob

import numpy as np
import torch
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import img2tensor


class ScanDataset(Dataset):
    def __init__(self, image_dir='dataset/synthetic_v1', test_size=None, resize=None):
        self.resize = resize
        self.image_dir = image_dir
        self.all_images = np.array(glob(os.path.join(image_dir, '*.npy')))
        if test_size is not None:
            inds = np.random.permutation(len(self.all_images)).astype(int)
            self.train_ind, self.val_ind = train_test_split(inds, test_size=test_size)
            self.set_mode('train')
        else:
            self.images = self.all_images
        print('Find {} images'.format(len(self.all_images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = np.load(image_path)
        if self.resize and (image.shape[0] != self.resize):
            image = zoom(image, (self.resize[0], self.resize[1], self.resize[2]/image.shape[2]))
        image = img2tensor(image)

        side = os.path.splitext(os.path.basename(image_path))[0].split('_')[1]
        target = 1 if side == 'r' else 0
        target = torch.tensor(target).long()
        sample = {'image': image.unsqueeze(0), 'target': target}
        return sample

    def set_mode(self, mode):
        # mode train or val
        if mode == 'train':
            self.images = self.all_images[self.train_ind]
        elif mode == 'val':
            self.images = self.all_images[self.val_ind]
        else:
            raise ValueError


def create_data(save_path, size=[512, 512, 350], n_samples=400):
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(n_samples)):
        label = 'r' if random.random() > 0.35 else 'l'
        name = f'{i}_{label}.npy'
        add_s = 4 if label == 'r' else -4
        np.save(os.path.join(save_path, name),
                np.random.randint(50, 200, size=[size[0], size[1], size[2] + random.randint(-50, 50)]).astype(np.uint8) + add_s
                )


if __name__ == "__main__":
    create_data(save_path='dataset/synthetic_v1', n_samples=400)
    # dataset = ScanDataset(test_size=0.1, resize=(1/8, 1/8, 48))
    # for i, d in tqdm(enumerate(dataset)):
    #     print(d)
