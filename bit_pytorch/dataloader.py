import torch.utils.data as data
from PIL import Image, ImageOps
import pickle
import numpy as np
from bit_pytorch.grid_divider import read_img
from bit_pytorch.utils import read_txt
import os
import torch

# def recycle(iterable):
#   """Variant of itertools.cycle that does not save iterates."""
  # while True:
  #   for i in iterable:
      # yield i

class GetLoader(data.Dataset):
    def __init__(self, img_folder, annot_path):
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)
        self.classes = {'credential': 0, 'noncredential':1}


    def __getitem__(self, item):

        image_file = list(set(self.paths))[item]
        img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file]
        img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file]

        if len(img_coords) == 0:
            raise IndexError('list index out of range')

        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]]

        grid_arr = read_img(img_path=os.path.join(self.img_folder, image_file+'.png'),
                            coords=img_coords,
                            classes=img_classes)

        return grid_arr, img_label

    def __len__(self):
        return self.num_imgs

if __name__ == '__main__':
    train_set = GetLoader(img_folder='./data/first_round_3k3k/all_imgs',
                          annot_path='./data/first_round_3k3k/all_coords.txt')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, drop_last=False,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=512))

    for x, y in train_loader:
        x_arr = x.numpy()
        print(x)
        print(y)
