import torch.utils.data as data
import numpy as np
from bit_pytorch.grid_divider import read_img
from bit_pytorch.utils import read_txt
import os
import torch

import matplotlib.pyplot as plt
from PIL import Image

class GetLoader(data.Dataset):

    def __init__(self, img_folder, annot_path):
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)
        self.classes = {'credential': 0, 'noncredential': 1}

    def __getitem__(self, item):

        image_file = list(set(self.paths))[item] # image path
        img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file] # box coordinates
        img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file] # box types

        if len(img_coords) == 0:
            raise IndexError('list index out of range')

        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential

        grid_arr = read_img(img_path=os.path.join(self.img_folder, image_file+'.png'),
                            coords=img_coords, types=img_classes, grid_num=10)

        return grid_arr, img_label

    def __len__(self):
        return self.num_imgs

if __name__ == '__main__':

    train_set = GetLoader(img_folder='./data/first_round_3k3k/all_imgs',
                          annot_path='./data/first_round_3k3k/all_coords.txt')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, drop_last=False, shuffle=False)

    print(len(train_set))
    for x, y in train_loader:
        print(x.data)
        plt.imshow(Image.open(os.path.join('./data/first_round_3k3k/all_imgs', file_path[0]+'.png')))
        plt.show()
        break
        # print(x)
        # print(y)
