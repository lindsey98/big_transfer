import torch.utils.data as data
from PIL import Image, ImageOps
import pickle
import numpy as np
from bit_pytorch.grid_divider import read_img
from bit_pytorch.utils import read_txt
import os


class GetLoader(data.Dataset):
    def __init__(self, img_folder, annot_path, transform=None):
        self.transform = transform
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.labels, self.paths, self.preprocess_coordinates, self.classes = read_txt(annot_path)

    def __getitem__(self, item):

        image_file = os.listdir(self.img_folder)[item]
        img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file.split('.png')[0]]
        img_classes = np.asarray(self.classes)[np.asarray(self.paths) == image_file.split('.png')[0]]
        img_label = np.asarray(self.labels)[np.asarray(self.paths) == image_file.split('.png')[0]][0]

        if len(img_label) == 0:
            raise IndexError('list index out of range')

        grid_arr = read_img(img_path=os.path.join(self.img_folder, image_file),
                            coords=img_coords,
                            classes=img_classes)


        if self.transform is not None:
            grid_arr = self.transform(grid_arr)

        return grid_arr, img_label

    def __len__(self):
        return len(os.listdir(self.img_folder))

if __name__ == '__main__':
    data_loader = GetLoader(img_folder='./data/first_round_3k3k/credential', annot_path='./data/first_round_3k3k/all_coords.txt')
    grid_arr, img_label = data_loader.__getitem__(2999)
    print(len(data_loader))
    print(grid_arr)
    print(img_label)