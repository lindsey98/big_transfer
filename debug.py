
import numpy as np
import torch
import torchvision as tv
from bit_pytorch.utils import read_xml
import os

a = np.zeros((9, 20, 20))
# print(a.shape)
train_tx = tv.transforms.Compose([
      tv.transforms.ToTensor(),
  ])

grid_arr = train_tx(a)
# print(grid_arr.shape)

# print(len(set([x.strip().split('\t')[0] for x in open('./data/first_round_3k3k/all_coords.txt').readlines()])))


dir = './data/first_round_3k3k/noncredential-q1_xml'
coord_path = './data/first_round_3k3k/all_coords.txt'
for file in os.listdir(dir):
    if file.endswith('.xml'):
        types, boxes = read_xml(os.path.join(dir, file))
        for j in range(len(types)):
            with open(coord_path, 'a+') as f:
                f.write(file.split('.xml')[0] + '\t')
                f.write('(' + ','.join(list(map(str, boxes[j]))) + ')' + '\t')
                f.write(types[j] + '\t')
                f.write('noncredential' + '\n')