# Getting Started
More information can always be found in the [README.md](./README.md) file.

The Big Transfer (BiT) framework allows anyone to train an image classification model in a wholly automated fashion, using models pretrained on a much larger dataset than ImageNet to achieve good results out of the box (an upgrade from the typical transfer learning pipeline of `from torchvision import models`).

Hyparameters such as input image size, data augmentations and training schedule etc. are all infered from the statistics of the dataset, using what the authors call the BiT-HyperRule. They claim that these hyperparameter settings are designed to generalize across many datasets.

That said, you can definitely achieve better performance by finetuning these hyperparameters on your own specific application.

# Installation
Follow the installation instructions in the [README.md](./README.md) file.

# Changes from the Original Code
- Added feature extractor method (`model.features(x)`) for the ResNetv2 model.
- Supports custom PyTorch datasets.
- Supports finetuning on different number of classes.

# Using Custom Datasets
1. In the [bit_hyperrule.py](./bit_hyperrule.py) file, specify your `dataset` name and image size in the `known_dataset_sizes` variable.
    - The BiT-HyperRule resizes your images using this rule: `(160, 128) if img_area < 96*96 else (512, 480)`
2. In the [bit_pytorch/train.py](./bit_pytorch/train.py) file, define your own custom dataloading logic (at line 76), i.e. the `train_set` and `valid_set` variables should be PyTorch `Dataset` objects.

# Reproducing my Training
As follows was my training process:

## Pretraining on Logo2k
I first pretrained on the Logos2k dataset, using a pretrained BiT-M ResNet50x1 model, which we have to download first:
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz
```
This command runs the fine-tuning on the downloaded model:
```
python -m bit_pytorch.train \
    --name {exp_name} \  # Name of this run. Used for monitoring and checkpointing.
    --model BiT-M-R50x1 \  # Which pretrained model to use.
    --logdir {log_dir} \  # Where to log training info.
    --dataset logo_2k \  # Name of custom dataset as specified and self-implemented above.
```

## Finetuning on Application
Saving and utilizing the weights in the previous step, I finetune the model once again on our intended task:
```
python -m bit_pytorch.train \
    --name {exp_name} \  # Name of this run. Used for monitoring and checkpointing.
    --model BiT-M-R50x1 \  # Which pretrained model to use.
    --logdir {log_dir} \  # Where to log training info.
    --dataset targetlist \  # Name of custom dataset as specified and self-implemented above.
    --weights_path {weights_path} \  # Path to weights saved in the previous step, i.e. bit.pth.tar.
```