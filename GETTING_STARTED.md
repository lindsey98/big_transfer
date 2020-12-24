# Getting Started
More information can always be found in the [README.md](./README.md) file.

The Big Transfer (BiT) framework allows anyone to train an image classification model in a wholly automated fashion, using models pretrained on a much larger dataset than ImageNet to achieve good results out of the box (an upgrade from the typical transfer learning pipeline of `from torchvision import models`).

Hyparameters such as input image size, data augmentations and training schedule etc. are all infered from the statistics of the dataset, using what the authors call the BiT-HyperRule. They claim that these hyperparameter settings are designed to generalize across many datasets.

That said, you can definitely achieve better performance by finetuning these hyperparameters on your own specific application.

# Using Custom Datasets
Train on website screenshots to distinguish between credential-requiring / non-credential-requiring page

# Reproducing my Training
As follows was my training process:

## Training 
This command runs the training:
```
python -m bit_pytorch.train \
    --name {exp_name} \  # Name of this run. Used for monitoring and checkpointing.
    --model {FCMax|FCAvg} \  # Which pretrained model to use.
    --logdir {log_dir} \  # Where to log training info.
    --dataset web \  # Name of custom dataset as specified and self-implemented above.
    --base_lr {base_learning_rate} \ # Initial learning rate
    --batch 256 # Batch size
```
```