# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv


import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule

from .dataloader import GetLoader
from torch.utils.tensorboard import SummaryWriter
import os

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i


def mktrainval(args, logger):
  """Returns train and validation datasets."""
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  train_tx = tv.transforms.Compose([
      # tv.transforms.Resize((precrop, precrop)),
      # tv.transforms.RandomCrop((crop, crop)),
      # tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      # tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)

  # TODO: Define custom dataloading logic here for custom datasets
  elif args.dataset == "web":
    train_set = GetLoader(img_folder='./data/first_round_3k3k/all_imgs',
                          annot_path='./data/first_round_3k3k/all_coords.txt')

    valid_set = GetLoader(img_folder='./data/first_round_3k3k/all_imgs',
                          annot_path='./data/first_round_3k3k/all_coords.txt')
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")
  logger.info(f"Num of classes: {len(valid_set.classes)}")

  micro_batch_size = args.batch // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, logger, step):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1 = [], []
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True, dtype=torch.float)
      y = y.to(device, non_blocking=True, dtype=torch.long)

      # compute output, measure accuracy and record loss.
      logits = model(x)
      c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
      top1 = topk(logits, y)[0]
      all_c.extend(c.cpu())  # Also ensures a sync point.
      all_top1.extend(top1.cpu())

    logger.info(f"Validation batch {b:d}, "
                f"Validation@{step} loss {np.mean(all_c):.5f}, "
                f"top1 {np.mean(all_top1):.2%}, ")

  model.train()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, ")
  logger.flush()
  return all_c, all_top1


def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b

def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

def main(args):
  writer = SummaryWriter(os.path.join(args.logdir, args.name, 'tensorboard_write'))
  logger = bit_common.setup_logger(args)

  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)
  print(len(valid_loader))
  model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes), zero_head=True)

  step = 0
  # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
  # If pretrained weights are specified
  if args.weights_path:
    # logger.info(f"Loading weights from {args.weights_path}")
    checkpoint = torch.load(args.weights_path, map_location="cpu")
    # New task might have different classes; remove the pretrained classifier weights
    del checkpoint['model']['module.head.fc1.weight']
    del checkpoint['model']['module.head.fc1.bias']
    del checkpoint['model']['module.head.fc2.weight']
    del checkpoint['model']['module.head.fc2.bias']
    model.load_state_dict(checkpoint["model"], strict=False)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit.pth.tar")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at step {step}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")

  model = model.to(device)
  optim.zero_grad()

  model.train()
  cri = torch.nn.CrossEntropyLoss().to(device)

  logger.info("Starting training!")
  accum_steps = 0

  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):

      # Schedule sending to GPU(s)
      print(x.shape)
      print(y.shape)
      x = x.to(device, non_blocking=True, dtype=torch.float)
      y = y.to(device, non_blocking=True, dtype=torch.long)

      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr

      # compute output
      logits = model(x)
      c = cri(logits, y)
      c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

      # Accumulate grads
      (c / args.batch_split).backward()
      accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr})")  # pylint: disable=logging-format-interpolation
      logger.flush()
      # ...log the running loss
      writer.add_scalar('training_loss',  c_num, accum_steps)
      writer.close()
      writer.add_histogram('model.fc1.weights', model.fc1.weight.data, accum_steps)
      writer.close()
      writer.add_histogram('model.fc2.weights', model.fc2.weight.data, accum_steps)
      writer.close()
      writer.add_histogram('model.fc1.grad', model.fc1.weight.grad.data, accum_steps)
      writer.close()
      writer.add_histogram('model.fc2.grad', model.fc2.weight.grad.data, accum_steps)
      writer.close()

      # Update params
      if accum_steps == args.batch_split:
        optim.step()
        optim.zero_grad()
        step += 1
        accum_steps = 0

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          _, eval_all_top1 = run_eval(model, valid_loader, device, logger, step)  # TODO: Final eval at end of training.
          writer.add_scalar('eval_top1_acc', np.mean(eval_all_top1), step)
          writer.close()
          if args.save:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim" : optim.state_dict(),
            }, savename)

    # TODO: Final eval at end of training.
    # run_eval(model, valid_loader, device, logger, step='end')



if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  # parser.add_argument("--datadir", required=True,
  #                     help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=0,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  main(parser.parse_args())
