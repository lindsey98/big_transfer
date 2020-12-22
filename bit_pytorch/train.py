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
from torchsummary import summary

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

  # TODO: Define custom dataloading logic here for custom datasets
  train_set = GetLoader(img_folder='./data/first_round_3k3k/all_imgs',
                        annot_path='./data/first_round_3k3k/all_coords.txt')

  valid_set = GetLoader(img_folder='./data/first_round_3k3k/all_imgs',
                        annot_path='./data/first_round_3k3k/all_coords.txt')

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")
  logger.info(f"Num of classes: {len(valid_set.classes)}")

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=512, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  train_loader = torch.utils.data.DataLoader(
      train_set, batch_size=args.batch, shuffle=True,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, logger, step):
  # Switch to evaluate mode
  model.eval()
  logger.info("Running validation...")
  logger.flush()

  correct = 0
  total = 0
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True, dtype=torch.float)
      y = y.to(device, non_blocking=True, dtype=torch.long)

      # Compute output, measure accuracy
      logits = model(x)
      preds = torch.argmax(logits, dim=1)
      correct += preds.eq(y).sum().item()
      total += len(logits)
      print(float(correct/total))

  model.train()
  logger.info(f"top1 {float(correct/total):.2%}, ")
  logger.flush()
  return float(correct/total)

# def mixup_data(x, y, l):
#   """Returns mixed inputs, pairs of targets, and lambda"""
#   indices = torch.randperm(x.shape[0]).to(x.device)
#
#   mixed_x = l * x + (1 - l) * x[indices]
#   y_a, y_b = y, y[indices]
#   return mixed_x, y_a, y_b

# def mixup_criterion(criterion, pred, y_a, y_b, l):
#   return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

def main(args):
  writer = SummaryWriter(os.path.join(args.logdir, args.name, 'tensorboard_write3'), flush_secs=60)
  logger = bit_common.setup_logger(args)

  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)
  model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes), grid_num=10)

  # Note: no weight-decay!
  step = 0
  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit3.pth.tar")
  # try:
  #   logger.info(f"Model will be saved in '{savename}'")
  #   checkpoint = torch.load(savename, map_location="cpu")
  #   logger.info(f"Found saved model to resume from at '{savename}'")
  #   step = checkpoint["step"]
  #   model.load_state_dict(checkpoint["model"])
  #   optim.load_state_dict(checkpoint["optim"])
  #   logger.info(f"Resumed at step {step}")
  # except FileNotFoundError:
  #   logger.info("Fine-tuning from BiT")

  # Print the model summary
  model = model.to(device)
  summary(model, (9, 10, 10))

  # Start training
  optim.zero_grad()
  model.train()
  cri = torch.nn.CrossEntropyLoss().to(device)

  logger.info("Starting training!")

  # Get initial training acc
  correct_rate = run_eval(model, valid_loader, device, logger, 0)
  logger.info(f"[Initial training accuracy {correct_rate}]")
  logger.flush()
  writer.add_scalar('train_top1_acc', correct_rate, 0)
  writer.flush()

  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):
      print('Batch input shape:', x.shape)
      print('Batch target shape:', y.shape)

      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True, dtype=torch.float)
      y = y.to(device, non_blocking=True, dtype=torch.long)

      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr

      # Compute output
      logits = model(x)
      c = cri(logits, y)
      c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

      # BP
      optim.zero_grad()
      c.backward()
      optim.step()
      step += 1

      # Write
      logger.info(f"[step {step}]: loss={c_num:.5f} (lr={lr})")  # pylint: disable=logging-format-interpolation
      logger.flush()

      # ...log the gradients/weights
      writer.add_scalar('training_loss',  c_num, step)
      writer.flush()
      writer.add_histogram('model.fc1.weights', model.fc1.weight.data,step)
      writer.flush()
      writer.add_histogram('model.fc2.weights', model.fc2.weight.data, step)
      writer.flush()
      writer.add_histogram('model.fc3.weights', model.fc3.weight.data, step)
      writer.flush()
      writer.add_histogram('model.fc1.grad', model.fc1.weight.grad.data, step)
      writer.flush()
      writer.add_histogram('model.fc2.grad', model.fc2.weight.grad.data, step)
      writer.flush()
      writer.add_histogram('model.fc3.grad', model.fc3.weight.grad.data, step)
      writer.flush()
      writer.add_histogram('model.input', x.data, step)
      writer.flush()
      writer.add_histogram('model.input.grad', x.grad.data, step)
      writer.flush()


      # Get train_acc
      if step % 20 == 0:
          correct_rate = run_eval(model, valid_loader, device, logger, step)  # TODO: Final eval at end of training.
          writer.add_scalar('train_top1_acc', correct_rate, step)
          writer.flush()

          # Save model
          torch.save({
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
          }, savename)


    # TODO: Final eval at end of training.
    # run_eval(model, valid_loader, device, logger, step='end')


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--workers", type=int, default=0,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  main(parser.parse_args())
