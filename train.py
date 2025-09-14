import datetime
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import timm.optim
import timm.scheduler
import torch
import torch.nn as nn
import yaml
from cruw import CRUW
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset.rod2021 import ROD2021Dataset, collate_fn, data_augment
from utils.confmap import decode_confmap
from utils.evaluate import evaluate_ols

# https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train(config: dict, resume: str, device_name: str):

    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(device_name)

    # Initialize model
    if config['name'] == 'mRadNet':
        from model.mRadNet import mRadNet
        model = mRadNet(
            model_cfg=config['model'],
            dataset_cfg=config['dataset']
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {config['name']}")

    # Initialize datasets
    print('Building training set...')
    train_set = ROD2021Dataset(
        dataset_cfg=config['dataset'],
        model_cfg=config['model'],
        training=True,
        root_path=config['dataset']['root_path']
    )
    print('Building testing set...')
    test_set = ROD2021Dataset(
        dataset_cfg=config['dataset'],
        model_cfg=config['model'],
        training=False,
        root_path=config['dataset']['root_path']
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=os.cpu_count() or 8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count() or 8,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Initialize dataset helper
    dataset_helper = CRUW(data_root=config['dataset']['root_path'],
                          sensor_config_name=config['dataset']['helper_sensor_config'],
                          object_config_name=config['dataset']['helper_object_config'])

    num_epochs = config['train']['num_epochs']

    # Initialize optimizer
    optimizer = timm.optim.adamp.AdamP(
        model.parameters(),
        lr=config['train']['learning_rate']
    )
    scheduler = timm.scheduler.cosine_lr.CosineLRScheduler(
        optimizer, num_epochs,
    )

    loss_fn = nn.SmoothL1Loss().to(device)

    # Resume training
    if resume:
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Create experiment directory
    dt = datetime.datetime.now()
    exp_name = f'{config['name']}_{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'
    print(exp_name)
    exp_dir = os.path.join(config['exp_dir'],
                           f'{dt.year}{dt.month:02d}{dt.day:02d}',
                           exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    shutil.copyfile(f'model/{config['name']}.py',
                    os.path.join(exp_dir, 'model.py'))
    shutil.copyfile(f'train.py', os.path.join(exp_dir, 'train.py'))

    # Initialize Tensorboard
    writer = SummaryWriter(log_dir=exp_dir)

    for epoch in range(start_epoch, num_epochs):

        bar = tqdm(desc=f"Train {epoch+1}/{num_epochs}",
                   total=len(train_loader), dynamic_ncols=True)
        scheduler.step(epoch)  # timm.scheduler

        # Training
        model.train()
        for i, data in enumerate(train_loader):
            inputs = data['rad'].to(device)  # [B, T, R, A, C=2*chirps]
            confmap = data['confmap'].to(device)  # [B, T, R, A, classes]

            # Data augmentation
            inputs, confmap = data_augment(inputs, confmap, rate=0.5)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs['output'], confmap)
            loss.backward()
            optimizer.step()

            bar.update(1)
            writer.add_scalar('train/loss', loss.item(),
                              epoch * len(train_loader) + i)
        bar.close()

        # Validation
        bar = tqdm(desc=f"Valid {epoch+1}/{num_epochs}",
                   total=len(test_loader), dynamic_ncols=True)
        model.eval()
        predictions = {}  # {seq: {frame: [[R, A, class, conf], ...]}}
        for seq in test_set.rads.keys():
            predictions[seq] = {}
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                assert len(data['rad']) == 1, f"Testing batch size must be 1"

                inputs = data['rad'].to(device)  # [1, T, R, A, C=2*chirps]
                confmap = data['confmap'].to(device)  # [1, T, R, A, classes]

                # Forward pass
                outputs = model(inputs)
                loss = loss_fn(outputs['output'], confmap)
                losses.append(loss.item())

                bar.update(1)

                seq, frames = data['seq'][0], data['frames'][0]
                for j, frame in enumerate(frames):
                    if frame in predictions[seq]:
                        continue
                    confmap = outputs['output'][0, j].cpu()
                    predictions[seq][frame] = decode_confmap(
                        confmap, config['dataset'], config['model'])
        bar.close()
        avg_loss = np.mean(losses)
        writer.add_scalar('test/loss', avg_loss, epoch)

        AP, AR = evaluate_ols(dataset_helper, predictions, config)
        writer.add_scalar('test/OLS_AP', AP, epoch)
        writer.add_scalar('test/OLS_AR', AR, epoch)
        print(f"Valid {epoch+1}/{num_epochs} - "
              f"Avg Loss: {avg_loss:.3f}, AP: {AP:.3f}, AR: {AR:.3f}, AP/AR: {AP/AR:.3f}")

        # Save the model
        if (epoch % 2 == 0 and AP > .80) or AP > .85:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(exp_dir, f'checkpoint_{epoch}.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='./config/mRadNet.yaml', type=str)
    parser.add_argument('-r', '--resume', default='', type=str)
    parser.add_argument('-d', '--device', default='cuda:0', type=str)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config, resume=args.resume, device_name=args.device)
