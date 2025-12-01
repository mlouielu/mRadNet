import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from cruw import CRUW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.rod2021 import ROD2021Dataset, collate_fn
from utils.confmap import decode_confmap
from utils.evaluate import evaluate_ols

# https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Monkey patch numpy
if not hasattr(np, "float"):
    np.float = float


def test(config: dict, resume: str, device_name: str):

    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(device_name)

    # Initialize datasets
    print("Building testing set...")
    test_set = ROD2021Dataset(
        dataset_cfg=config["dataset"],
        model_cfg=config["model"],
        training=False,
        root_path=config["dataset"]["root_path"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Initialize dataset helper
    dataset_helper = CRUW(
        data_root=config["dataset"]["root_path"],
        sensor_config_name=config["dataset"]["helper_sensor_config"],
        object_config_name=config["dataset"]["helper_object_config"],
    )

    # Initialize model
    if config["name"] == "mRadNet":
        from model.mRadNet import mRadNet

        model = mRadNet(model_cfg=config["model"], dataset_cfg=config["dataset"]).to(
            device
        )
    else:
        raise ValueError(f"Unknown model name: {config['name']}")

    checkpoint = torch.load(resume, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    bar = tqdm(total=len(test_loader))
    model.eval()
    predictions = {}  # {seq: {frame: [[R, A, class, conf], ...]}}
    for seq in test_set.rads.keys():
        predictions[seq] = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            assert len(data["rad"]) == 1, f"Testing batch size must be 1"

            inputs = data["rad"].to(device)  # [1, T, R, A, C=2*chirps]
            confmap = data["confmap"].to(device)  # [1, T, R, A, classes]

            # Forward pass
            outputs = model(inputs)["output"]

            bar.update(1)

            seq, frames = data["seq"][0], data["frames"][0]
            for j, frame in enumerate(frames):
                if frame in predictions[seq]:
                    continue
                confmap = outputs[0][j].cpu()
                predictions[seq][frame] = decode_confmap(
                    confmap, config["dataset"], config["model"]
                )
    bar.close()

    AP, AR = evaluate_ols(dataset_helper, predictions, config)
    print(f"Test - OLS AP: {AP:.4f}, AR: {AR:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="./config/mRadNet.yaml", type=str)
    parser.add_argument("-r", "--resume", default="", type=str)
    parser.add_argument("-d", "--device", default="cuda:0", type=str)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    test(config, resume=args.resume, device_name=args.device)
