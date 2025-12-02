import os
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.confmap import encode_confmaps


class ROD2021Dataset(Dataset):
    def __init__(
        self, dataset_cfg: dict, model_cfg: dict, training: bool, root_path: str
    ):
        super().__init__()
        self.root_dir = root_path
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        seq_path = os.path.join(root_path, "sequences", "train" if training else "test")
        if not os.path.exists(dataset_cfg["confamaps_save_path"]):
            os.makedirs(dataset_cfg["confamaps_save_path"])

        # {seq: {frame: rad_data}}
        self.rads: dict[str, dict[int, torch.Tensor]] = {}
        # {seq: {frame: [(r, a, class), ...]}}
        self.annos: dict[str, dict[int, list[tuple[float, float, str]]]] = {}
        # {seq: {frame: confmap}}
        self.confmaps: dict[str, dict[int, torch.Tensor]] = {}
        seqs = sorted(os.listdir(seq_path))
        bar = tqdm(total=len(seqs), dynamic_ncols=True)

        for seq in seqs:
            # find all frames, read RAD data
            bar.set_description(f"Reading radar data for {seq}")

            self.rads[seq] = {}
            self.annos[seq] = {}

            seq_dir = pathlib.Path(os.path.join(seq_path, seq))
            seq_npy = seq_dir / "RADAR_RA_H.npy"
            if seq_npy.exists():
                frames = np.load(seq_npy)  # shape (num_frames, num_chirps, R, A, 2)
                for frame_idx in range(frames.shape[0]):
                    self.rads[seq][frame_idx] = torch.from_numpy(frames[frame_idx])
                    self.annos[seq][frame_idx] = []
            else:
                try:
                    files = list(os.listdir(os.path.join(seq_path, seq, "RADAR_RA_H")))
                except:
                    continue
                frames = sorted(list(set([int(f.split("_")[0]) for f in files])))
                for frame in frames:
                    self.annos[seq][frame] = []
                for frame in frames:
                    num_chirps = len(dataset_cfg["chirps"])
                    self.rads[seq][frame] = torch.zeros(
                        (
                            num_chirps,
                            dataset_cfg["input_size"][0],
                            dataset_cfg["input_size"][1],
                            2,
                        )
                    )
                    for i, chirp in enumerate(dataset_cfg["chirps"]):
                        radar_name = os.path.join(
                            seq_path,
                            seq,
                            "RADAR_RA_H",
                            f"{frame:06d}_{chirp:04d}" + ".npy",
                        )
                        ra = torch.from_numpy(np.load(radar_name))  # [128, 128, 2]
                        self.rads[seq][frame][i, :, :, :] = ra
                        # [1, 128, 128, 2] or [4, 128, 128, 2]

            # read annotations
            bar.set_description(f"Reading annotation for {seq}")
            anno_path = os.path.join(
                self.root_dir,
                "annotations",
                "train" if training else "test",
                f"{seq}.txt",
            )
            with open(anno_path, "r") as f:
                data = f.readlines()
            for line in data:
                frame, r, a, class_name = line.rstrip().split()
                frame = int(frame)
                r = float(r)
                a = float(a)
                self.annos[seq][frame].append((r, a, class_name))

            confmap_file = os.path.join(dataset_cfg["confamaps_save_path"], f"{seq}.pt")
            if os.path.exists(confmap_file):
                self.confmaps[seq] = torch.load(confmap_file, weights_only=False)
            else:
                # generate confmaps
                bar.set_description(f"Generating confmap for {seq}")
                self.confmaps[seq] = encode_confmaps(
                    dataset_cfg, model_cfg, self.annos[seq]
                )
                torch.save(self.confmaps[seq], confmap_file)

            bar.update(1)
        bar.close()

        # list the starting points of the windows
        self.window_start_list: list[tuple[str, int]] = []
        for seq in self.rads.keys():
            shift_size = model_cfg["window_size"]
            shift_step = (
                model_cfg["window_step"] if training else model_cfg["window_size"] // 2
            )
            self.window_start_list += [
                (seq, frame)
                for frame in range(0, len(self.rads[seq]) - shift_size + 1, shift_step)
            ]
            if not training:
                self.window_start_list.append(
                    (seq, len(self.rads[seq]) - shift_size)
                )  # add the last window to cover the end of the sequence

    def __len__(self) -> int:
        return len(self.window_start_list)

    def __getitem__(self, index: int) -> dict:
        # find the frames in the window
        seq, frame_start = self.window_start_list[index]
        frames = list(range(frame_start, frame_start + self.model_cfg["window_size"]))
        data = {}
        data["seq"] = seq
        data["frames"] = frames
        data["rad"] = torch.zeros(
            (
                self.model_cfg["window_size"],
                len(self.dataset_cfg["chirps"]),
                self.dataset_cfg["input_size"][0],
                self.dataset_cfg["input_size"][1],
                2,
            )
        )
        data["confmap"] = torch.zeros(
            (
                self.model_cfg["window_size"],
                self.model_cfg["output_size"][0],
                self.model_cfg["output_size"][1],
                len(self.dataset_cfg["class_names"]),
            )
        )
        data["annos"] = []

        for i, frame in enumerate(frames):
            data["rad"][i] = self.rads[seq][frame]  # [T, chirps, R, A, 2]
            data["confmap"][i] = self.confmaps[seq][frame]  # [T, R, A, classes]
            data["annos"].append(self.annos[seq][frame])

        return data


def collate_fn(batch: list[dict]) -> dict:
    data = {}
    data["seq"] = [b["seq"] for b in batch]
    data["frames"] = [b["frames"] for b in batch]
    data["rad"] = torch.stack([b["rad"] for b in batch], dim=0)
    data["confmap"] = torch.stack([b["confmap"] for b in batch], dim=0)
    data["annos"] = [b["annos"] for b in batch]
    return data


def data_augment(
    input_tensor: torch.Tensor, label_tensor: torch.Tensor, rate: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    # input: [B, T, chirp, R, A, 2]
    # label: [B, T, R, A, classes]
    assert (
        len(input_tensor.shape) == 6 and len(label_tensor.shape) == 5
    ), f"Must be 6D for input and 5D for label, got {input_tensor.shape} and {label_tensor.shape}"
    # Temporal flipping
    if np.random.rand() <= rate:
        input_tensor = torch.flip(input_tensor, dims=[1, 2])
        label_tensor = torch.flip(label_tensor, dims=[1])

    # Horizontal flipping
    if np.random.rand() <= rate:
        input_tensor = torch.flip(input_tensor, dims=[4])
        label_tensor = torch.flip(label_tensor, dims=[3])

    return input_tensor, label_tensor
