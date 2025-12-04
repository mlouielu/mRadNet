import math

import numpy as np
import torch
from cruw.mapping.coor_transform import pol2cart_ramap


def encode_confmaps(dataset_cfg: dict, model_cfg: dict, annotations: dict) -> dict:
    '''
    Encode the annotations into confidence maps.
    Args:
        dataset_cfg (dict): The dataset configuration.
        model_cfg (dict): The model configuration.
        annotations (dict): {seq: [[R, A, class], ...]}
        device (str): The device to use.
    Returns:
        dict: {seq: torch.Tensor, ...}
    '''
    r_range = dataset_cfg['input_range'][0:2]  # [r_min, r_max]
    a_range = dataset_cfg['input_range'][2:4]  # [a_min, a_max]
    map_size = model_cfg['output_size']  # [r_size, a_size]

    confmaps = {}
    for seq in annotations.keys():

        confmap = np.zeros((map_size[0], map_size[1],
                            len(dataset_cfg['class_names'])))
        # [r_size, a_size, num_classes]

        for label in annotations[seq]:
            # [R, A, num_classes]
            obj_size = dataset_cfg['class_sizes'][label[2]]
            obj_r = (label[0] - r_range[0]) / \
                (r_range[1] - r_range[0]) * map_size[0]
            obj_a = (label[1] - a_range[0]) / \
                (a_range[1] - a_range[0]) * map_size[1]
            class_idx = dataset_cfg['class_names'].index(label[2])

            obj_r_min = max(np.floor(obj_r - obj_size/2), 0)
            obj_r_max = min(np.ceil(obj_r + obj_size/2), map_size[0])
            for i in range(int(obj_r_min), int(obj_r_max)):
                obj_a_min = max(np.floor(obj_a - obj_size), 0)
                obj_a_max = min(np.ceil(obj_a + obj_size), map_size[1])
                for j in range(int(obj_a_min), int(obj_a_max)):
                    dist = np.sqrt(((i-obj_r)*2)**2 + (j-obj_a)**2)
                    if dist > obj_size:
                        continue
                    confmap[i, j, class_idx] = 1 - dist / obj_size
        confmaps[seq] = torch.from_numpy(confmap)

    return confmaps


def decode_confmap(confmap: torch.Tensor, dataset_cfg: dict, model_cfg: dict, nms: bool = True) -> list:
    '''
    Decode the confidence map into a list of objects.
    Args:
        confmap (torch.Tensor): The confidence map, shape [R, A, classes].
        dataset_cfg (dict): The dataset configuration.
        model_cfg (dict): The model configuration.
    Returns:
        list: [(R, A, class, conf), ...]
    '''
    confmap_cfg = dataset_cfg['confmap']
    r_min, r_max, a_min, a_max = dataset_cfg['input_range']
    map_size = model_cfg['output_size']  # [r_size, a_size]
    class_sizes = [0.5, 1.0, 3.0]
    peaks: list[tuple] = []
    for class_idx in range(len(dataset_cfg['class_names'])):
        class_conf = confmap[:, :, class_idx].numpy()
        class_peaks = [(i / map_size[0] * (r_max - r_min) + r_min,  # R
                        j / map_size[1] * (a_max - a_min) + a_min,  # A
                        class_idx, np.clip(class_conf[i, j], 0, 1))
                       for i, j in find_3x3_peaks(class_conf)
                       if class_conf[i, j] > confmap_cfg['conf_threshold']]

        peaks.extend(
            l_nms(class_peaks, class_sizes[class_idx],
                  confmap_cfg['lnms_threshold'])
            if nms else class_peaks
        )
    peaks = sorted(peaks, key=lambda p: p[3], reverse=True)[:100]
    return peaks


def find_3x3_peaks(arr: np.ndarray) -> list[tuple]:
    '''
    Find the peaks in a 3x3 neighborhood.
    Args:
        arr (np.ndarray): The input array, shape [R, A].
    Returns:
        list: List of peaks' coordinates [(i, j), ...] where arr[i, j] is a peak.
    '''
    up = np.roll(arr, -1, axis=0)
    down = np.roll(arr, 1, axis=0)
    left = np.roll(arr, -1, axis=1)
    right = np.roll(arr, 1, axis=1)
    neighbor_max = np.max([
        up, down, left, right,
        np.roll(up, -1, axis=1),  # upper left
        np.roll(up, 1, axis=1),  # upper right
        np.roll(down, -1, axis=1),  # down left
        np.roll(down, 1, axis=1),  # down right
    ], axis=0)
    return np.argwhere(arr >= neighbor_max).tolist()


def l_nms(peaks: list[tuple], class_size: float, threshold: float) -> list:
    new_list: list[tuple] = []
    peaks = sorted(peaks, key=lambda p: p[3], reverse=True)
    while len(peaks):
        p_cur = peaks.pop(0)
        new_list.append(p_cur)
        peaks = [p for p in peaks if ols(p_cur, p, class_size) < threshold]
    return new_list


def ols(p1: tuple[float, float, int, float],
        p2: tuple[float, float, int, float],
        class_size: float) -> float:
    '''
    Compute the OLS between two objects.
    Args:
        p1 (tuple): The first object, (R, A, class, conf).
        p2 (tuple): The second object, (R, A, class, conf).
        class_size (float): The size of the class.
    Returns:
        float: The OLS score.
    '''
    assert p1[2] == p2[2], f'Objects must be of the same class, got {p1[2]} and {p2[2]}'
    x1, y1 = pol2cart_ramap(p1[0], p1[1])
    x2, y2 = pol2cart_ramap(p2[0], p2[1])
    dx = x1 - x2
    dy = y1 - y2
    s_square = x1 ** 2 + y1 ** 2
    kappa = class_size / 100
    e = (dx ** 2 + dy ** 2) / (2 * s_square * kappa)
    return math.exp(-e)
