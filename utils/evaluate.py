import os
import shutil
import time

from cruw import CRUW
from .eval_rod2021 import evaluate_rod2021


def evaluate_ols(dataset_helper: CRUW, predictions: dict, config: dict) -> tuple[float, float]:
    '''
    Evaluate the predictions using the OLS metric.
    Args:
        dataset_helper (CRUW): The cruw-devkit helper.
        predictions (dict): {seq: {frame: [[R, A, class, conf], ...]}}
        dataset_cfg (dict): The dataset configuration.
    Returns:
        dict: The evaluation results.
    '''
    tmp_dir = f'{config['eval_tmp_dir']}_{time.time()}'
    dataset_cfg = config['dataset']
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Save predictions
    for seq in predictions.keys():
        if os.path.exists(os.path.join(tmp_dir, seq+'.txt')):
            os.remove(os.path.join(tmp_dir, seq+'.txt'))
        with open(os.path.join(tmp_dir, seq+'.txt'), 'w') as f:
            for frame in predictions[seq].keys():
                pred = predictions[seq][frame]
                if len(pred) == 0:
                    continue
                for p in pred:
                    f.write(f'{int(frame)} '
                            f'{p[0]} '
                            f'{p[1]} '
                            f'{dataset_cfg['class_names'][p[2]]} '
                            f'{p[3]}\n')
    AP, AR = evaluate_rod2021(tmp_dir, os.path.join(dataset_cfg['root_path'], 'annotations', 'test'), dataset_helper)
    shutil.rmtree(tmp_dir)
    return AP, AR
