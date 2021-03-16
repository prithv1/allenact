"""
Script to conduct analysis
and extract interesting behaviors
for pointnav and objectnav agents
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint

SETTING_DICT = {
    "cam_crack_s5": "Camera-Crack",
    # "clean": "Clean",
    "clean_drift_deg_10": "Drift",
    "clean_mb_const": "M.B. (C)",
    "clean_mb_stoch": "M.B. (S)",
    "clean_motfail": "M.F.",
    "clean_pyrobot_ilqr_1": "PyRobot-ILQR",
    "defocus_blur_s5": "Defocus Blur",
    "defocus_blur_s5_drift_deg_10": "D.B. + Drift",
    "defocus_blur_s5_mb_stoch": "D.B. + M.B. (S)",
    "fov_s5": "FOV",
    "lighting_s5": "Lighting",
    "motion_blur_s5": "Motion Blur",
    "spatter_s5": "Spatter",
    "spatter_s5_drift_deg_10": "Spt. + Drift",
    "spatter_s5_mb_stoch": "Spt. + M.B. (S)",
    "speckle_noise_s5": "Speckle Noise",
    "speckle_noise_s5_drift_deg_10": "S.N. + Drift",
    "speckle_noise_s5_mb_stoch": "S.N. + M.B. (S)",
}

INDEX = [
    "Clean",
    "Motion Blur",
    "Lighting",
    "FOV",
    "Defocus Blur",
    "Camera-Crack",
    "Speckle Noise",
    "Spatter",
    "M.B. (C)",
    "M.B. (S)",
    "Drift",
    "M.F.",
    "PyRobot-ILQR",
    "D.B. + M.B. (S)",
    "S.N. + M.B. (S)",
    "Spt. + M.B. (S)",
    "D.B. + Drift",
    "S.N. + Drift",
    "Spt. + Drift",
]

# Flatten and store all the results in one dataframe
def parse_results_to_df(search_dir):
    # Get all the json files
    files = glob.glob(search_dir + "/**/**/*.json")
    data = []

    # Load all files
    for i in tqdm(range(len(files))):
        f = files[i]
        res = json.load(open(f, "r"))[0]
        res_set = None
        for k, v in SETTING_DICT.items():
            if k in f and k + "_" not in f:
                res_set = v
                break
        if res_set is None:
            res_set = "Clean"
        for task in res["tasks"]:
            task_dict = {k: v for k, v in task.items() if k != "task_info"}
            for k, v in task["task_info"].items():
                task_dict[k] = v
            task_dict["setting"] = res_set
            data.append(task_dict)

    data_df = pd.DataFrame(data)
    return data_df, data


# Get failed action stats
# Supposed to indicate the fraction of collisions per-episode
def collision_stat(data_df):
    # Try a global approach
    sub_df = data_df[["setting", "action_success"]]
    sub_df["action_success"] = sub_df["action_success"].apply(lambda x: np.mean(x))
    mean_collision_df = sub_df.groupby(["setting"], as_index=False)[
        "action_success"
    ].mean()
    print(mean_collision_df)


if __name__ == "__main__":
    RES_DIR = {
        "pnav_rgb": "storage/robothor-pointnav-rgb-resnetgru-dppo-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
        "pnav_drcn_rgb": "storage/robothor-pointnav-rgb-resnetgru-dppo-drcn-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
        "pnav_daug_rgb": "storage/robothor-pointnav-rgb-daug-resnetgru-dppo-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
        "pnav_rgbd": "storage/robothor-pointnav-rgbd-resnetgru-dppo-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGBD-ResNet-DDPPO",
        "pnav_drcn_rgbd": "storage/robothor-pointnav-rgbd-resnetgru-dppo-drcn-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGBD-ResNet-DDPPO",
        "onav_rgb": "storage/robothor-objectnav-rgb-resnetgru-dppo-s2s-eval/metrics/Objectnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
        "onav_rgbd": "storage/robothor-objectnav-rgbd-resnetgru-dppo-s2s-eval/metrics/Objectnav-RoboTHOR-Vanilla-RGBD-ResNet-DDPPO",
    }

    parser = argparse.ArgumentParser(
        description="allenact", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="pnav_rgb",
        required=True,
        help="Mode specifies the experimental setting",
    )

    args = parser.parse_args()
    data_df, data = parse_results_to_df(RES_DIR[args.mode])

    collision_stat(data_df)
