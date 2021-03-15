# Parse the outputs of the evaluation JSONs
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


def parse_results(search_dir):
    # Get all the json files
    files = glob.glob(search_dir + "/**/**/*.json")
    data = []
    metrics = ["success", "spl", "ep_length"]

    # Load all files
    df_index = []
    for f in files:
        res = json.load(open(f, "r"))[0]
        res_set = None
        for k, v in SETTING_DICT.items():
            if k in f and k + "_" not in f:
                res_set = v
                break
        if res_set is None:
            res_set = "Clean"
        for task in res["tasks"]:
            task_dict = {k: v for k, v in task.items() if k in metrics}
            print("Success is", task_dict["success"])
            task_dict["success"] = float(task_dict["success"] == "true")
            task_dict["difficulty"] = task["task_info"]["difficulty"]
            task_dict["setting"] = res_set
            data.append(task_dict)

    # Convert to data-frame
    data_df = pd.DataFrame(data)
    mean_df = data_df.groupby(["setting", "difficulty"], as_index=False)[metrics].mean()
    # print(mean_df)
    print(mean_df[["setting", "difficulty"] + metrics])


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
    parse_results(RES_DIR[args.mode])
