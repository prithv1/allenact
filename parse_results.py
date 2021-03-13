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
    "clean": "Clean",
    "clean_drift_deg_10": "Drift",
    "clean_mb_const": "M.B. (C)",
    "clean_mb_stoch": "M.B. (S)",
    "clean_motfail": "M.F.",
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
    metrics = ["success", "spl", "soft_spl", "soft_progress", "ep_length"]

    # Load all files
    df_index = []
    for f in files:
        res = json.load(open(f, "r"))[0]
        res = {k: v for k, v in res.items() if k in metrics}
        for k, v in res.items():
            if k in ["success", "spl", "soft_spl", "soft_progress"]:
                res[k] = round(v * 100, 2)
            else:
                res[k] = round(v, 2)
        res_set = None
        for k, v in SETTING_DICT.items():
            if k in f and k + "_" not in f:
                res_set = v
                break
        if res_set is None:
            res_set = "Clean"
        res["setting"] = res_set
        df_index.append(res_set)
        data.append(res)

    # Convert to dataframe
    data_df = pd.DataFrame(data, index=[INDEX.index(x) for x in df_index])
    data_df = data_df.sort_index(ascending=True)
    # print(data_df[["setting"] + metrics])
    print(data_df[["setting"] + metrics].to_latex())


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
