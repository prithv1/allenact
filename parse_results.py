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
    "clean_drift": "Drift",
    "clean_mb_const": "M.B. (C)",
    "clean_mb_stoch": "M.B. (S)",
    "clean_motfail": "M.F.",
    "defocus_blur_s5": "Defocus Blur",
    "defocus_blur_s5_drift": "D.B. + Drift",
    "defocus_blur_s5_mb_stoch": "D.B. + M.B. (S)",
    "fov_s5": "FOV",
    "lighting_s5": "Lighting",
    "motion_blur_s5": "Motion Blur",
    "spatter_s5": "Spatter",
    "spatter_s5_drift": "Spt. + Drift",
    "spatter_s5_mb_stoch": "Spt. + M.B. (S)",
    "speckle_noise_s5": "Speckle Noise",
    "speckle_noise_s5_drift": "S.N. + Drift",
    "speckle_noise_s5_mb_stoch": "S.N. + M.B. (S)",
}


def parse_results(search_dir):
    # Get all the json files
    files = glob.glob(search_dir + "/**/**/*.json")
    data = []
    metrics = ["success", "spl", "soft_spl", "soft_progress", "ep_length"]

    # Load all files
    for f in files:
        res = json.load(open(f, "r"))[0]
        res = {k: v for k, v in res.items() if k in metrics}
        res_set = None
        for k, v in SETTING_DICT.items():
            if k in f:
                res_set = k
                break
        if res_set is None:
            res_set = "clean"
        res["setting"] = res_set
        data.append(res)

    # Convert to dataframe
    data_df = pd.DataFrame(data)
    print(data_df)


if __name__ == "__main__":
    RES_DIR = {
        "pnav_rgb": "storage/robothor-pointnav-rgb-resnetgru-dppo-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
        "pnav_drcn_rgb": "storage/robothor-pointnav-rgb-resnetgru-dppo-drcn-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
        "pnav_daug_rgb": "storage/robothor-pointnav-rgb-daug-resnetgru-dppo-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
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
