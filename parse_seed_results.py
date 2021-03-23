import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint

SEEDS = ["12345", "67891", "23456"]

SETTING_DICT = {
    "cam_crack_s5": "Camera-Crack",
    # "clean": "Clean",
    "clean_drift_deg_10": "Drift",
    "clean_mb_const": "M.B. (C)",
    "clean_mb_stoch": "M.B. (S)",
    "clean_motfail": "M.F.",
    "clean_pyrobot_ilqr_1": "Pyr-ILQR",
    "defocus_blur_s5": "Defocus Blur (S5)",
    "defocus_blur_s5_drift_deg_10": "D.B. (S5) + Drift",
    "defocus_blur_s5_mb_stoch": "D.B. (S5) + M.B. (S)",
    "defocus_blur_s5_pyrobot_ilqr_1": "D.B. (S5) + Pyr-ILQR",
    "defocus_blur_s3": "Defocus Blur (S3)",
    "defocus_blur_s3_drift_deg_10": "D.B. (S3) + Drift",
    "defocus_blur_s3_mb_stoch": "D.B. (S3) + M.B. (S)",
    "defocus_blur_s3_pyrobot_ilqr_1": "D.B. (S3) + Pyr-ILQR",
    "fov_s5": "FOV",
    "lighting_s5": "Lighting (S5)",
    "lighting_s3": "Lighting (S3)",
    "motion_blur_s5": "Motion Blur (S5)",
    "motion_blur_s3": "Motion Blur (S3)",
    "spatter_s5": "Spatter (S5)",
    "spatter_s5_drift_deg_10": "Spt. (S5) + Drift",
    "spatter_s5_mb_stoch": "Spt. (S5) + M.B. (S)",
    "spatter_s5_pyrobot_ilqr_1": "Spt. (S5) + Pyr-ILQR",
    "spatter_s3": "Spatter (S3)",
    "spatter_s3_drift_deg_10": "Spt. (S3) + Drift",
    "spatter_s3_mb_stoch": "Spt. (S3) + M.B. (S)",
    "spatter_s3_pyrobot_ilqr_1": "Spt. (S3) + Pyr-ILQR",
    "speckle_noise_s5": "Speckle Noise (S5)",
    "speckle_noise_s5_drift_deg_10": "S.N. (S5) + Drift",
    "speckle_noise_s5_mb_stoch": "S.N. (S5) + M.B. (S)",
    "speckle_noise_s5_pyrobot_ilqr_1": "S.N. (S5) + Pyr-ILQR",
    "speckle_noise_s3": "Speckle Noise (S3)",
    "speckle_noise_s3_drift_deg_10": "S.N. (S3) + Drift",
    "speckle_noise_s3_mb_stoch": "S.N. (S3) + M.B. (S)",
    "speckle_noise_s3_pyrobot_ilqr_1": "S.N. (S3) + Pyr-ILQR",
}

INDEX = [
    # "Clean",
    "Motion Blur (S3)",
    "Motion Blur (S5)",
    "Lighting (S3)",
    "Lighting (S5)",
    "FOV",
    "Defocus Blur (S3)",
    "Defocus Blur (S5)",
    "Camera-Crack",
    "Speckle Noise (S3)",
    "Speckle Noise (S5)",
    "Spatter (S3)",
    "Spatter (S5)",
    "M.B. (C)",
    "M.B. (S)",
    "Drift",
    "M.F.",
    "D.B. (S3) + M.B. (S)",
    "D.B. (S5) + M.B. (S)",
    "S.N. (S3) + M.B. (S)",
    "S.N. (S5) + M.B. (S)",
    "Spt. (S3) + M.B. (S)",
    "Spt. (S5) + M.B. (S)",
    "D.B. (S3) + Drift",
    "D.B. (S5) + Drift",
    "S.N. (S3) + Drift",
    "S.N. (S5) + Drift",
    "Spt. (S3) + Drift",
    "Spt. (S5) + Drift",
    "D.B. (S3) + Pyr-ILQR",
    "D.B. (S5) + Pyr-ILQR",
    "S.N. (S3) + Pyr-ILQR",
    "S.N. (S5) + Pyr-ILQR",
    "Spt. (S3) + Pyr-ILQR",
    "Spt. (S5) + Pyr-ILQR",
]


def parse_results(search_dir):
    # Get all the json files
    files = glob.glob(search_dir + "/**/**/*.json")
    data = []
    # metrics = ["success", "spl", "soft_spl", "soft_progress", "ep_length"]
    metrics = ["success", "spl"]

    # Load all files
    df_index = []
    for i in tqdm(range(len(files))):
        f = files[i]
        res = json.load(open(f, "r"))[0]
        tasks = res["tasks"]

        seed = None
        for sd in SEEDS:
            if sd in f:
                seed = sd
                break

        res_set = None
        for k, v in SETTING_DICT.items():
            check_str = f.replace("_seed_" + seed, "")
            if k in f and k + "_" not in check_str:
                res_set = v
                break
        if res_set is None:
            res_set = "Clean"

        for task in tasks:
            task_dict = {k: v for k, v in task.items() if k in metrics}
            task_dict["success"] = float(task_dict["success"] == True)
            task_dict["difficulty"] = task["task_info"]["difficulty"]
            task_dict["setting"] = res_set
            task_dict["seed"] = seed
            data.append(task_dict)

    # Convert to data-frame
    data_df = pd.DataFrame(data)
    mean_df = data_df.groupby(["setting"], as_index=False)[metrics].mean()
    print(mean_df)
    sem_df = data_df[["setting"] + metrics]
    # sem_df = sem_df.groupby(["setting"], as_index=False)[metrics].sem()
    sem_df = sem_df.groupby(["setting"], as_index=True)
    sem_df = sem_df.sem().reset_index()

    # Mean rename dict
    mean_rename = {k: k + "_mean" for k in metrics}
    sem_rename = {k: k + "_sem" for k in metrics}
    disp_names = ["success_mean", "success_sem", "spl_mean", "spl_sem"]

    # Rename columns
    mean_df = mean_df.rename(columns=mean_rename)
    sem_df = sem_df.rename(columns=sem_rename)

    # Merge data-frames
    ov_df = pd.merge(mean_df, sem_df, on="setting")

    print(ov_df[["setting"] + disp_names].to_latex())
    return data_df


if __name__ == "__main__":
    RES_DIR = {
        "pnav_rgbd": "storage/robothor-pointnav-rgbd-resnetgru-dppo-s2s-ms-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGBD-ResNet-DDPPO",
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
