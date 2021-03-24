"""
Filter trajectories so that they can be visualized
as videos by a different script
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

SUBSET_SEARCH = [
    "Clean",
    "Motion Blur",
    "Lighting",
    "Defocus Blur",
    "Spatter",
    "Speckle Noise",
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
        tasks = res["tasks"]

        res_set = None
        for k, v in SETTING_DICT.items():
            if k in f and k + "_" not in f:
                res_set = v
                break
        if res_set is None:
            res_set = "Clean"

        for task in tasks:
            task_dict = {k: v for k, v in task.items() if k != "task_info"}
            for k, v in task["task_info"].items():
                task_dict[k] = v
            task_dict["success"] = float(task_dict["success"] == True) * 100.0
            task_dict["spl"] = task_dict["spl"] * 100.0
            task_dict["difficulty"] = task["task_info"]["difficulty"]
            task_dict["setting"] = res_set
            if res_set == "Clean":
                task_dict["corruption"] = None
                # task_dict["severity"] = None
            else:
                task_dict["corruption"] = res_set
                # task_dict["severity"] = 5
            if res_set not in [
                "Clean",
                "Camera-Crack",
                "FOV",
                "M.B. (C)",
                "M.B. (S)",
                "M.F.",
                "Drift",
                "PyRobot-ILQR",
            ]:
                task_dict["anno_text"] = res_set + " Sev. " + str(5)
                task_dict["severity"] = 5
            else:
                task_dict["anno_text"] = res_set
                task_dict["severity"] = None
            data.append(task_dict)
    data_df = pd.DataFrame(data)
    return data_df


def filter_res_df(data_df, succ_thresh):
    """
    We want to filter instances per-ID such that
    success is 100 under clean settings
    but fails under all other settings
    Let the sum_thresh argument decide this.
    This might vary for pointnav and objectnav

    - We want to group by IDs and check for this
    """
    filter_ep = []
    data_df = data_df[data_df["setting"].isin(SUBSET_SEARCH)]
    # Episode IDs
    ep_ids = data_df["id"].tolist()
    print("Looping over episode IDs")
    for i in tqdm(range(len(ep_ids))):
        # if i >= 300:
        #     break
        ep_id = ep_ids[i]
        sub_df = data_df[data_df["id"] == ep_id]
        # Check clean success
        clean_succ = sub_df[sub_df["setting"] == "Clean"]["success"].tolist()[0]
        if clean_succ == 100.0:
            rest_succ = sub_df[sub_df["setting"] != "Clean"]["success"].tolist()
            sum_succ = np.sum(rest_succ)
            if sum_succ <= succ_thresh:
                filter_ep.append(sub_df.T.to_dict().values())

    print(len(filter_ep))
    return filter_ep


if __name__ == "__main__":
    RES_DIR = {
        "pnav_rgb": "storage/robothor-pointnav-rgb-resnetgru-dppo-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO",
        "pnav_rgbd": "storage/robothor-pointnav-rgbd-resnetgru-dppo-s2s-eval/metrics/Pointnav-RoboTHOR-Vanilla-RGBD-ResNet-DDPPO",
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

    PNAV_THRESH = 300.0
    ONAV_THRESH = 200.0

    if "pnav" in args.mode:
        THRESH = PNAV_THRESH
    else:
        THRESH = ONAV_THRESH

    data_df = parse_results_to_df(RES_DIR[args.mode])
    filter_ep = filter_res_df(data_df, THRESH)
    with open(args.mode + ".json", "w") as f:
        json.dump(filter_ep, f)
