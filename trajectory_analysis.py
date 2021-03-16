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
    sub_df["collisions"] = sub_df["action_success"].apply(
        lambda x: (len(x) - np.sum(x))
    )
    mean_collision_df = (
        sub_df.groupby(["setting"], as_index=False)["collisions"].mean().round(2)
    )
    print(mean_collision_df[["setting", "collisions"]])


# Minimum distance to target
# Supposed to indicate the closest the agent arrives to the target
def min_dist_stat(data_df):
    # Try a global approach
    sub_df = data_df[["setting", "far_from_goal"]]
    sub_df["min_dist"] = sub_df["far_from_goal"].apply(lambda x: np.min(x))
    min_dist_df = (
        sub_df.groupby(["setting"], as_index=False)["min_dist"].mean().round(2)
    )
    print(min_dist_df[["setting", "min_dist"]])


def goal_out_range_took_end(x):
    taken_actions = x["taken_actions"]
    goal_in_range = x["goal_in_range"]
    n_satisfy = len(
        [i for i, j in zip(taken_actions, goal_in_range) if i == "End" and not j]
    )
    return 100 * n_satisfy


def stop_fail_pos(data_df):
    """
    Number of times the agent invokes an end action
    when the goal is not in range
    """
    # Try a global approach
    sub_df = data_df[data_df["took_end_action"] == True]
    sub_df = sub_df[["setting", "success", "taken_actions", "goal_in_range"]]
    sub_df["stop_fail_pos"] = sub_df.apply(lambda x: goal_out_range_took_end(x), axis=1)
    stop_fail_pos_df = sub_df.groupby(["setting"], as_index=False)[
        "stop_fail_pos"
    ].mean()
    print(stop_fail_pos_df[["setting", "stop_fail_pos"]])


def goal_in_range_not_took_end(x):
    taken_actions = x["taken_actions"]
    goal_in_range = x["goal_in_range"]
    n_satisfy = len(
        [i for i, j in zip(taken_actions, goal_in_range) if i != "End" and j]
    )
    n_goal_in_range = len([i for i in goal_in_range if i])
    return 100 * float(n_satisfy / float(n_goal_in_range))


def stop_fail_neg(data_df):
    """
    Number of times the agent fails to invoke an end action
    when the goal is in range
    """
    sub_df = data_df[data_df.apply(lambda x: True in x["goal_in_range"], axis=1)]
    sub_df = sub_df[["setting", "success", "taken_actions", "goal_in_range"]]
    sub_df["stop_fail_neg"] = sub_df.apply(
        lambda x: goal_in_range_not_took_end(x), axis=1
    )
    stop_fail_neg_df = sub_df.groupby(["setting"], as_index=False)[
        "stop_fail_neg"
    ].mean()
    print(stop_fail_neg_df[["setting", "stop_fail_neg"]])


# def get_hist(x):
#     hist, bins = np.histogram(x[""])

# def succ_dist_curve_state(data_df):
#     # Try a global approach
#     sub_df = data_df[["setting", "success", "distance_to_target"]]
#     min_dist = min(sub_df["distance_to_target"].to_list())
#     max_dist = max(sub_df["distance_to_target"].to_list())
#     bins = np.arange(min_dist, max_dist, 0.5)
#     sub_df =


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

    # collision_stat(data_df)
    # min_dist_stat(data_df)
    # stop_fail_pos(data_df)
    stop_fail_neg(data_df)
