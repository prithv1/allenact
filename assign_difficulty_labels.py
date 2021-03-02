"""
Latest ObjectNav dataset does not have
difficulty labels (this is not a problem in PointNav)
- Load the episodes per-split-per-scene
- Sort the shortest path lengths
- Lower 20% threshold (Easy)
- 20-60% threshold (Medium)
- >60% threshold (Hard)
"""
import os
import sys
import json
import math
import gzip
import collections

import pandas as pd
import numpy as np

from tqdm import tqdm
from pprint import pprint


def load_jsongz(filename):
    with gzip.GzipFile(filename, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
    return data


def write_jsongz(filename, data):
    with gzip.GzipFile(filename, "w") as fout:
        fout.write(json.dumps(data).encode("utf-8"))


def assign_labels(episode_f, lt=0.2, mt=0.6):
    episodes = load_jsongz(episode_f)
    episode_df = pd.DataFrame(episodes)

    # Get shortest path lengths
    shortest_path_len = episode_df["shortest_path_length"].tolist()
    shortest_path_len.sort()

    # Get difficulty thresholds
    easy_thresh = shortest_path_len[math.ceil(0.2 * len(shortest_path_len))]
    medium_thresh = shortest_path_len[math.ceil(0.6 * len(shortest_path_len))]

    print("Easy Threshold ", easy_thresh)
    print("Medium Threshold ", medium_thresh)

    for i in tqdm(range(len(episodes))):
        slen = episodes[i]["shortest_path_length"]
        if slen < easy_thresh:
            episodes[i]["difficulty"] = "easy"
        elif slen < medium_thresh and slen >= easy_thresh:
            episodes[i]["difficulty"] = "medium"
        elif slen >= medium_thresh:
            episodes[i]["difficulty"] = "hard"

    # Check difficulty assignments
    difficulty = [x["difficulty"] for x in episodes]
    id_diff_tuple = [(x["difficulty"], x["id"]) for x in episodes]

    easy_ep = [x for x in id_diff_tuple if x[0] == "easy"]
    med_ep = [x for x in id_diff_tuple if x[0] == "medium"]
    hard_ep = [x for x in id_diff_tuple if x[0] == "hard"]

    print("Overall distribution of episodes according to difficulty")
    print("Easy ", len(easy_ep), float(100 * len(easy_ep) / len(episodes)), "%")
    print("Medium ", len(med_ep), float(100 * len(med_ep) / len(episodes)), "%")
    print("Hard ", len(hard_ep), float(100 * len(hard_ep) / len(episodes)), "%")

    return episodes


def create_diff_assigned_splits(EPISODES_PATH,):
    scenes = [x for x in os.listdir(EPISODES_PATH) if "json.gz" in x]
    for scene in scenes:
        print("*" * 10)
        print("Scene is ", scene)
        scene_episodes = EPISODES_PATH + scene
        diff_assigned_episodes = assign_labels(scene_episodes)
        # write_jsongz(scene_episodes, diff_assigned_episodes)

        print("-" * 10)
        print("Total Assigned Tasks ", len(diff_assigned_episodes))
        print("-" * 10)


if __name__ == "__main__":

    # ObjectNav
    print("ObjectNav Splits..")
    NUM_VAL_SCENES = 60
    DATASPLIT_PATH = "datasets/robothor-objectnav/train/"
    # NUM_VAL_SCENES = 15
    # DATASPLIT_PATH = "datasets/robothor-objectnav/val/"
    EPISODES_PATH = DATASPLIT_PATH + "episodes/"
    create_diff_assigned_splits(EPISODES_PATH,)
