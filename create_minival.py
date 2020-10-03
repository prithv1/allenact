"""
Create RoboTHOR PointNav Mini-val Splits
- No need to modify the distance caches
- Only subsample episodes
- Subsample equal number of easy, medium and hard episodes
"""
import os
import sys
import json
import gzip
import collections

import pandas as pd
import numpy as np

from pprint import pprint


def load_jsongz(filename):
    with gzip.GzipFile(filename, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
    return data


def write_jsongz(filename, data):
    with gzip.GzipFile(filename, "w") as fout:
        fout.write(json.dumps(data).encode("utf-8"))


def create_minival(episode_f, keep=100):
    episodes = load_jsongz(episode_f)
    episode_df = pd.DataFrame(episodes)

    difficulty = [x["difficulty"] for x in episodes]
    id_diff_tuple = [(x["difficulty"], x["id"]) for x in episodes]
    counter = collections.Counter(difficulty)
    pprint(dict(counter))

    easy_ep = [x for x in id_diff_tuple if x[0] == "easy"]
    med_ep = [x for x in id_diff_tuple if x[0] == "medium"]
    hard_ep = [x for x in id_diff_tuple if x[0] == "hard"]

    # Simple Subsampling
    easy_ep = easy_ep[:keep]
    med_ep = med_ep[:keep]
    hard_ep = hard_ep[:keep]
    keep_eps = easy_ep + med_ep + hard_ep
    keep_ids = [x[1] for x in keep_eps]
    keep_ep_df = episode_df[episode_df["id"].isin(keep_ids)]
    sub_episodes = list(keep_ep_df.T.to_dict().values())

    return sub_episodes


def subsample_episodes(EPISODES_PATH, SAVE_PATH, keep=100):
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(SAVE_PATH + "episodes/")

    scenes = [x for x in os.listdir(EPISODES_PATH) if "json.gz" in x]
    for scene in scenes:
        scene_episodes = EPISODES_PATH + scene
        sub_episodes = create_minival(scene_episodes, keep)
        write_jsongz(SAVE_PATH + "episodes/" + scene, sub_episodes)


if __name__ == "__main__":
    DIFF_LEVELS = 3
    NUM_VAL_SCENES = 15
    KEEP = 20
    # KEEP = 2
    DATASPLIT_PATH = "datasets/robothor-pointnav/val/"
    EPISODES_PATH = DATASPLIT_PATH + "episodes/"
    SAVE_PATH = (
        "datasets/robothor-pointnav/minival_" + str(KEEP * DIFF_LEVELS) + "_per_sc/"
    )
    subsample_episodes(EPISODES_PATH, SAVE_PATH, KEEP)
