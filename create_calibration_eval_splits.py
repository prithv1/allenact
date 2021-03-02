"""
Create RoboTHOR mini-val splits
(for both PointNav and ObjectNav)
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


def create_minival(episode_f, split_ratio=0.4):
    episodes = load_jsongz(episode_f)
    episode_df = pd.DataFrame(episodes)

    difficulty = [x["difficulty"] for x in episodes]
    id_diff_tuple = [(x["difficulty"], x["id"]) for x in episodes]
    counter = collections.Counter(difficulty)
    pprint(dict(counter))

    easy_ep = [x for x in id_diff_tuple if x[0] == "easy"]
    med_ep = [x for x in id_diff_tuple if x[0] == "medium"]
    hard_ep = [x for x in id_diff_tuple if x[0] == "hard"]

    print("Overall distribution of episodes according to difficulty")
    print("Easy ", len(easy_ep))
    print("Medium ", len(med_ep))
    print("Hard ", len(hard_ep))

    # Get calibration and evaluation indices
    easy_calib_ind = int(len(easy_ep) * split_ratio)
    med_calib_ind = int(len(med_ep) * split_ratio)
    hard_calib_ind = int(len(hard_ep) * split_ratio)

    easy_calib_ep = easy_ep[:easy_calib_ind]
    med_calib_ep = med_ep[:med_calib_ind]
    hard_calib_ep = hard_ep[:hard_calib_ind]

    easy_eval_ep = [x for x in easy_ep if x not in easy_calib_ep]
    med_eval_ep = [x for x in med_ep if x not in med_calib_ep]
    hard_eval_ep = [x for x in hard_ep if x not in hard_calib_ep]

    calib_ep = easy_calib_ep + med_calib_ep + hard_calib_ep
    eval_ep = easy_eval_ep + med_eval_ep + hard_eval_ep

    print("Calibration Episodes ", len(calib_ep))
    print("Evaluation Episodes ", len(eval_ep))

    calib_keep_ids = [x[1] for x in calib_ep]
    eval_keep_ids = [x[1] for x in eval_ep]

    calib_ep_df = episode_df[episode_df["id"].isin(calib_keep_ids)]
    eval_ep_df = episode_df[episode_df["id"].isin(eval_keep_ids)]

    calib_episodes = list(calib_ep_df.T.to_dict().values())
    eval_episodes = list(eval_ep_df.T.to_dict().values())

    return eval_episodes, calib_episodes


def create_calib_eval_splits(
    EPISODES_PATH, CALIIB_SAVE_PATH, EVAL_SAVE_PATH, split_ratio=0.4
):
    if not os.path.isdir(CALIB_SAVE_PATH):
        os.mkdir(CALIB_SAVE_PATH)
        os.mkdir(CALIB_SAVE_PATH + "episodes/")

    if not os.path.isdir(EVAL_SAVE_PATH):
        os.mkdir(EVAL_SAVE_PATH)
        os.mkdir(EVAL_SAVE_PATH + "episodes/")

    calib_counter, eval_counter = 0, 0

    scenes = [x for x in os.listdir(EPISODES_PATH) if "json.gz" in x]
    for scene in scenes:
        print("*" * 10)
        print("Scene is ", scene)
        scene_episodes = EPISODES_PATH + scene
        eval_episodes, calib_episodes = create_minival(scene_episodes, split_ratio)
        calib_counter += len(calib_episodes)
        eval_counter += len(eval_episodes)
        write_jsongz(CALIB_SAVE_PATH + "episodes/" + scene, calib_episodes)
        write_jsongz(EVAL_SAVE_PATH + "episodes/" + scene, eval_episodes)

    print("-" * 10)
    print("Total Calibration Tasks ", calib_counter)
    print("Total Evaluation Tasks ", eval_counter)
    print("-" * 10)


if __name__ == "__main__":

    # PointNav
    # print("PointNav Splits..")
    # DIFF_LEVELS = 3
    # NUM_VAL_SCENES = 15
    # SPLIT_RATIO = 0.4
    # DATASPLIT_PATH = "datasets/robothor-pointnav/val/"
    # EPISODES_PATH = DATASPLIT_PATH + "episodes/"
    # CALIB_SAVE_PATH = (
    #     "datasets/robothor-pointnav/calib_val_" + str(SPLIT_RATIO) + "_per_sc/"
    # )
    # EVAL_SAVE_PATH = (
    #     "datasets/robothor-pointnav/eval_val_" + str(SPLIT_RATIO) + "_per_sc/"
    # )
    # create_calib_eval_splits(
    #     EPISODES_PATH, CALIB_SAVE_PATH, EVAL_SAVE_PATH, SPLIT_RATIO
    # )

    # ObjectNav
    print("ObjectNav Splits..")
    DIFF_LEVELS = 3
    NUM_VAL_SCENES = 15
    SPLIT_RATIO = 0.4
    DATASPLIT_PATH = "datasets/robothor-objectnav/val/"
    EPISODES_PATH = DATASPLIT_PATH + "episodes/"
    CALIB_SAVE_PATH = (
        "datasets/robothor-objectnav/calib_val_" + str(SPLIT_RATIO) + "_per_sc/"
    )
    EVAL_SAVE_PATH = (
        "datasets/robothor-objectnav/eval_val_" + str(SPLIT_RATIO) + "_per_sc/"
    )
    create_calib_eval_splits(
        EPISODES_PATH, CALIB_SAVE_PATH, EVAL_SAVE_PATH, SPLIT_RATIO
    )
