import gzip
import json
import os
import shutil
from typing import Sequence

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler


def create_debug_dataset_from_train_dataset(
    scene: str,
    target: str,
    episodes_subset: Sequence[int],
    train_dataset_path: str,
    base_debug_output_path: str,
):
    downloaded_episodes = os.path.join(
        train_dataset_path, "episodes", scene + ".json.gz"
    )

    assert os.path.exists(downloaded_episodes), (
        "'{}' doesn't seem to exist or is empty. Make sure you've downloaded to download the appropriate"
        " training dataset with"
        " datasets/download_navigation_datasets.sh".format(downloaded_episodes)
    )

    # episodes
    episodes = ObjectNavDatasetTaskSampler.load_dataset(
        scene=scene, base_directory=os.path.join(train_dataset_path, "episodes")
    )

    target_episodes = [ep for ep in episodes if ep["object_type"] == target]

    ids = ["{}_{}_{}".format(scene, target, epit) for epit in episodes_subset]
    debug_episodes = [ep for ep in target_episodes if ep["id"] in ids]

    # sort by episode_ids
    debug_episodes = [
        idep[1]
        for idep in sorted(
            [(int(ep["id"].split("_")[-1]), ep) for ep in debug_episodes],
            key=lambda x: x[0],
        )
    ]
    assert len(debug_episodes) == len(episodes_subset)

    episodes_dir = os.path.join(base_debug_output_path, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)
    episodes_file = os.path.join(episodes_dir, scene + ".json.gz")

    json_str = json.dumps(debug_episodes)
    json_bytes = json_str.encode("utf-8")
    with gzip.GzipFile(episodes_file, "w") as fout:
        fout.write(json_bytes)
    assert os.path.exists(episodes_file)


if __name__ == "__main__":
    SCENE = "FloorPlan_Train1_1"
    TARGET = "Television"
    EPISODES = [0, 7, 11, 12]
    BASE_OUT = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets", "robothor-objectnav", "debug"
    )

    create_debug_dataset_from_train_dataset(
        scene=SCENE,
        target=TARGET,
        episodes_subset=EPISODES,
        train_dataset_path=os.path.join(
            ABS_PATH_OF_TOP_LEVEL_DIR, "datasets", "robothor-objectnav", "train"
        ),
        base_debug_output_path=BASE_OUT,
    )
