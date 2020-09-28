"""Entry point to training/validating/testing for a user given experiment
name."""
# Import these first to avoid
# conflict with other imports
import skimage
import scipy

import argparse
import importlib
import inspect
import os
import sys
from typing import Dict, Tuple

import gin
from setproctitle import setproctitle as ptitle

from core.algorithms.onpolicy_sync.runner import OnPolicyRunner
from core.base_abstractions.experiment_config import ExperimentConfig
from utils.system import get_logger


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = argparse.ArgumentParser(
        description="allenact", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment", type=str, help="experiment configuration file name",
    )

    parser.add_argument(
        "-et",
        "--extra_tag",
        type=str,
        default="",
        required=False,
        help="Add an extra tag to the experiment when trying out new ideas (will be used"
        "as a subdirectory of the tensorboard path so you will be able to"
        "search tensorboard logs using this extra tag).",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        type=str,
        default="experiment_output",
        help="experiment output folder",
    )

    parser.add_argument(
        "-s", "--seed", required=False, default=None, type=int, help="random seed",
    )
    parser.add_argument(
        "-b",
        "--experiment_base",
        required=False,
        default="experiments",
        type=str,
        help="experiment configuration base folder",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        default=None,
        type=str,
        help="optional checkpoint file name to resume training or test",
    )
    parser.add_argument(
        "-r",
        "--restart_pipeline",
        dest="restart_pipeline",
        action="store_true",
        required=False,
        help="for training, if checkpoint is specified, DO NOT continue the training pipeline from where"
        "training had previously ended. Instead restart the training pipeline from scratch but"
        "with the model weights from the checkpoint.",
    )
    parser.set_defaults(restart_pipeline=False)

    parser.add_argument(
        "-d",
        "--deterministic_cudnn",
        dest="deterministic_cudnn",
        action="store_true",
        required=False,
        help="sets CuDNN in deterministic mode",
    )
    parser.set_defaults(deterministic_cudnn=False)

    parser.add_argument(
        "-t",
        "--test_date",
        default=None,
        type=str,
        required=False,
        help="tests the experiment run on specified date (formatted as %%Y-%%m-%%d_%%H-%%M-%%S), assuming it was "
        "previously trained. If no checkpoint is specified, it will run on all checkpoints enabled by "
        "skip_checkpoints",
    )

    parser.add_argument(
        "-k",
        "--skip_checkpoints",
        required=False,
        default=0,
        type=int,
        help="optional number of skipped checkpoints between runs in test if no checkpoint specified",
    )

    parser.add_argument(
        "-m",
        "--max_sampler_processes_per_worker",
        required=False,
        default=None,
        type=int,
        help="maximal number of sampler processes to spawn for each worker",
    )

    parser.add_argument(
        "--gp", default=None, action="append", help="values to be used by gin-config.",
    )

    # To ensure agents always sample the mode action
    # Default to False when not specified
    parser.add_argument(
        "-e",
        "--deterministic_agents",
        dest="deterministic_agents",
        action="store_true",
        required=False,
        help="enable deterministic agents (i.e. always taking the mode action) during validation/testing",
    )

    parser.add_argument(
        "-vc",
        "--visual_corruption",
        default=None,
        type=str,
        required=False,
        help="What visual corruption to apply to the input egocentric observation",
    )

    parser.add_argument(
        "-vs",
        "--visual_severity",
        default=0,
        type=int,
        required=False,
        help="To what degree of severity should we apply the visual corruption",
    )

    parser.add_argument(
        "-trd",
        "--training_dataset",
        default=None,
        type=str,
        required=False,
        help="Specify the training dataset",
    )

    parser.add_argument(
        "-vld",
        "--validation_dataset",
        default=None,
        type=str,
        required=False,
        help="Specify the validation dataset",
    )

    parser.add_argument(
        "-tsd",
        "--test_dataset",
        default=None,
        type=str,
        required=False,
        help="Specify the testing dataset",
    )

    parser.add_argument(
        "-irc",
        "--random_crop",
        default=False,
        type=str,
        required=False,
        help="Specify if random crop is to be applied to the egocentric observations",
    )

    parser.add_argument(
        "-icj",
        "--color_jitter",
        default=False,
        type=str,
        required=False,
        help="Specify if random crop is to be applied to the egocentric observations",
    )

    parser.add_argument(
        "-tsg",
        "--test_gpus",
        default=None,
        # type=int,
        type=str,
        required=False,
        help="Specify the GPUs to run test on",
    )

    parser.add_argument(
        "-np",
        "--num_processes",
        default=None,
        type=int,
        required=False,
        help="Specify the number of processes per worker-node",
    )

    parser.set_defaults(deterministic_agents=False)

    return parser.parse_args()


def _config_source(args) -> Dict[str, Tuple[str, str]]:
    path = os.path.abspath(os.path.normpath(args.experiment_base))
    package = os.path.basename(path)

    module_path = "{}.{}".format(os.path.basename(path), args.experiment)
    modules = [module_path]
    res: Dict[str, Tuple[str, str]] = {}
    while len(modules) > 0:
        new_modules = []
        for module_path in modules:
            if module_path not in res:
                res[module_path] = (os.path.dirname(path), module_path)
                module = importlib.import_module(module_path, package=package)
                for m in inspect.getmembers(module, inspect.isclass):
                    new_module_path = m[1].__module__
                    if new_module_path.split(".")[0] == package:
                        new_modules.append(new_module_path)
        modules = new_modules
    return res


def load_config(args) -> Tuple[ExperimentConfig, Dict[str, Tuple[str, str]]]:
    path = os.path.abspath(os.path.normpath(args.experiment_base))
    sys.path.insert(0, os.path.dirname(path))
    importlib.invalidate_caches()
    module_path = ".{}".format(args.experiment)

    importlib.import_module(os.path.basename(path))
    module = importlib.import_module(module_path, package=os.path.basename(path))

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    gin.parse_config_files_and_bindings(None, args.gp)

    config = experiments[0]()
    sources = _config_source(args)
    return config, sources


def main():
    args = get_args()

    if args.visual_corruption is not None and args.visual_severity > 0:  # This works
        VISUAL_CORRUPTION = [args.visual_corruption.replace("_", " ")]
        VISUAL_SEVERITY = [args.visual_severity]
    else:
        VISUAL_CORRUPTION = args.visual_corruption
        VISUAL_SEVERITY = args.visual_severity

    TRAINING_DATASET_DIR = args.training_dataset
    VALIDATION_DATASET_DIR = args.validation_dataset
    TEST_DATASET_DIR = args.test_dataset

    RANDOM_CROP = args.random_crop
    COLOR_JITTER = args.color_jitter

    NUM_PROCESSES = None
    if args.num_processes is not None:
        NUM_PROCESSES = args.num_processes

    TESTING_GPUS = None
    if args.test_gpus is not None:
        TESTING_GPUS = [int(x) for x in args.test_gpus.split(",")]
        # TESTING_GPUS = [args.test_gpus]

    get_logger().info("Running with args {}".format(args))

    ptitle("Master: {}".format("Training" if args.test_date is None else "Testing"))

    cfg, srcs = load_config(args)

    if TESTING_GPUS is not None:
        cfg.TESTING_GPUS = TESTING_GPUS

    if NUM_PROCESSES is not None:
        cfg.NUM_PROCESSES = NUM_PROCESSES

    cfg.monkey_patch_sensor(
        VISUAL_CORRUPTION, VISUAL_SEVERITY, RANDOM_CROP, COLOR_JITTER
    )

    cfg.monkey_patch_datasets(
        TRAINING_DATASET_DIR, VALIDATION_DATASET_DIR, TEST_DATASET_DIR
    )

    if args.test_date is None:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="train",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
        ).start_train(
            checkpoint=args.checkpoint,
            restart_pipeline=args.restart_pipeline,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
        )
    else:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="test",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
        ).start_test(
            experiment_date=args.test_date,
            checkpoint=args.checkpoint,
            skip_checkpoints=args.skip_checkpoints,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
        )


if __name__ == "__main__":
    main()
