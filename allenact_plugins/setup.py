import glob
import os
from pathlib import Path

from setuptools import find_packages, setup


def parse_req_file(fname, initial=None):
    """Reads requires.txt file generated by setuptools and outputs a
    new/updated dict of extras as keys and corresponding lists of dependencies
    as values.

    The input file's contents are similar to a `ConfigParser` file, e.g.
    pkg_1
    pkg_2
    pkg_3

    [extras1]
    pkg_4
    pkg_5

    [extras2]
    pkg_6
    pkg_7
    """
    reqs = {} if initial is None else initial
    cline = None
    with open(fname, "r") as f:
        for line in f.readlines():
            line = line[:-1].strip()
            if len(line) == 0:
                continue
            if line[0] == "[":
                # Add new key for current extras (if missing in dict)
                cline = line[1:-1].strip()
                if cline not in reqs:
                    reqs[cline] = []
            else:
                # Only keep dependencies from extras
                if cline is not None:
                    reqs[cline].append(line)
    return reqs


def get_version(fname):
    """Reads PKG-INFO file generated by setuptools and extracts the Version
    number."""
    res = "UNK"
    with open(fname, "r") as f:
        for line in f.readlines():
            line = line[:-1]
            if line.startswith("Version:"):
                res = line.replace("Version:", "").strip()
                break
    if res in ["UNK", ""]:
        raise ValueError(f"Missing Version number in {fname}")
    return res


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(Path(__file__)))

    if not os.path.exists(
        os.path.join(base_dir, "allenact_plugins.egg-info/dependency_links.txt")
    ):
        # Build mode for sdist

        # Extra dependencies required for various plugins
        extras = {}
        for plugin_path in glob.glob(os.path.join(base_dir, "*_plugin")):
            plugin_name = os.path.basename(plugin_path).replace("_plugin", "")
            extra_reqs_path = os.path.join(plugin_path, "extra_requirements.txt")
            if os.path.exists(extra_reqs_path):
                with open(extra_reqs_path, "r") as f:
                    # Filter out non-PyPI dependencies
                    extras[plugin_name] = [
                        clean_dep
                        for clean_dep in (dep.strip() for dep in f.readlines())
                        if clean_dep != ""
                        and not clean_dep.startswith("#")
                        and "@ git+https://github.com/" not in clean_dep
                    ]
        extras["all"] = sum(extras.values(), [])

        os.chdir(os.path.join(base_dir, ".."))

        with open(".VERSION", "r") as f:
            __version__ = f.readline().strip()
    else:
        # Install mode from sdist
        __version__ = get_version(
            os.path.join(base_dir, "allenact_plugins.egg-info/PKG-INFO")
        )
        extras = parse_req_file(
            os.path.join(base_dir, "allenact_plugins.egg-info/requires.txt")
        )

    setup(
        name="allenact_plugins",
        version=__version__,
        description="Plugins for the AllenAct framework",
        long_description=(
            "A collection of plugins/extensions for use within the AllenAct framework."
        ),
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        keywords=["reinforcement learning", "embodied-AI", "AI", "RL", "SLAM"],
        url="https://github.com/allenai/allenact",
        author="Allen Institute for Artificial Intelligence",
        author_email="lucaw@allenai.org",
        license="MIT",
        packages=find_packages(include=["allenact_plugins", "allenact_plugins.*"]),
        install_requires=[
            "gym>=0.17.0,<0.18.0",
            "torch>=1.6.0",
            "torchvision>=0.7.0",
            "numpy>=1.19.1",
            "wheel>=0.36.2",
            f"allenact=={__version__}",
        ],
        setup_requires=["pytest-runner"],
        tests_require=["pytest", "pytest-cov"],
        extras_require=extras,
    )
