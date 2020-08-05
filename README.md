# Embodied-AI

`embodied-ai` is a framework designed for research in embodied AI with a focus on modularity and flexibility. 
Due to requirements of double blind review, certain components of our framework have been redacted (e.g.
our contribution guidelines and download links).
 
## Table of contents

1. [Why embodied-ai?](#why)
1. [Installation](#installation)
1. [Contributions](#contributions)
1. [Citiation](#citation)

## Why `embodied-ai`?

There are an increasingly [large collection](https://winderresearch.com/a-comparison-of-reinforcement-learning-frameworks-dopamine-rllib-keras-rl-coach-trfl-tensorforce-coach-and-more/) of deep reinforcement learning packages and so it is natural to question why we introduce another framework reproducing many of the same algorithms and ideas. After performing of survey of existing frameworks we could not find a package delivering all of the following features, each of which we considered critical.

## Table of contents

1. *Decoupled tasks and environments*: In embodied AI research it is important to be able to define many tasks for a single environment; for instance, the [AI2-THOR](https://ai2thor.allenai.org/) environment has been used with tasks such as  
    * [semantic/object navigation](https://arxiv.org/abs/1810.06543),
    * [interactive question answering](https://arxiv.org/abs/1712.03316),
    * [multi-agent furniture lifting](https://prior.allenai.org/projects/two-body-problem), and
    * [adversarial hide-and-seek](https://arxiv.org/abs/1912.08195). 
We have designed `embodied-ai` to easily support a wide variety of tasks designed for individual environments.
1. *Support for several environments*: We support different environments used for Embodied AI research such as AI2-THOR, Habitat and MiniGrid. We have made it easy to incorporate new environments.
1. *Different input modalities*: The framework supports a variety of input modalities such as RGB images, depth, language and GPS readings. 
1. *Various training pipelines*: The framework includes not only various training algorithms already implemented, but also a mechanism to integrate different types of algorithms (e.g., imitation learning followed by reinforcement learning). 
1. *First-class PyTorch support*: While many well-developed libraries exist for reinforcement learning in 
   tensorflow, we are one of a few to target pytorch.
1. *Configuration as code*: In `embodied-ai` experiments are defined using python classes, so knowing how to extend an abstract python class we can define an experiment.
1. *Type checking and documentation*: We have put significant effort into providing extensive documentation and type annotations throughout our codebase.
1. *Tutorials*: We provide step-by-step tutorials for different tasks and environments so the new users can make faster progress in implementing and prototyping new ideas.
 
## Installation

Begin by cloning this repository to your local machine and moving into the top-level directory
REDACTED FOR DOUBLE-BLIND REVIEW
This library has been tested **only in python 3.6**, the following assumes you have a working version of **python 3.6** installed locally. In order to install requirements we recommend using [`pipenv`](https://pipenv.kennethreitz.org/en/latest/) but also include instructions if you would prefer to install things directly using `pip`.

### Installing requirements with `pipenv` (*recommended*)

If you have already installed [`pipenv`](https://pipenv.kennethreitz.org/en/latest/), you may run the following to install all requirements.
```bash
pipenv install --skip-lock --dev
```

### Installing requirements with `pip`

Note: *do not* run the following if you have already installed requirements with `pipenv`
as above. If you prefer using `pip`, you may install all requirements as follows
```bash
pip install -r requirements.txt
```
Depending on your machine configuration, you may need to use `pip3` instead of `pip` in the
above.

### Run your first experiment

You are now ready to [run your first experiment](./overview/running-your-first-experiment.md).

## Acknowledgments

This work builds upon the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) library of Ilya Kostrikov and uses some data structures from FAIR's [habitat-api](https://github.com/facebookresearch/habitat-api).

## Contributions

REDACTED FOR DOUBLE-BLIND REVIEW
 
## Citation

If you use this work, please cite:

REDACTED FOR DOUBLE-BLIND REVIEW