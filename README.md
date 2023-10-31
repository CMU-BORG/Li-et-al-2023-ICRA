# FRR-Adaptive-Grasping

## Overview
This repository provides an implementation of the gantry system controlled by a Synthetic Nervous Systems (SNSs).

<p align="center">
    <img src="pick_and_place.gif" width="289.2" height="316.8" />
</p>

This environment builds upon the Pybullet simulator (https://pybullet.org/wordpress) and the [keras ncp](https://github.com/mlech26l/keras-ncp) by Mathias Lechner, Institute of Science and Technology Austria (IST Austria) ([Paper](https://www.nature.com/articles/s42256-020-00237-3?ref=https://coder.social)).

The current release provides following features:
* Implementation of the gantry system and simple objects in Pybullet.
* Implementation of supervised learning in the sensory layer.
* Support for using a hand-tuned SNS to control the gantry system accomplish pick-and-place tasks.

## Code Structure
The main environment for simulating a gantry system with picked objects is
in [GantrySimulation.py](envs/GantrySimulation.py). The neural network parameters are defined in [SNS_layer.py](controller/SNS_layer.py) and the supervised learning environment is defined in [train.py](controller/train.py)

```bash
controller
├── torchSNS
├── SNS_layer.py
└── train.py
envs
├── GantrySimulation.py
└── sinusoidgui_programmaticcheck.py
pick_and_place.py
```

## Verifying the Gantry Environment

You can run the [sinusoidgui_programmaticcheck.py](envs/sinusoidgui_programmaticcheck.py) script to verify your environment setup.

```bash
python -m envs.sinusoidgui_programmaticcheck
```

If it runs then you have set the gantry
environment correctly.

## Training a Model
To train the sensory layer, run [train.py](controller/train.py).

```bash
python -m controller.train
```

## Evaluating the Controller

To evaluate the designed SNS controller, run [pick_and_place.py](pick_and_place.py).

```bash
python pick_and_place.py
```

## Support

For questions about the code, please create an issue in the repository.