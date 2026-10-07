<p align="center">
  <h1 align="center">FASTER: Rethinking Real-Time Flow VLAs</h1>
  <p align="center">
    <strong><a href="https://innovator-zero.github.io/">Yuxiang Lu</a><sup>1,2</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://happinesslz.github.io/">Zhe Liu</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://www.linkedin.com/in/xianzhefan">Xianzhe Fan</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://huster-yzy.github.io/">Zhenya Yang</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://scholar.google.com/citations?user=aoqtBAsAAAAJ&hl=en">Jinghua Hou</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <br>
    <strong><a href="https://provencestar.github.io/">Junyi Li</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://kxding.github.io">Kaixin Ding</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://i.cs.hku.hk/~hszhao/">Hengshuang Zhao</a><sup>1</sup></strong>
  </p>

  <p align="center">
    <sup>1</sup> The University of Hong Kong
    <sup>2</sup> ACE Robotics
  </p>

  <p align="center">
  <img alt='neurips' src="https://img.shields.io/badge/NeurIPS-2026-blue.svg">
  <a href="https://arxiv.org/abs/2603.19199"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2603.19199-b31b1b.svg"></a>
  <a href="https://hku-sail.github.io/FASTER"><img alt='proj' src="https://img.shields.io/badge/Project Page-82B366.svg"></a>
  </p>
</p>

This repository provides the official implementation of FASTER, together with an unofficial implementation of [Training-time RTC](https://arxiv.org/abs/2512.05964), built on top of [openpi](https://github.com/Physical-Intelligence/openpi).


## TL;DR

Real-time reaction in VLAs is constrained not only by inference latency, but also by how action chunks are generated and executed. **FASTER** introduces a new paradigm for fast action sampling under asynchronous execution. By compressing the sampling process for immediate reaction into a single step, FASTER achieves **10x acceleration** over $\pi_{0.5}$ and X-VLA, enabling real-time responsiveness in highly dynamic tasks such as table tennis.

<img width="2960" height="836" alt="teaser" src="https://github.com/user-attachments/assets/121bf40a-20dc-41ff-bac0-c6d96edfb1c0" />

[Demo](https://github.com/user-attachments/assets/c16bd3fa-48ac-4d1b-aef9-4b0f4d839011)

## 📰 News

- **Sep 25 2026**: FASTER is accepted to NeurIPS 2026.
- **Apr 30 2026**: Code released.
- **Mar 19 2026**: Paper released.


## ✨ Abstract
Real-time execution is crucial for deploying Vision-Language-Action (VLA) models in the physical world. Existing asynchronous inference methods primarily optimize trajectory smoothness, but neglect the critical latency in reacting to environmental changes. By rethinking the notion of reaction in action chunking policies, this paper presents a systematic analysis of the factors governing reaction time. We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon. Moreover, we reveal that the standard practice of applying a constant schedule in flow-based VLAs can be inefficient and forces the system to complete all sampling steps before any movement can start, forming the bottleneck in reaction latency. To overcome this issue, we propose **F**ast **A**ction **S**ampling for Immedia**TE** **R**eaction (**FASTER**). By introducing a Horizon-Aware Schedule, FASTER adaptively prioritizes near-term actions during flow sampling, compressing the denoising of the immediate reaction by tenfold (*e.g.*, in $\pi_{0.5}$ and X-VLA) into a single step, while preserving the quality of long-horizon trajectory. Coupled with a streaming client-server pipeline, FASTER substantially reduces the effective reaction latency on real robots, especially when deployed on consumer-grade GPUs. Real-world experiments, including a highly dynamic table tennis task, prove that FASTER unlocks unprecedented real-time responsiveness for generalist policies, enabling rapid generation of accurate and smooth trajectories.

## ⚙️ Setup

> This repository is built on top of [openpi](https://github.com/Physical-Intelligence/openpi) (jax version). We strongly recommend getting familiar with the original openpi workflow before moving on.

### Requirements

FASTER follows the same requirements as openpi. To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimates assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

This repository has been tested with full fine-tuning $\pi_{0.5}$ on Ubuntu 22.04.

### Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules https://github.com/innovator-zero/FASTER.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Note: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot (dataset `v2.1`, commit `0cf8648`) as a dependency.

## 🚀 Usage

We provide a detailed guide on how to train and deploy FASTER on the AgileX Piper platform. If you have already adapted openpi to your robot platform, including data processing and mapping, running FASTER should be straightforward.

For simulation benchmarks, you can refer to [LIBERO](examples/libero/README.md) and [CALVIN](examples/calvin/README.md) guides.

### Policy Training

FASTER follows the standard fine-tuning pipeline in openpi. The main difference lies in the  **HAS (Horizon-Aware Schedule)** instead of the conventional constant schedule.

#### 1. Prepare Data

We use LeRobot dataset v2.1 as the data loader. If you use [AgileX Piper](https://global.agilex.ai/products/piper) robotic arms or [Cobot Magic](https://global.agilex.ai/products/cobot-magic) systems, we strongly recommend our [piper-aio](https://github.com/innovator-zero/piper-aio) toolkit. It covers the full robot-learning loop: hardware setup, teleoperated data collection, data replay, LeRobot dataset conversion, and policy inference.

#### 2. Define Config

The data-processing config for AgileX Piper data is provided in [`AgilexInputs`](src/openpi/policies/agilex_policy.py), [`AgilexOutputs`](src/openpi/policies/agilex_policy.py), and [`LeRobotAgilexDataConfig`](src/openpi/training/config.py).
We provide the following example training configs in [`config.py`](src/openpi/training/config.py):

- `pi05_agilex`: fine-tuning the $\pi_{0.5}$ model with the constant schedule.
- `pi05_rtc_agilex`: fine-tuning $\pi_{0.5}$ model with action conditioning strategy proposed by [Training-time RTC](https://arxiv.org/abs/2512.05964).
- `pi05_faster_agilex`: fine-tuning $\pi_{0.5}$ model with FASTER, using the following hyperparameters:
  - `max_delay`: maximum prefix length $d_\text{max}$, which simulates inference delay during training via action conditioning strategy. The default is `10`, supporting TTFA (Time to First Action) up to 333.3 ms on a 30 Hz robot.
  - `mix_prob`: mixing probability $p$ in the **mixed scheduling strategy**. Each action sample uses HAS with probability $p$ and retains the original constant schedule with probability $1 - p$. The default is `0.5`.
  - `alpha`: HAS hyperparameter controlling how hit times vary across action indices. The default is `0.6`.
  - `u0`: global timestep at which the first action is finalized, set to $(N-1)/N$ for $N$ inference sampling steps. Since $\pi_{0.5}$ uses 10 steps, we set `u0=0.9` by default.

#### 3. Launch Training

Remember to compute normalization statistics for the training data before launching training. These statistics can be shared across configs with a symlink (`ln -s`):

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_faster_agilex
```

Then you can launch training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_faster_agilex --exp-name=my_experiment
```

### Policy Deployment

Since the robot controller typically requires a different environment or a separate machine, we use a client-server interface for policy deployment. The server handles policy inference while the client runs the robot controller, with communication established through WebSocket. You can spin up the policy server on the same machine or on a LAN-connected workstation. For remote inference, prefer wired LAN to reduce latency and packet loss.

**On the client side, we use the inference scripts provided in our [piper-aio](https://github.com/innovator-zero/piper-aio/tree/main/inference) toolkit. If you use your own robot platform, you can adapt those scripts to your setup.**

#### Sync Inference

This is the standard practice for action-chunking policies and serves as a baseline for VLAs such as $\pi_{0.5}$. The robot executes one action chunk and requests the next chunk only after the current chunk has been consumed. During policy inference, the robot controller pauses and resumes only when the new actions arrive.

Start the policy server with the constant schedule:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_agilex --policy.dir=checkpoints/pi05_agilex/my_experiment/49999
```

Start the robot client in sync mode:

```bash
python inference/infer_sync.py # ... other arguments
```

#### Async Inference

The robot initiates inference for the next chunk before the current chunk is fully executed. Once the next inference request is triggered, the robot continues executing the ongoing actions in the current chunk. Before the final action is completed, the newly predicted chunk is expected to be available, enabling seamless execution without halting.

Start the policy server with HAS (match `alpha` and `u0` used in training):

```bash
uv run scripts/serve_policy.py \
    --use-custom-sample-kwargs \
    --infer-time-schedule=HAS \
    --alpha=0.6 \
    --u0=0.9 \
    policy:checkpoint \
    --policy.config=pi05_faster_agilex --policy.dir=checkpoints/pi05_faster_agilex/my_experiment/49999
```

Start the robot client in async (rtc) mode:

```bash
python inference/infer_async.py --mode=rtc --delay=4 --exec_horizon=25 # ... other arguments
```

Arguments:

- `--delay`: inference delay $d:= \lfloor \Delta t_\text{infer}/\Delta t_{\text{ctrl}}\rfloor$, determined by the inference latency $\Delta t_\text{infer}$ and control period $\Delta t_{\text{ctrl}}$. We recommend setting it one step larger to account for variation in inference time and transmission latency.
- `--exec_horizon`: execution horizon $s$ for the action chunk. The client executes only the first $s$ valid actions (excluding delayed ones), then triggers a new inference request. This value should be larger than or equal to $d$.

#### Streaming Inference

This mode is built on our **Streaming Client-Server Interface** and achieves the lowest TTFA (Time to First Action). Early actions can be dispatched to the robot client instantly upon completion. While the robot executes these initial movements, the policy server continues refining subsequent actions in parallel and progressively replenishes the client’s action buffer. It is designed for tasks that require fast reactions, such as playing table tennis.

Start the policy server in streaming mode:

```bash
uv run scripts/serve_policy.py \
    --use-custom-sample-kwargs \
    --infer-time-schedule=HAS \
    --alpha=0.6 \
    --u0=0.9 \
    --streaming \
    --early-stop-actions=4 \
    policy:checkpoint \
    --policy.config=pi05_faster_agilex --policy.dir=checkpoints/pi05_faster_agilex/my_experiment/49999
```

Arguments:

- `--early-stop-actions`: if the policy has already generated this number of valid actions and dispatched them to the client, the remaining action-sampling iterations stop early. This value should be larger than or equal to `exec_horizon`, because the remaining noisy actions will not be executed by the robot and do not need to be generated. The policy server can then prepare for the next inference request.

Start the robot client in streaming mode:

```bash
python inference/infer_async.py --mode=rtc --delay=3 --exec_horizon=4 --streaming # ... other arguments
```

Arguments:

- `--delay`: can be smaller than in async mode because inference latency, measured by TTFA, is reduced.
- `--exec_horizon`: a smaller $s$ can help highly dynamic tasks by increasing inference frequency and tightening the inference-execution loop. For everyday tasks such as pick-and-place or folding towels, a small $s$ is usually unnecessary and may increase motion jitter.

## 📖 Citation

```
 @article{lu2026faster,
  title={FASTER: Rethinking Real-Time Flow VLAs}, 
  author={Yuxiang Lu and Zhe Liu and Xianzhe Fan and Zhenya Yang and Jinghua Hou and Junyi Li and Kaixin Ding and Hengshuang Zhao},
  year={2026},
  journal={arXiv preprint arXiv:2603.19199}
}
```

## 🙏 Acknowledgements

We thank the following repositories for their references and prior work:

- [openpi](https://github.com/Physical-Intelligence/openpi)
- [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World)
