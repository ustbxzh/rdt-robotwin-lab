# RDT × RoboTwin Lab

An end-to-end **bimanual VLA training and simulation framework** built around released RDT-1B and RoboTwin. The project focuses on the engineering path around the upstream model: expert-data generation, multimodal alignment, multi-task adaptation, policy serving, and closed-loop evaluation.

The main experiment uses **one ARX-X5 embodiment, four manipulation tasks, one shared RDT policy, and two evaluation domains**:

```text
4 RoboTwin Tasks
      ↓
Expert Planning + Successful Trajectory Filtering
      ↓
Multiview RGB / State / Action / Language HDF5
      ↓
Temporal + Episode-language + Control-frequency Alignment
      ↓
Mixed Multi-task Dataset
      ↓
ONE RDT-1B Fine-tuning Run
      ↓
ONE Shared Checkpoint
      ↓
4 Tasks × {Clean, Randomized} Closed-loop Evaluation
```

## What I built

The RDT backbone follows the upstream implementation. My work is concentrated on making it usable as a complete RoboTwin training/evaluation pipeline:

- **Expert-data pipeline** — generate programmatic RoboTwin demonstrations, retain only trajectories passing planning and task-success gates, replay successful seeds, and serialize three-view RGB, bimanual state/action, and episode instructions.
- **Multimodal alignment** — enforce `State(t) → Action(t+1)`, keep the 15 Hz control semantics consistent, and bind each HDF5 episode to its own precomputed T5 embedding.
- **Shared multi-task adaptation** — combine demonstrations from four tasks on the same ARX-X5 robot into one HDF5 training tree and fine-tune one released RDT-1B checkpoint with DeepSpeed ZeRO-2, BF16 and gradient accumulation.
- **Online policy deployment** — connect RoboTwin to RDT through XPolicyLab, maintain two-frame observation history, execute 64-step action chunks, and expose checkpoint/input/same-seed audit hooks.
- **Unified evaluation** — run the same checkpoint on every task under shared Clean and Randomized profiles and use the task's own `check_success()` predicate as the closed-loop metric.

> **Scope.** Foundation-model pretraining is retained for source inspection but is not rerun here. Raw HDF5 datasets, model weights, checkpoints and RoboTwin assets remain external to Git. Evaluation is simulation-only.

## Demo

The four-task suite progresses from short-horizon single-object control to longer bimanual and multi-object manipulation.

| Difficulty | Task | Policy rollout | Expert demonstration |
| --- | --- | --- | --- |
| 1 | **Adjust bottle and hold upright** (`adjust_bottle`) | <img src="assets/task1.gif" alt="RDT policy rollout adjusting a bottle" width="360"> | <img src="assets/task1_ex.gif" alt="Expert adjust bottle demonstration" width="360"> |
| 2 | **Bimanual pot lift** (`lift_pot`) | <img src="assets/task2.gif" alt="RDT policy rollout lifting a pot" width="360"> | <img src="assets/task2_ex.gif" alt="Expert pot lift demonstration" width="360"> |
| 3 | **Arm-to-arm block handover** (`handover_block`) | <img src="assets/task3.gif" alt="RDT policy rollout handing over a block" width="360"> | <img src="assets/task3_ex.gif" alt="Expert handover demonstration" width="360"> |
| 4 | **Rank blocks by size** (`blocks_ranking_size`) | <img src="assets/task4.gif" alt="RDT policy rollout ranking blocks" width="360"> | <img src="assets/task4_ex.gif" alt="Expert block-ranking demonstration" width="360"> |

The corresponding task definitions are retained in `robotwin/envs/` so the expert motion, scene construction and `check_success()` criteria can be inspected directly.

## RDT architecture overview

RDT-1B is included so the full training/deployment path can be followed from source. The architecture itself is upstream RDT rather than a new model contribution of this repository.

```text
Language ── T5 ────────────────┐
                               │
3-camera RGB × 2-frame history │
        └── SigLIP ────────────┼── Modality Adapters ──┐
                               │                       │
Robot state + validity mask ───┘                       ▼
                                               RDT / DiT Backbone
Diffusion timestep ────────────────────────────► 28 Transformer blocks
Control frequency ─────────────────────────────► 2048 hidden / 32 heads
                                                       │
                                                       ▼
                                                64-step Action Chunk
```

The retained configuration uses a 128-D unified state/action space, three cameras, two-frame visual history, and a 64-step prediction horizon. Training uses DDPM forward noise on action chunks; inference applies DPM-Solver denoising. See [`docs/architecture.md`](docs/architecture.md).

## Robot data pipeline

### 1. Expert demonstration generation

The training data are generated from RoboTwin tasks rather than downloaded as a finished offline dataset. Each task implements scene construction, `play_once()` expert motion and `check_success()`.

```text
Task Definition
      ↓
Scene Initialization
      ↓
Expert Motion Primitives
      ↓
plan_success && check_success()
      ↓
Successful Seed + Joint Path
      ↓
Replay and Multimodal Capture
      ↓
Episode HDF5
```

The shared collection profile is `robotwin/env_cfg/task_config/train_clean.yml`: **ARX-X5, 50 target episodes/task, three RGB views, joint state/action, 15 Hz**.

Collect all four tasks:

```bash
bash scripts/collect_expert_suite.sh 0
```

The resulting tree is expected under:

```text
robotwin/data/train_clean/
├── adjust_bottle/arx_x5/data/*.hdf5
├── lift_pot/arx_x5/data/*.hdf5
├── handover_block/arx_x5/data/*.hdf5
└── blocks_ranking_size/arx_x5/data/*.hdf5
```

### 2. Temporal and semantic alignment

For a joint trajectory `q[0..N-1]`:

```text
RGB    = RGB[:-1]
state  = q[:-1]
action = q[1:]
```

so the training target follows `RGB(t), State(t) → Action(t+1)`. Language is aligned at episode granularity:

```text
episode_0000007.hdf5  ↔  episode_0000007.pt
```

`policy/rdt_1b/encode_language.py` precomputes T5 embeddings and mirrors the HDF5 relative path, keeping vision, proprioception, action and instruction bound to the same demonstration.

Prepare the complete four-task HDF5 tree and language embeddings:

```bash
bash scripts/prepare_multitask_data.sh 0
```

Dataset statistics are then recomputed from the mixed training tree:

```bash
bash scripts/compute_dataset_stats.sh
```

Exact sequence conventions are documented in [`docs/dataset_alignment.md`](docs/dataset_alignment.md).

## Shared multi-task experiment

Runtime authority for the main experiment lives in [`experiments/rdt_robotwin_multitask/`](experiments/rdt_robotwin_multitask/):

```text
experiments/rdt_robotwin_multitask/
├── dataset.yaml
├── train.env
├── deploy.yaml
├── tasks/
│   ├── adjust_bottle.yaml
│   ├── lift_pot.yaml
│   ├── handover_block.yaml
│   └── blocks_ranking_size.yaml
├── eval/
│   ├── clean.yaml
│   └── randomized.yaml
├── stats/
└── results/
```

The key distinction is that these are **not four independently trained models**. The task datasets are mixed into one training source and one shared RDT checkpoint is evaluated across the full suite.

### Training

The local adaptation schedule in `train.env` uses a 10k-step run with checkpoints every 2k steps, BF16, DeepSpeed ZeRO-2 and gradient accumulation. This is the local constrained experiment schedule rather than the upstream RDT recommended foundation-scale recipe.

```bash
bash scripts/train_multitask.sh 0 0
#                              GPU seed
```

### Evaluation

The same checkpoint is evaluated under two common RoboTwin profiles:

- **Clean** — training-like environment, seen instruction split.
- **Randomized** — randomized background/table height/lighting/clutter, unseen instruction split.

This produces an explicit `4 tasks × 2 domains = 8 conditions` matrix.

```bash
# all four tasks, both domains
bash scripts/eval_multitask_suite.sh rdt_robotwin_multitask all 0 0 0

# or one domain only
bash scripts/eval_multitask_suite.sh rdt_robotwin_multitask clean 0 0 0
```

RoboTwin observations are sent to the XPolicyLab RDT server; the model returns a 64-step action chunk, which is executed in closed loop until the task succeeds or terminates.

## Model assets

Weights are not committed. The project uses the official released components:

| Component | Official source | Expected path |
| --- | --- | --- |
| RDT-1B | [robotics-diffusion-transformer/rdt-1b](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | `policy/rdt_1b/weights/RDT/rdt-1b` |
| T5-v1.1-XXL | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `policy/rdt_1b/weights/RDT/t5-v1_1-xxl` |
| SigLIP SO400M Patch14-384 | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | `policy/rdt_1b/weights/RDT/siglip-so400m-patch14-384` |

Check local assets or explicitly download with the Hugging Face CLI:

```bash
bash scripts/setup_models.sh
bash scripts/setup_models.sh download
```

More details: [`docs/model_assets.md`](docs/model_assets.md).

## Results

Final quantitative results will be added under `experiments/rdt_robotwin_multitask/results/` and summarized here. The planned report keeps local rollout metrics and upstream RoboTwin/RDT reference numbers separate.

| Task | Clean | Randomized | Robustness retention |
| --- | ---: | ---: | ---: |
| adjust_bottle | TBD | TBD | TBD |
| lift_pot | TBD | TBD | TBD |
| handover_block | TBD | TBD | TBD |
| blocks_ranking_size | TBD | TBD | TBD |

Planned analysis includes per-task success rate, mean Clean/Randomized success, robustness retention, task-complexity comparison and qualitative failure cases.

## Repository layout

```text
rdt-robotwin-lab/
├── rdt/                    # upstream RDT core retained for source-level completeness
├── robotwin/               # task definitions, expert collection and closed-loop evaluation
├── policy/
│   ├── rdt_1b/             # RoboTwin ↔ RDT data/training/inference adaptation
│   └── xpolicylab/         # policy client/server transport
├── experiments/
│   └── rdt_robotwin_multitask/  # one shared-policy experiment
├── scripts/                # project-level one-command entry points
├── docs/                   # architecture, data alignment and model assets
├── environment/            # software/CUDA records
└── patches/                # retained modification history
```

## End-to-end quick start

```bash
# 1. Prepare RDT / T5 / SigLIP
bash scripts/setup_models.sh

# 2. Collect expert data for all four tasks
bash scripts/collect_expert_suite.sh 0

# 3. Link the mixed HDF5 tree and precompute episode language embeddings
bash scripts/prepare_multitask_data.sh 0

# 4. Compute statistics from the mixed dataset
bash scripts/compute_dataset_stats.sh

# 5. Fine-tune one shared RDT policy
bash scripts/train_multitask.sh 0 0

# 6. Evaluate the same checkpoint on 4 tasks × 2 domains
bash scripts/eval_multitask_suite.sh rdt_robotwin_multitask all 0 0 0
```

RoboTwin assets can be linked under `robotwin/assets` or supplied with `ROBOTWIN_ASSETS_ROOT`.

## Attribution

RDT and RoboTwin sources are MIT licensed; XPolicyLab is Apache-2.0. See `THIRD_PARTY_NOTICES.md` and the component license files for upstream revisions and modification scope.
