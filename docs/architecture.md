# Architecture and execution path

This document describes the model structure retained from the upstream RDT
implementation and how it is connected to the RoboTwin adaptation in this
repository. **The RDT backbone, multimodal conditioning design, and diffusion
formulation are upstream work; they are documented here for project
completeness rather than presented as original model contributions.**

## 1. RDT architecture overview

RDT-1B is a diffusion policy built around a Diffusion Transformer (DiT). It
predicts a sequence of future robot actions conditioned on language, visual
observations, proprioception, diffusion timestep, and control frequency.

```text
Language instruction ── T5 ───────────────┐
                                            │
Head / left wrist / right wrist RGB        │
        × 2-frame history ── SigLIP ───────┼─ Condition adapters
                                            │       ↓
Robot proprioception + validity mask ──────┘   shared hidden space
                                                    ↓
                                  ┌────────────────────────────┐
Diffusion timestep ──────────────►│                            │
Control frequency ───────────────►│     RDT / DiT backbone     │
Current state ───────────────────►│   28 blocks, 2048 hidden   │
Noisy future action sequence ────►│                            │
                                  └────────────────────────────┘
                                                    ↓
                                      denoised action chunk
                                                    ↓
                                      RoboTwin robot control
```

The retained `rdt/configs/base.yaml` uses:

| Component | Configuration |
| --- | --- |
| Historical visual context | 2 frames |
| Cameras | 3: head + left wrist + right wrist |
| Unified state/action space | 128 dimensions |
| Future action horizon | 64 steps |
| Language token dimension | 4096 |
| Image token dimension | 1152 |
| RDT hidden size | 2048 |
| Transformer depth | 28 blocks |
| Attention heads | 32 |

The 128-dimensional state/action representation is RDT's unified robot space.
Embodiment-specific values occupy their assigned dimensions and a validity mask
marks which dimensions are meaningful for the current robot. RoboTwin's
bimanual joint representation is therefore adapted into this common interface
before entering the policy, while only the valid robot dimensions are unpacked
again at deployment.

## 2. Multimodal conditioning

The model does not concatenate raw RGB, text, and robot vectors directly.
Each modality is first encoded in its own feature space and then projected to
the RDT hidden dimension:

```text
Instruction ── T5 ───────── 4096-D tokens ── language adapter ─┐
                                                                │
RGB history ── SigLIP ───── 1152-D tokens ── image adapter ─────┼─ 2048-D
                                                                │
State + validity mask ─────── 128-D unified space ─ state adapter┘
```

`RDTRunner` uses MLP-GELU adapters (`mlp2x_gelu` for language/image and
`mlp3x_gelu` for state/action) to place all conditions in the shared Transformer
hidden space. The RDT sequence additionally contains embeddings for the
**diffusion timestep** and **control frequency**, so the model is conditioned on
both denoising progress and the physical time scale of the robot trajectory.

Inside `rdt/models/rdt/model.py`, RDT blocks process the state/action token
sequence while alternating cross-modal conditions between language and image
tokens. Positional embeddings distinguish timestep, control frequency, current
state, and future action positions. The final layer retains only the future
action tokens.

## 3. Diffusion action generation

RDT formulates robot control as conditional trajectory denoising rather than
single-step action regression.

### Training

For a ground-truth future action chunk, `RDTRunner.compute_loss()` samples a
random diffusion timestep and adds DDPM noise to the clean actions:

```text
clean action chunk
      ↓ add noise at timestep t
noisy action chunk
      + current state
      + language condition
      + visual condition
      + control frequency
      ↓
RDT
      ↓
predicted clean action
      ↓
MSE supervision
```

The retained configuration uses a 1000-step DDPM training schedule with
`squaredcos_cap_v2` beta scheduling and `prediction_type: sample`, so the model
is supervised against the clean action sample.

### Inference

Inference starts from Gaussian action noise and iteratively denoises it with a
DPM-Solver scheduler. The retained configuration uses 5 inference denoising
steps to produce the complete future chunk:

```text
Gaussian noise [64 × action_dim]
        ↓
DPM-Solver + RDT condition
        ↓
progressive denoising
        ↓
64-step action chunk
```

This is the main distinction from a one-step behavior-cloning policy: one model
call represents a temporally structured future action sequence instead of only
the next instantaneous command.

## 4. RoboTwin adaptation boundary

The model architecture above remains upstream RDT. This repository's project
work begins at the **adaptation boundary** around it:

```text
RoboTwin expert data
        ↓
HDF5 temporal / language / frequency alignment
        ↓
128-D RDT state/action representation
        ↓
Released RDT-1B checkpoint
        ↓
Task-specific fine-tuning
        ↓
XPolicyLab policy server
        ↓
RoboTwin closed-loop evaluation
```

At deployment, the RoboTwin process sends three-camera RGB observations,
bimanual joint state, instruction, and control frequency through the XPolicyLab
protocol. `policy/rdt_1b/model.py` builds the two-frame image history, encodes
language with T5 and vision with SigLIP, calls `RDTRunner.predict_action()`, and
maps the valid dimensions of the predicted action chunk back to RoboTwin's
bimanual control representation.

## 5. RoboTwin task side

Each representative RoboTwin task provides three pieces required by the
end-to-end experiment:

- `load_actors()`: scene creation and randomized task state;
- `play_once()`: programmatic expert motion using grasp/place/move primitives;
- `check_success()`: task-specific terminal success predicate.

Collection first finds seeds for which planning and `check_success()` pass,
saves the expert joint paths, then replays them with observation capture.
Evaluation replaces expert replay with action chunks returned by the RDT policy
server and uses the same task success predicate.

## Code map

| Function | Source |
| --- | --- |
| RDT Transformer | `rdt/models/rdt/model.py` |
| Attention / DiT blocks | `rdt/models/rdt/blocks.py` |
| Condition adapters + diffusion | `rdt/models/rdt_runner.py` |
| T5 / SigLIP encoders | `rdt/models/multimodal_encoder/` |
| RDT model configuration | `rdt/configs/base.yaml` |
| RoboTwin/RDT deployment adapter | `policy/rdt_1b/model.py` |
| Expert data collection | `robotwin/scripts/collect_data.py` |
| Closed-loop evaluation | `robotwin/scripts/eval_policy_xpolicylab.py` |
