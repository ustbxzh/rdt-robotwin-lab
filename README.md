# RDT × RoboTwin Lab

A source-first, end-to-end view of RDT-1B adaptation to RoboTwin:

```text
Task Construction
→ Expert Planning & Trajectory Filtering
→ Multiview RGB / State / Action Collection
→ Temporal & Language Alignment
→ Normalization / Control-Frequency Adaptation
→ RDT Fine-tuning
→ Policy Deployment
→ RoboTwin Closed-loop Evaluation
```

This is not a patch-only archive and does not vendor the full upstream
repositories. It retains the algorithm and the exact integration path needed
to understand the experiment. No weights, checkpoints, datasets, or RoboTwin
assets are committed.

## What is here

- `rdt/`: the RDT Transformer, diffusion runner, multimodal encoders, official
  pretraining pipeline, and training loop.
- `robotwin/`: the minimal simulator/task chain for `adjust_bottle` and
  `pick_dual_bottles`, including expert motion, collection, conversion, and
  success evaluation.
- `policy/rdt_1b/`: RoboTwin observation/state adaptation, episode language
  encoding, fine-tuning, checkpoint loading, and inference.
- `policy/xpolicylab/`: the minimal WebSocket/TCP policy runtime used by the
  actual deployment path.
- `patches/`: local modification history retained for auditability.
- `experiments/adjust_bottle/`: the dataset statistics used by the local
  fine-tuning experiment.

The staged files under `configs/` are retained experiment/audit snapshots.
Runtime authority lives in `rdt/configs`, `robotwin/env_cfg`, and
`policy/rdt_1b/deploy.yml`.

## Experiment scope

The official RDT pretraining entry is retained for inspection, but this
project did **not** rerun RDT-1B foundation-model pretraining. The actual run
starts from a released RDT-1B checkpoint and fine-tunes on RoboTwin expert
demonstrations.

RoboTwin data and RDT conditioning are aligned at episode granularity:

```text
episode_0000000.hdf5 → episode_0000000.pt
episode_0000001.hdf5 → episode_0000001.pt
```

The encoder selects the episode instruction stored in the HDF5 file (or the
matching candidate using the episode index), while the training loader derives
the `.pt` path from the HDF5 relative path. The experiment uses 15 Hz control.

## RDT model architecture

The model architecture retained in this repository follows the **upstream
RDT-1B design** and is included to make the training/deployment path
self-contained; it is not presented as an original architecture contribution
of this project.

RDT-1B uses a **Diffusion Transformer (DiT)** policy that predicts a future
action sequence from multimodal robot observations:

```text
Language ── T5 ────────────────┐
                               │
3-camera RGB × 2-frame history │
        └── SigLIP ────────────┼── modality adapters ──┐
                               │                       │
Robot state + validity mask ───┘                       ▼
                                               RDT / DiT backbone
Diffusion timestep ────────────────────────────► 28 Transformer blocks
Control frequency ─────────────────────────────► 2048 hidden / 32 heads
                                                       │
                                                       ▼
                                                64-step Action Chunk
```

The retained configuration uses a **128-D unified state/action space**, 3
cameras, 2-frame visual history, and a 64-step prediction horizon. T5 language
tokens, SigLIP image tokens, and robot state/action tokens are projected into
the common RDT hidden space through modality-specific adapters. During
training, DDPM noise is added to the ground-truth action chunk and RDT learns to
recover the clean sequence; inference starts from Gaussian action noise and
uses DPM-Solver iterative denoising to produce the final action chunk.

The project-specific work is concentrated around this upstream model: RoboTwin
data generation/alignment, state/action adaptation, task-level fine-tuning,
policy serving, and closed-loop evaluation. See
[`docs/architecture.md`](docs/architecture.md) for the module-level structure,
dimensions, and train/inference flow.

## Data pipeline

The data path is treated as a robot-data engineering problem rather than a
file-format conversion. Task-specific demonstrations are generated in
**RoboTwin/SAPIEN**, validated at trajectory level, serialized as multimodal
HDF5 episodes, and then aligned to the temporal, language, control-frequency,
and numerical conventions expected by RDT.

```text
RoboTwin/SAPIEN Task
        ↓
Expert Planning + Success Gate
        ↓
Seed Replay + Multimodal Capture
        ↓
Episode HDF5
        ↓
Temporal / Language / Frequency Alignment
        ↓
Normalization + Dataset Validation
        ↓
RDT Fine-tuning Dataset
```

### 1. Simulation expert data and quality gate

Instead of VR/teleoperation, this project uses RoboTwin's **programmatic expert
motion primitives** as the demonstration source. SAPIEN provides the physics
simulation, while RoboTwin supplies task definitions, embodiment/camera
configuration, deterministic seeds, motion planning, and task-specific success
predicates.

`robotwin/scripts/collect_data.py` accepts an episode only when

```text
plan_success && check_success()
```

holds. The successful seed and joint path are then replayed for observation
capture and checked again before serialization. This is the primary data
cleaning strategy in the project: **trajectory-level validity filtering** rather
than applying artificial sensor-denoising rules to simulator data.

Representative task definitions live in `robotwin/envs/adjust_bottle.py` and
`robotwin/envs/pick_dual_bottles.py`.

### 2. Multimodal schema and control-time alignment

Each accepted episode keeps all policy conditions inside one HDF5 boundary:

```text
episode_xxxxxxx.hdf5
├── /vision/{cam_head, cam_left_wrist, cam_right_wrist}/colors
├── /state/{left/right arm, left/right end-effector}
├── /action/{left/right arm, left/right end-effector}
├── /instructions
└── /additional_info/frequency
```

The head camera provides global scene context and the two wrist cameras retain
local manipulation evidence around grasp/contact/occlusion. Because these
signals are sampled from the same simulator step, the project does not invent a
hardware-style 30 Hz camera / 100 Hz encoder synchronization problem; the key
alignment issue is the **control semantics between observation and next action**.

For a joint trajectory `q[0..N-1]`, the training sequence is constructed as

```text
RGB    = RGB[:-1]
state  = q[:-1]
action = q[1:]
```

so each sample follows `RGB(t), State(t) → Action(t+1)` and all modalities share
an `N-1` horizon. The demonstration and policy paths also keep the **15 Hz**
control setting consistent, since control frequency determines the physical
time represented by an action step and an action chunk.

See `docs/dataset_alignment.md` for the exact sequence convention.

### 3. Episode-level semantic alignment

Language is aligned at demonstration granularity instead of assigning one
embedding to an entire task. `policy/rdt_1b/encode_language.py` encodes the
instruction of each episode with T5 and mirrors its relative HDF5 path:

```text
data/.../episode_0000007.hdf5
             ↕
lang_embeds/.../episode_0000007.pt
```

The HDF5 loader derives the corresponding language path from the episode path,
keeping **vision + proprioception + action + language** semantically bound to the
same demonstration. This prevents task-level language labels from silently
mismatching an individual trajectory.

### 4. Distribution adaptation and robustness boundary

RoboTwin state/action values are normalized with task-specific statistics before
fine-tuning; the `adjust_bottle` statistics used by the local experiment are
retained in `experiments/adjust_bottle/dataset_stat.json` and injected into the
RDT HDF5 loader.

The task configuration also makes the data distribution explicit. The retained
`demo_clean.yml` collects 50 episodes with background/light/camera/table domain
randomization disabled, establishing a controlled training distribution.
RoboTwin exposes these randomization axes separately for robustness evaluation,
which allows later Clean-vs-Randomized analysis without confusing data cleaning
with domain randomization.

This repository is simulation-only and does not claim a real-robot or Sim2Real
result. Foundation-model pretraining data belongs to the upstream RDT pipeline;
this project's data work starts from RoboTwin task-specific demonstrations and
their adaptation to RDT fine-tuning.

### Data-engineering capabilities represented here

| Engineering capability | Concrete implementation |
| --- | --- |
| Expert data generation | RoboTwin/SAPIEN motion primitives + deterministic seed replay |
| Data quality governance | Planning/success gate + post-replay episode validation |
| Multimodal data modeling | Three RGB views + bimanual state/action + instruction |
| Temporal/control alignment | `State(t) → Action(t+1)` + consistent 15 Hz semantics |
| Cross-modal alignment | One T5 embedding bound to each HDF5 episode |
| Distribution adaptation | Task-specific state/action statistics + explicit clean/randomized boundary |

## Fine-tuning and evaluation

The downstream stage is designed as **low-shot task adaptation** rather than a
second foundation-model training run. A released RDT-1B checkpoint provides the
cross-task robot prior; RoboTwin fine-tuning isolates the task-specific HDF5,
language embeddings, control frequency, and normalization statistics. The
local task setup uses about **50 expert demonstrations** and a short downstream
schedule around **10k optimization steps**.

```text
Released RDT-1B
      ↓
~50 task demonstrations
      ↓
Task-specific Fine-tuning
      ↓
Policy Server
      ↓
RoboTwin Closed-loop Rollout
      ↓
check_success()
```

### 1. Training engineering

The training launcher in `policy/rdt_1b/train.sh` turns the upstream trainer
into a configurable RoboTwin fine-tuning entry point. The project uses
**DeepSpeed ZeRO-2 + BF16**, configurable gradient accumulation, HDF5 loading,
precomputed language embeddings, image augmentation, periodic checkpointing,
and W&B-compatible reporting. Runtime-sensitive CUDA/NCCL settings are exposed
as environment variables instead of being treated as model changes.

The retained environment records an **NVIDIA RTX 6000 Ada**, PyTorch 2.1 / CUDA
12.1, DeepSpeed 0.14.2, and FlashAttention 2.5.5. This keeps the training setup
reproducible without coupling the algorithm to one workstation configuration.

### 2. Stability and monitoring

Training diagnostics separate **optimization failure** from **data-conditioning
failure**. Loss/NaN behavior, learning rate, GPU memory and throughput are
checked together with the HDF5 episode, normalization statistics, language
embedding, action mask, and control-frequency inputs. Input-audit hooks are kept
in the deployment path for the same reason: a numerically stable diffusion loss
does not guarantee that the robot is receiving semantically correct conditions.

The optimization loss measures action-sequence reconstruction under diffusion;
final policy quality is therefore not selected from loss alone. Closed-loop
task completion remains the meaningful metric.

### 3. Closed-loop evaluation and reproducibility

Evaluation replaces expert replay with an XPolicyLab policy client/server loop:

```text
RoboTwin Observation
        ↓  WebSocket / TCP
RDT Policy Server
        ↓
64-step Action Chunk
        ↓
RoboTwin Execution
        ↓
Task-specific check_success()
```

`robotwin/scripts/eval_policy_xpolicylab.py` keeps evaluation outputs separated
by task/checkpoint and adds **same-seed / fixed-instruction audit hooks**. Initial
robot state, camera observations, instruction, and the first predicted action
chunk can be captured together, making it possible to distinguish environment
initialization differences from policy-input or policy-output differences.

### 4. Benchmark boundary

The evaluation design separates two questions:

- **Clean**: did task-specific fine-tuning learn the manipulation skill under the
  training-like distribution?
- **Randomized**: how much of that performance survives changes in scene,
  lighting, background, object pose, or camera configuration?

Representative tasks retained in this repository are `adjust_bottle` and
`pick_dual_bottles`, covering arm selection/single-object manipulation and
synchronized bimanual manipulation respectively. Upstream RoboTwin/RDT numbers
are treated as **reference benchmarks**; results are only compared directly
when embodiment, data protocol, control mode, rollout count, and success
predicate are aligned.

## External requirements

Place or symlink model assets under:

```text
policy/rdt_1b/weights/RDT/
├── rdt-1b/
├── t5-v1_1-xxl/
└── siglip-so400m-patch14-384/
```

For simulation, either create `robotwin/assets` as a symlink to a compatible
RoboTwin asset tree or set:

```bash
export ROBOTWIN_ASSETS_ROOT=/path/to/RoboTwin/assets
```

## Entry points

```bash
# Expert demonstration and native XPolicyLab-v1.0 HDF5 collection
bash scripts/collect_demo.sh adjust_bottle demo_clean 0

# Optional legacy raw-HDF5 conversion
bash scripts/prepare_dataset.sh adjust_bottle demo_clean 50

# Link HDF5 and create one language embedding per episode
bash scripts/encode_language.sh RoboTwin adjust_bottle arx_x5 joint 50 /path/to/data --gpu 0

# Inspect/reproduce the official upstream pretraining pipeline
bash scripts/pretrain_rdt.sh

# Fine-tune released RDT-1B; GPU list may be `0` or `0,1,...`
bash scripts/finetune_rdt.sh RoboTwin adjust_bottle arx_x5 joint 0 0

# Start only the RDT policy server
bash scripts/serve_rdt.sh

# Start server and RoboTwin evaluation client
bash scripts/eval_robotwin.sh RoboTwin adjust_bottle <checkpoint> arx_x5 joint 0 0 0 rdt_1b robotwin
```

The collection path writes state at frame `t` and action from frame `t+1`,
plus RGB from head/left-wrist/right-wrist cameras and episode instructions.
The compatibility converter in `robotwin/scripts/convert_legacy_hdf5.py` is for
older raw RoboTwin HDF5 trees and is not required for newly collected data.

## Attribution

RDT and RoboTwin sources are MIT licensed; XPolicyLab is Apache-2.0. See
`THIRD_PARTY_NOTICES.md` and the component license files. Debugging features
such as same-seed audits, input traces, local NCCL defaults, and third-view
videos are reproducibility utilities, not algorithmic contributions.
