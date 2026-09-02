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

## Data pipeline

The RoboTwin-to-RDT path is implemented as a complete robot-demonstration
pipeline rather than a simple offline dataset conversion:

```text
RoboTwin Task
    ↓
Scene Randomization
    ↓
Expert Motion Planning
    ↓
Successful Trajectory Filtering
    ↓
Trajectory Replay + Observation Capture
    ↓
Episode-level HDF5
    ↓
Temporal / Language / Frequency Alignment
    ↓
Normalization & Dataset Statistics
    ↓
RDT Fine-tuning Dataset
```

### 1. Expert demonstration generation and quality filtering

Task definitions such as `robotwin/envs/adjust_bottle.py` and
`robotwin/envs/pick_dual_bottles.py` provide three key pieces of the data
source: randomized scene construction, expert motion primitives, and the final
`check_success()` predicate.

`robotwin/scripts/collect_data.py` first searches randomized seeds for which
both motion planning and task execution succeed. Only trajectories satisfying

```text
plan_success && check_success()
```

are accepted. Their joint paths and seeds are stored, then replayed under the
same task initialization for observation capture. The replayed trajectory is
checked again before the episode is kept, so failed expert motions are filtered
before they enter the training set.

This gives the data pipeline an explicit **trajectory-level validity gate**
instead of relying on raw simulator rollouts.

### 2. Multimodal episode schema

Each successful demonstration is serialized as one HDF5 episode containing
vision, proprioception, actions, language, and control metadata:

```text
episode_xxxxxxx.hdf5
├── /vision
│   ├── cam_head/colors
│   ├── cam_left_wrist/colors
│   └── cam_right_wrist/colors
├── /state
│   ├── left_arm_joint_states
│   ├── left_ee_joint_states
│   ├── right_arm_joint_states
│   └── right_ee_joint_states
├── /action
│   ├── left_arm_joint_states
│   ├── left_ee_joint_states
│   ├── right_arm_joint_states
│   └── right_ee_joint_states
├── /instructions
└── /additional_info/frequency
```

The head camera supplies global scene context, while the two wrist cameras
retain local manipulation evidence during grasping, occlusion, and bimanual
interaction. This keeps the visual observations and the robot state/action
stream inside the same episode boundary.

### 3. State-action temporal alignment

For imitation learning, the supervision target associated with the current
observation is the **next control state/action**, not the state that has already
been observed. For a captured joint sequence

```text
q[0], q[1], ..., q[N-1]
```

the native collection path constructs

```text
state  = q[:-1]
action = q[1:]
RGB    = RGB[:-1]
```

so that every training sample follows the same control semantics:

```text
RGB(t), State(t) → Action(t+1)
```

All modalities therefore share an `N-1` horizon. This avoids a one-step
semantic mismatch between robot observation and action supervision. The exact
alignment is documented in `docs/dataset_alignment.md`.

### 4. Control-frequency consistency

The RoboTwin fine-tuning setup uses **15 Hz** control. The frequency is treated
as part of the action semantics rather than passive metadata because it defines
the physical duration represented by one action step and by an entire action
chunk.

The same control-frequency setting is therefore kept consistent across

```text
demonstration data
        =
training conditioning
        =
policy execution
```

which prevents an action sequence from representing different physical time
horizons during training and inference.

### 5. Episode-level language alignment

Language conditioning is aligned at demonstration granularity rather than by
sharing one embedding for an entire task. `policy/rdt_1b/encode_language.py`
encodes each episode instruction with T5 and mirrors the HDF5 path under the
language-embedding directory:

```text
data/.../episode_0000007.hdf5
             ↕
lang_embeds/.../episode_0000007.pt
```

The HDF5 training loader derives the language path from the episode path, so
vision, state, action, and language all originate from the same demonstration.
This turns language handling from task-level labeling into **episode-level
semantic alignment**.

### 6. Normalization and dataset statistics

RoboTwin robot states/actions must be adapted to the numerical distribution
expected by the pretrained RDT policy. The fine-tuning pipeline therefore uses
task-specific dataset statistics rather than blindly reusing unrelated robot
data statistics.

For the `adjust_bottle` experiment, the resulting statistics are retained in

```text
experiments/adjust_bottle/dataset_stat.json
```

and are supplied to the RDT HDF5 loader during fine-tuning. Together with the
expert-success gate and episode-level correspondence checks, this forms the
final dataset preparation stage before optimization.

### Data-engineering capabilities represented by this pipeline

| Capability | Project implementation |
| --- | --- |
| Expert data generation | Task-level motion primitives + successful-seed replay |
| Multimodal data modeling | Three RGB views + bimanual state/action + instruction |
| Temporal alignment | `State(t) → Action(t+1)` with a shared `N-1` horizon |
| Control-time semantics | 15 Hz conditioning kept consistent across train/inference |
| Cross-modal alignment | One T5 embedding per HDF5 episode |
| Distribution adaptation | Task-specific RDT state/action dataset statistics |

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
