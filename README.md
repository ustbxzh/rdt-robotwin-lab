# RDT × RoboTwin Lab

A source-first, end-to-end view of RDT-1B adaptation to RoboTwin:

```text
task construction → expert demonstration → RGB/qpos/action collection
→ episode-level language alignment → RDT fine-tuning
→ policy deployment → RoboTwin evaluation → task success
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
