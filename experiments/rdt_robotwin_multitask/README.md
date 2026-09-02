# RDT × RoboTwin Multi-task Experiment

This directory defines one shared RDT-1B policy adapted to multiple manipulation tasks on the same **ARX-X5** embodiment.

```text
4 task datasets
      ↓
shared HDF5 / language pipeline
      ↓
ONE RDT-1B fine-tuning run
      ↓
ONE checkpoint
      ↓
4 tasks × {Clean, Randomized}
```

## Task suite

| Task | Capability |
| --- | --- |
| `adjust_bottle` | arm selection, grasping, orientation control |
| `lift_pot` | synchronized bimanual grasp and lift |
| `handover_block` | arm-to-arm transfer and target placement |
| `blocks_ranking_size` | visual size reasoning and long-horizon multi-object placement |

All tasks share the same robot, observation/action contract, RDT checkpoint and deployment runtime. Task YAML files describe only task-specific metadata; `eval/clean.yaml` and `eval/randomized.yaml` are reused across the full suite.

## Experiment files

- `dataset.yaml` — mixed-dataset layout and task list.
- `train.env` — local low-shot fine-tuning schedule.
- `deploy.yaml` — shared policy/runtime contract.
- `tasks/` — four task profiles.
- `eval/` — common Clean and Randomized evaluation profiles.
- `stats/` — statistics generated from the mixed dataset before training.
- `results/` — final per-task and aggregate metrics.

The committed configuration is the experiment protocol. Raw HDF5 data, model weights and checkpoints remain external to Git.
