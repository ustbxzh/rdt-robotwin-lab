# Project entry points

## Main shared-policy workflow

Use these for the four-task ARX-X5 experiment:

1. `setup_models.sh` — check/download RDT-1B, T5 and SigLIP.
2. `collect_expert_suite.sh` — collect all four RoboTwin expert datasets.
3. `prepare_multitask_data.sh` — assemble the mixed HDF5 tree and encode language.
4. `compute_dataset_stats.sh` — compute statistics from the mixed dataset.
5. `train_multitask.sh` — fine-tune one shared RDT checkpoint.
6. `eval_multitask_suite.sh` — evaluate the same checkpoint over 4 tasks × 2 domains.

## Low-level / compatibility wrappers

`collect_demo.sh`, `prepare_dataset.sh`, `encode_language.sh`, `finetune_rdt.sh`, `serve_rdt.sh`, and `eval_robotwin.sh` remain available for single-stage debugging and backward-compatible manual runs. The README quick start uses the shared-policy workflow above.
