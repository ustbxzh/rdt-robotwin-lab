# RDT-1B policy adapter

This directory contains only the RoboTwin-facing RDT adapter and orchestration
code. The model and training implementation is the single top-level `rdt/`
package.

Runtime assets are resolved relative to `weights/RDT/` by default and may be
overridden with `TEXT_ENCODER_NAME`, `VISION_ENCODER_NAME`, and
`RDT_PRETRAINED_MODEL`. Generated data, language embeddings, and checkpoints
stay under ignored directories in this folder.

The training sequence is:

```text
process_data.sh → encode_language.py → train.sh
```

`encode_language.py` writes an embedding for every episode, preserving the
HDF5 relative path. `train.sh` uses the 15 Hz RoboTwin dataset configuration
and defaults to `experiments/adjust_bottle/dataset_stat.json`.
