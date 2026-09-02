# Mixed-dataset statistics

`dataset_stat.json` is generated from the complete four-task HDF5 tree before training:

```bash
bash scripts/compute_dataset_stats.sh
```

The previous `experiments/adjust_bottle/dataset_stat.json` was a single-task snapshot and is not runtime authority for the shared multi-task policy. The active statistics file must be recomputed after the final task data are assembled.
