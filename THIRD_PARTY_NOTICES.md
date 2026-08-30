# Third-Party Notices

This repository combines a curated subset of three upstream projects. The
root Apache-2.0 license applies to this project's original integration code and
documentation; it does not replace the component licenses below.

## RoboticsDiffusionTransformer

- Source: <https://github.com/thu-ml/RoboticsDiffusionTransformer>
- Revision: `cd79363a1387e8f81c7724d070ef7e45fd23150f`
- License: MIT, retained in `rdt/LICENSE`
- Scope: `rdt/models`, the upstream pretraining/data pipeline, training loop,
  configs, and the AgileX inference example.

The RoboTwin HDF5 reader, local T5 path support, and fine-tuning dataset glue
were adapted through XPolicyLab and are marked as modified in source.

## RoboTwin

- Source: <https://github.com/RoboTwin-Platform/RoboTwin>
- Revision: `30954692d06ba7e89f7a6b76064f4062c488fa81`
- License: MIT, retained in `robotwin/LICENSE`
- Scope: `robotwin/envs`, representative task definitions, expert motion and
  collection code, instruction generation, HDF5 serialization, and policy
  evaluation.

Project modifications are limited to third-view evaluation video support,
same-seed/fixed-instruction audits, curated import paths, external asset-root
support, and lazy loading of the clutter asset catalog.

## XPolicyLab

- Source: the `XPolicyLab` checkout distributed with RoboTwin
- Revision: `c07a09614dd44cc4a67483bcb9a82e7439d99926`
- License: Apache License 2.0, retained in `policy/xpolicylab/LICENSE`
- Scope: the minimal policy server/client runtime in `policy/xpolicylab` and
  the RDT adapter and orchestration scripts in `policy/rdt_1b`.

Files derived from XPolicyLab and changed by this project include
`policy/rdt_1b/model.py`, `deploy.yml`, `train.sh`, `process_data.sh`, the eval
setup scripts, and package/path adaptations in `policy/xpolicylab`.

## Model and dataset artifacts

RDT-1B, T5, SigLIP weights, fine-tuned checkpoints, RoboTwin assets, generated
demonstrations, language embeddings, and evaluation videos are not included.
Their own download terms and licenses apply.
