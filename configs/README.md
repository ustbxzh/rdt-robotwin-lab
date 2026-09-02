# Configuration snapshots

This directory keeps earlier project configuration snapshots for audit/reference only.

Active runtime configuration is now organized by ownership:

- RDT model/training defaults: `rdt/configs/`
- RoboTwin collection/evaluation profiles: `robotwin/env_cfg/task_config/`
- XPolicyLab RDT runtime: `policy/rdt_1b/deploy.yml`
- Main shared-policy experiment: `experiments/rdt_robotwin_multitask/`

Do not treat files under `configs/robotwin`, `configs/training`, or `configs/deployment` as the authoritative experiment entry points.
