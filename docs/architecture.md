# Architecture and execution path

## RDT

`rdt/models/rdt/blocks.py` implements timestep embeddings, self-attention, and
cross-attention blocks. `model.py` alternates language and image conditions
over the noisy action/state sequence. `rdt_runner.py` adapts language, image,
and state tokens, applies DDPM forward noise during training, computes MSE
against the configured diffusion target, and uses DPM-Solver for iterative
action denoising at inference.

`rdt/train/train.py` loads T5 and SigLIP, creates `RDTRunner`, and performs the
Accelerate/DeepSpeed optimization loop. Pretraining consumes the Open-X
producer/buffer pipeline; RoboTwin fine-tuning selects the HDF5 consumer.

## RoboTwin

Each representative task defines three important methods:

- `load_actors()`: scene and randomized task state.
- `play_once()`: the expert motion program built from grasp/place/move APIs.
- `check_success()`: the task-specific terminal success predicate.

Collection first finds seeds for which planning and `check_success()` pass,
saves the expert joint paths, then replays them with observation capture.
Evaluation replaces the expert replay with action chunks returned by the RDT
policy server and uses the same task success predicate.

## Deployment boundary

The RoboTwin process sends RGB images, bimanual joint state, instruction, and
control frequency through the XPolicyLab protocol. `policy/rdt_1b/model.py`
forms the two-frame image history, encodes language with T5, calls RDT, and
unpacks its 14-dimensional bimanual action chunks for RoboTwin.
