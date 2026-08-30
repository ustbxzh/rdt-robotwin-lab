# Installation

Use the repository environment snapshots under `environment/`, or run
`install.sh`. The installer creates/uses a Conda environment, installs the
top-level package, and optionally downloads model assets.

To avoid downloads and use existing assets:

```bash
RDT_WEIGHTS_SRC=/path/to/RDT_weights bash policy/rdt_1b/install.sh
```

To install code only:

```bash
RDT_SKIP_WEIGHTS=1 bash policy/rdt_1b/install.sh
```

Weights and checkpoints are ignored and must never be committed.
