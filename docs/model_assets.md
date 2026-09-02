# Model assets

The repository keeps model weights external to Git. The runtime expects three released components:

| Component | Role | Official source | Expected local path |
| --- | --- | --- | --- |
| RDT-1B | diffusion VLA policy | https://huggingface.co/robotics-diffusion-transformer/rdt-1b | `policy/rdt_1b/weights/RDT/rdt-1b` |
| T5-v1.1-XXL | language encoder | https://huggingface.co/google/t5-v1_1-xxl | `policy/rdt_1b/weights/RDT/t5-v1_1-xxl` |
| SigLIP SO400M Patch14-384 | vision encoder | https://huggingface.co/google/siglip-so400m-patch14-384 | `policy/rdt_1b/weights/RDT/siglip-so400m-patch14-384` |

Check local paths:

```bash
bash scripts/setup_models.sh
```

Or download explicitly with the Hugging Face CLI:

```bash
bash scripts/setup_models.sh download
```

For fine-tuning, the project precomputes T5 embeddings per RoboTwin episode so T5-XXL does not need to remain resident in GPU memory throughout RDT training.
