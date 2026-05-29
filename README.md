# PRAGMA Lite

PRAGMA Lite is a lightweight transformer for user-level representation learning on transactional event sequences.

This repository focuses on inference and representation extraction from a pretrained checkpoint. The released model was trained with DDP on 4x NVIDIA RTX A4500 GPUs.

## What It Does

- Encodes a tokenized user sequence into contextual hidden states.
- Uses the `[USR]` position as the user-level representation.
- Supports binary prediction export and embedding export from saved checkpoints.

## Repository Entry Points

- `src/inference/predict.py`: batch prediction to a parquet file.
- `src/inference/extract_embeddings.py`: export user representations to a parquet file.
- `src/model/pragma_lite/model.py`: model definition.

## Inference

Run batched prediction from a saved checkpoint:

```bash
python -m src.inference.predict \
  --checkpoint runs/pretrain_ddp_4gpu_full/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split test \
  --split_dir data/splits/transxion \
  --batch_size 256 \
  --output_file outputs/test_predictions.parquet \
  --device cpu
```

Output columns:

- `record_id`
- `entity_id`
- `label` if available
- `probability`
- `prediction`

## Get Representations

Export one vector per entity:

```bash
python -m src.inference.extract_embeddings \
  --checkpoint runs/pretrain_ddp_4gpu_full/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split test \
  --split_dir data/splits/transxion \
  --batch_size 256 \
  --output_file outputs/test_embeddings.parquet \
  --device cpu
```

The output parquet contains:

- `entity_id`
- `label` if available
- `embedding_0 ... embedding_{d_model-1}`

## Representation API

If you want the representation directly in Python, use the hidden state at the `[USR]` position:

```python
import torch

from src.model.pragma_lite.model import PragmaLite, PragmaLiteConfig
from src.training.checkpoint import load_checkpoint

ckpt = load_checkpoint("runs/pretrain_ddp_4gpu_full/best.ckpt", map_location="cpu")
model = PragmaLite(PragmaLiteConfig(**ckpt["model_cfg"]))
model.load_state_dict(ckpt["model_state"])
model.eval()

with torch.no_grad():
    hidden = model(input_ids, attention_mask=attention_mask)
    user_repr = hidden[:, 0, :]
```

`user_repr` is the user-level embedding used for downstream tasks.

## Notes

- `--split all` runs on the full tokenized dataset without split filtering.
- If `--split_dir` is omitted, the inference scripts try to infer it from `--data_dir`.
- Checkpoints store both model weights and the tokenizer path used during training.
