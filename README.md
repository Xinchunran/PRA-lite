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
- `src/training/linear_probe.py`: frozen embedding probe with standard scaling and logistic regression.
- `scripts/run_lite_benchmark.py`: fixed-seed lite benchmark entry point.

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

If you want the representation directly in Python, use the structured batch interface and read `zh_usr`, `zh_evt`, or `record_embedding`:

```python
import torch

from src.model.pragma_lite.model import PragmaLiteModel, PragmaLiteConfig
from src.training.checkpoint import load_checkpoint

ckpt = load_checkpoint("runs/pretrain_ddp_4gpu_full/best.ckpt", map_location="cpu")
model = PragmaLiteModel(PragmaLiteConfig(**ckpt["model_cfg"]))
model.load_state_dict(ckpt["model_state"])
model.eval()

with torch.no_grad():
    outputs = model(
        profile_key_ids=profile_key_ids,
        profile_value_ids=profile_value_ids,
        profile_value_pos=profile_value_pos,
        profile_time=profile_time,
        profile_mask=profile_mask,
        event_key_ids=event_key_ids,
        event_value_ids=event_value_ids,
        event_value_pos=event_value_pos,
        event_token_mask=event_token_mask,
        event_time=event_time,
        calendar_features=calendar_features,
        event_mask=event_mask,
    )
    user_repr = outputs["zh_usr"]
    last_event_repr = outputs["zh_evt"][:, -1, :]
    record_repr = outputs["record_embedding"]
```

`zh_usr` is the user-level history embedding used by the frozen probe.

## Lite Benchmark

Run a fixed-seed frozen-probe benchmark over a sampled subset:

```bash
python -m scripts.run_lite_benchmark \
  --checkpoint runs/pretrain_ddp_4gpu_full/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split_dir data/splits/transxion \
  --output_dir outputs/lite_benchmark \
  --num_records 1000 \
  --seed 42 \
  --batch_size 256 \
  --device cpu
```

This benchmark:

- fixes `seed`, split sampling, and output layout
- evaluates `zh_usr`, `last_evt`, `concat`, and `record` representations
- fits `StandardScaler + LogisticRegression(solver="lbfgs")`
- writes per-representation predictions plus `benchmark_report.json`

The current lite benchmark is designed for reproducible small-scale comparisons. It is the recommended evaluation path for frozen embeddings.

## Lite-Scope Deviations

PRA-lite intentionally does not reproduce PRAGMA's large-scale training infrastructure.
For small reproducible benchmarks, we use:

- AdamW instead of Muon + AdamW
- fixed batch size instead of token-budget dynamic batching
- padded tensors instead of sequence packing with varlen attention
- local Parquet datasets instead of LMDB user index + event-count shards

These choices affect training efficiency, not the core architectural hypothesis.
The benchmark therefore evaluates whether PRAGMA-style key-value-time tokenization,
profile/event/history encoders, masked event modeling, and frozen user embeddings
are useful on small public transaction datasets.

## Probe Notes

- `src/training/linear_probe.py` now runs a frozen probe, not end-to-end finetuning.
- Supported representation choices are `zh_usr`, `last_evt`, `concat`, and `record`.
- The current downstream probe is closer to PRAGMA Section 3.1 than a trainable linear head.
- LoRA-style adaptation is not implemented yet; treat it as future work for a closer Section 3.1 match.

## Notes

- `--split all` runs on the full tokenized dataset without split filtering.
- If `--split_dir` is omitted, the inference scripts try to infer it from `--data_dir`.
- Checkpoints store both model weights and the tokenizer path used during training.
