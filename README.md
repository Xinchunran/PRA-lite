# PRAGMA Lite

PRAGMA Lite is a lightweight PRAGMA-inspired transformer for transactional event sequences. It supports structured tokenization, masked pretraining, checkpoint-based inference, and frozen-probe evaluation on public AML-style datasets such as TransXion and IBM AML.

## PRAGMA vs PRA-lite Implementation Status

This table compares the original [PRAGMA paper (arXiv:2604.08649)](https://arxiv.org/abs/2604.08649) features with the current PRA-lite implementation.

| Category | Feature | PRAGMA | PRA-lite |
| :--- | :--- | :---: | :---: |
| **Architecture** | Profile Encoder (Bidirectional Transformer) | ☑️ | ☑️ |
| | Event Encoder (Bidirectional Transformer) | ☑️ | ☑️ |
| | History Encoder (Bidirectional Transformer) | ☑️ | ☑️ |
| | Event-local attention kernel that prevents cross-event token attention | ☑️ | ✖️ |
| | RoPE time/position encoding in profile and history encoders | ☑️ | ☑️ |
| | Key-value-time representation with fused history context | ☑️ | ☑️ |
| **Data Processing** | Key-Value-Time Tokenisation | ☑️ | ☑️ |
| | Numeric Percentile Bucketing | ☑️ | ☑️ |
| | Log-seconds Relative Time Feature | ☑️ | ☑️ |
| | Calendar Time Features | ☑️ | ☑️ |
| | LMDB-backed storage / shard-based streaming pipeline | ☑️ | ☑️ |
| | Low-frequency Vocab Pruning | ✖️ | ☑️ |

## Main Entry Points

- `src/model/pragma_lite/model.py`: model definition
- `src/inference/predict.py`: batch prediction
- `src/inference/extract_embeddings.py`: embedding export
- `src/training/linear_probe.py`: frozen linear probe
- `scripts/run_lite_benchmark.py`: reproducible evaluation benchmark
- `scripts/run_transxion_benchmark.sh`: public TransXion prepare/train entry point
- `scripts/run_ibm_aml_medium_streaming.sh`: IBM AML medium streaming prepare + train entry point

## Install

```bash
conda create -n pragma-lite python=3.10 -y
conda activate pragma-lite
pip install -r requirements.txt
```

## Inference

Run prediction from a saved checkpoint:

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

Export one embedding per entity:

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

If you need the representation directly in Python, use `zh_usr`, `zh_evt`, or `record_embedding` from the model output.

## Lite Benchmark

Run the fixed-seed frozen-probe benchmark:

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

This benchmark evaluates `zh_usr`, `last_evt`, `concat`, and `record`, then writes predictions and `benchmark_report.json`.

## TransXion

Use the unified shell entry point:

```bash
bash scripts/run_transxion_benchmark.sh <action> <scale>
```

Supported actions:

- `download`
- `prepare`
- `train`
- `all`

Supported scales:

- `mini`
- `small`

Examples:

```bash
bash scripts/run_transxion_benchmark.sh prepare mini
bash scripts/run_transxion_benchmark.sh prepare small
NPROC_PER_NODE=2 bash scripts/run_transxion_benchmark.sh train small
```

## IBM AML

Download the Kaggle files:

```bash
conda activate pragma-lite

KAGGLE_FILE=LI-Small_Trans.csv bash scripts/download_ibm_aml_kaggle.sh
KAGGLE_FILE=LI-Small_accounts.csv bash scripts/download_ibm_aml_kaggle.sh
KAGGLE_FILE=LI-Medium_Trans.csv bash scripts/download_ibm_aml_kaggle.sh
KAGGLE_FILE=LI-Medium_accounts.csv bash scripts/download_ibm_aml_kaggle.sh
```

Prepare a static LMDB dataset:

```bash
RAW_CSV=LI-Small_Trans.csv bash scripts/prepare_ibm_aml_lmdb.sh
```

Train on IBM AML:

```bash
TRAIN_BATCH_SIZE=16 bash scripts/train_ibm_aml_lmdb.sh
```

For medium-scale streaming prepare + train:

```bash
conda activate pragma-lite
bash scripts/run_ibm_aml_medium_streaming.sh
```

Useful overrides:

```bash
ROWS_PER_SHARD=250000 \
TRAIN_BATCH_SIZE=16 \
NPROC_PER_NODE=4 \
PRECISION=bf16 \
TOKENIZE_NUM_WORKERS=8 \
bash scripts/run_ibm_aml_medium_streaming.sh
```

### Split Protocols

IBM AML experiments now keep two split protocols in parallel:

| Split Protocol | Default Data Root | Record Policy | Split Rule | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| Random / Hash Split | `data/streaming/ibm_aml_li_medium` | One account-level evaluation point per encoded sample in the legacy streaming pipeline | Hash-by-entity with `train/valid/test` fractions | Fast iteration, backward-compatible baselines, quick MLM debugging |
| Stage C PRAGMA-Style Split | `data/streaming/ibm_aml_li_medium_pragma_c` | Multi-evaluation-point account-centric records tied to real transaction timestamps | Global `evaluation_time` split with `train / embargo / valid / calibration / embargo / test` | Financial-style temporal validation, leakage control, isolated Stage C checkpoints |

The legacy random/hash split entry point remains:

```bash
bash scripts/run_ibm_aml_medium_streaming.sh
```

Stage C uses a separate data root, tokenizer, manifest, logs, plots, and checkpoints:

```bash
bash scripts/prepare_ibm_aml_li_pragma_c.sh
bash scripts/run_ibm_aml_li_medium_pragma_c_pretrain.sh
```

The Stage C train entry point defaults to `MAX_EVENTS=512` and `split_mode=pragma_c`. You can still fall back to the legacy split logic without changing training code:

```bash
SPLIT_MODE=random bash scripts/run_ibm_aml_li_medium_pragma_c_pretrain.sh
```

When Stage C is launched through `scripts/run_ibm_aml_li_medium_pragma_c_pretrain.sh`, logs and checkpoints are written to an isolated run root:

```text
runs/pretrain_ibm_aml_li_medium_pragma_c/
```

### Stage C Training Notes

- Stage C shards can contain batches where some DDP ranks have no masked MLM targets. The trainer now keeps all ranks on the same backward path by using a zero loss on target-free ranks instead of letting those ranks `continue`.
- Because of that safeguard, you may occasionally see `train_loss=-0.0000` in `train.log`. This is expected for the local rank on that step and does not mean the whole global step is invalid.
- A step is only fully skipped when **all** ranks have no supervised MLM targets. In that case the log prints `skipped=no_masked_targets`.
- Older Stage C runs may still contain `Grad strides do not match bucket view strides` warnings near the top of `train.log`. The current model code canonicalizes the grad layout for the learnable CLS tokens, so check the most recent launch block before assuming the warning is still active.
- If `train.log` contains multiple restarts, read it by launch block rather than assuming one continuous run. The newest run begins at the latest `Using TORCHRUN_BIN=...` line.

## Notes

- `--split all` runs on the full tokenized dataset without split filtering.
- If `--split_dir` is omitted, inference scripts try to infer it from `--data_dir`.
- Checkpoints store both model weights and the tokenizer path used during training.
