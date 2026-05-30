# PRAGMA Lite

PRAGMA Lite is a lightweight PRAGMA-inspired transformer for transactional event sequences. It supports structured tokenization, masked pretraining, checkpoint-based inference, and frozen-probe evaluation on public AML-style datasets such as TransXion and IBM AML.

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

## Notes

- `--split all` runs on the full tokenized dataset without split filtering.
- If `--split_dir` is omitted, inference scripts try to infer it from `--data_dir`.
- Checkpoints store both model weights and the tokenizer path used during training.
