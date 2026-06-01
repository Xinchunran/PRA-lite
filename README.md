# PRAGMA Lite

PRAGMA Lite is a lightweight PRAGMA-inspired transformer for transactional event sequences. It currently supports structured tokenization, masked pretraining, checkpoint-based inference, reusable embedding extraction, and frozen/downstream benchmarking on public AML-style datasets such as TransXion and IBM AML.

## Original Vs Realized

This table compares the original [PRAGMA paper (arXiv:2604.08649)](https://arxiv.org/abs/2604.08649) design intent with what is realized in the current PRA-lite codebase.

| Category | Feature | Original PRAGMA | Realized In PRA-lite | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Architecture** | Profile encoder | ☑️ | ☑️ | Bidirectional transformer over profile key/value tokens |
| | Event encoder | ☑️ | ☑️ | Bidirectional transformer over event-local key/value tokens |
| | History encoder | ☑️ | ☑️ | Bidirectional transformer over `[USR]` plus ordered `[EVT]` states |
| | Shared key/value embedding table | ☑️ | ☑️ | `key_ids` and `value_ids` share the same token embedding table |
| | `[USR]` / `[EVT]` taken from shared token table | ☑️ | ☑️ | No separate learned CLS parameters in the current path |
| | MLM logits matched against shared embedding table | ☑️ | ☑️ | Current default is tied MLM output projection |
| | Event-local attention kernel / varlen event packing | ☑️ | Partial | PRA-lite still uses rectangular padded tensors, not the paper's custom varlen packing kernel |
| | Additive time projection + history order embedding ablations | ☑️ | ☑️ | Controlled by `use_additive_time_proj` and `use_history_order_emb` |
| **Tokenizer / Data** | Key-value-time tokenization | ☑️ | ☑️ | Profile/event fields become key/value/time structured inputs |
| | Numeric bucket tokenizer | ☑️ | ☑️ | Numeric fields use bucket tokens and optional `#ZERO` handling |
| | Categorical tokenizer with field-aware unknowns | ☑️ | ☑️ | Field-specific `[UNK]` path is supported in tokenizer v2 |
| | Text/BPE-style tokenizer | ☑️ | ☑️ | Textual fields can expand to shared `T:*` tokens via tokenizer v2 |
| | Multi-value field expansion | ☑️ | ☑️ | Field keys are replicated and `value_pos` increments `0,1,2,...` |
| | Within-field positional embedding path | ☑️ | ☑️ | Implemented through `value_pos_emb` once tokenizer emits `value_pos > 0` |
| | Static sine/cosine within-field position encoding | ☑️ | ✖️ | PRA-lite uses learned `value_pos_emb`, not fixed sinusoidal within-field encoding |
| | Calendar features + 2-layer MLP | ☑️ | ☑️ | Event calendar features are projected with a 2-layer MLP |
| | Time-to-last-event anchoring | ☑️ | Partial | Configurable; current preprocessing path defaults to `last_event` |
| | Tokenizer v2 metadata | N/A | ☑️ | Stores `field_value_types`, `categorical_values`, text tokenizer metadata, and limits |
| | LMDB-backed tokenized shards | ☑️ | ☑️ | Used for static LMDB and manifest-driven streaming training |
| **Training / Evaluation** | Masked token / key / event objectives | ☑️ | ☑️ | Current collator supports token-, key-, and event-level masking |
| | Dynamic packing / token-budget batching | ☑️ | Partial | Event-count bucketed token-budget batching is implemented without custom varlen kernels |
| | Batch-local trimming | N/A | ☑️ | Collation trims to batch-local active profile/event lengths |
| | Full validation + quick validation | ☑️ | ☑️ | Same metric schema, different batch coverage |
| | Stratified validation metrics | N/A | ☑️ | Categorical / numerical / text / key / mask-source / top-k metrics are logged |
| | `metrics.jsonl` structured logging | N/A | ☑️ | Train and valid metrics are appended as JSONL records |
| **Inference / Downstream** | Embedding extractor | ☑️ | ☑️ | Exports `zh_usr`, `last_evt`, `concat`, or `record` embeddings |
| | Frozen linear probe | ☑️ | ☑️ | Logistic-regression probe on extracted representations |
| | Downstream benchmark against tree baselines | ☑️ | ☑️ | IBM AML downstream benchmark supports PRAGMA-lite vs XGBoost / CatBoost with CV model selection |

## Main Entry Points

- `src/model/pragma_lite/model.py`: model definition
- `src/inference/predict.py`: batch prediction
- `src/inference/extract_embeddings.py`: embedding export
- `src/training/linear_probe.py`: frozen linear probe
- `scripts/benchmarks/run_lite_benchmark.py`: reproducible evaluation benchmark
- `scripts/benchmarks/run_ibm_aml_downstream_benchmark.py`: IBM AML downstream benchmark from pretrained checkpoint
- `scripts/benchmarks/run_transxion_benchmark.sh`: public TransXion prepare/train entry point
- `scripts/train/run_ibm_aml_medium_streaming.sh`: IBM AML medium streaming prepare + train entry point

## Install

```bash
conda create -n pragma-lite python=3.10 -y
conda activate pragma-lite
pip install -r requirements.txt
```

## License

This repository is currently distributed under a non-commercial license. See `LICENSE` for the exact terms before using the code, weights, or derived artifacts.

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
  --repr_type concat \
  --batch_size 256 \
  --output_file outputs/test_embeddings.parquet \
  --device cpu
```

Supported `--repr_type` values are:

- `zh_usr`: user/history representation
- `last_evt`: last valid event representation
- `concat`: concatenation of `zh_usr` and `last_evt`
- `record`: fused record embedding

The exported parquet contains:

```text
entity_id
label            # if the tokenized dataset has labels
embedding_0
embedding_1
...
embedding_d
```

If you need the representation directly in Python, use `zh_usr`, `zh_evt`, or `record_embedding` from the model output, or call the reusable extraction helpers in `src/inference/extract_embeddings.py`.

## Downstream Embeddings

For frozen-probe style evaluation, you can either:

1. Export embeddings first and train your own downstream model.
2. Use the existing linear probe.
3. Run the IBM AML downstream benchmark script that compares PRAGMA-lite features against raw-feature tree baselines.

Export embeddings for a downstream split:

```bash
python -m src.inference.extract_embeddings \
  --checkpoint runs/pretrain_ddp_4gpu_full/best.ckpt \
  --data_dir data/processed/transxion_full/tokenized \
  --split test \
  --split_dir data/splits/transxion_full \
  --repr_type concat \
  --batch_size 256 \
  --output_file outputs/transxion_full_test_concat.parquet \
  --device cpu
```

Run the frozen linear probe:

```bash
python -m src.training.linear_probe \
  --checkpoint runs/pretrain_ddp_4gpu_full/best.ckpt \
  --data_dir data/processed/transxion_full/tokenized \
  --split_dir data/splits/transxion_full \
  --output_dir outputs/transxion_probe \
  --repr_type concat \
  --batch_size 256 \
  --device cpu
```

Run the downstream benchmark against raw-feature baselines:

```bash
python scripts/benchmarks/run_ibm_aml_downstream_benchmark.py \
  --checkpoint runs/pretrain_ibm_aml_li_medium_pragma_lite_full_20k_latest/best.ckpt \
  --stream_root data/streaming/ibm_aml_li_medium_pragma_lite_full \
  --output_dir runs/ibm_aml_downstream_balanced_from_best_20k \
  --sample_size 50000 \
  --repr_type concat \
  --batch_size 256 \
  --cv_folds 3 \
  --positive_fraction 0.5 \
  --device cpu
```

This downstream benchmark:

- samples roughly `5w` IBM AML downstream evaluation points with a target `0.5` positive fraction
- rebuilds a benchmark-only tokenized dataset that is compatible with the selected checkpoint tokenizer
- extracts PRAGMA-lite embeddings and trains a logistic-regression downstream head
- builds raw baseline features from profile-state and anchor-transaction statistics
- selects hyperparameters with cross-validation on the non-test splits
- reports only `test` set `PR-AUC`, `ROC-AUC`, `F1`, and `F0.5`
- writes artifacts under `runs/.../benchmark_data`, `runs/.../metrics`, `runs/.../plots`, and `runs/.../predictions`

Submit the CPU Slurm job:

```bash
sbatch scripts/slurm/benchmarks/run_ibm_aml_downstream_cpu.slurm
```

The submitted job writes to:

- `runs/ibm_aml_downstream_balanced_from_best_20k/benchmark_data`
- `runs/ibm_aml_downstream_balanced_from_best_20k/metrics`
- `runs/ibm_aml_downstream_balanced_from_best_20k/plots`
- `runs/ibm_aml_downstream_balanced_from_best_20k/predictions`

## IBM AML Dataset Stats

IBM AML dataset statistics and publication-style plots are generated with:

```bash
python tools/build_ibm_aml_dataset_stats.py \
  --processed_dir data/processed/ibm_aml_li_medium \
  --stream_root data/streaming/ibm_aml_li_medium_pragma_lite_full \
  --output_dir data/processed/ibm_aml_li_medium/stat_plot
```

Current outputs live under `data/processed/ibm_aml_li_medium/stat_plot` and include:

- overview tables for entities, events, eval points, shards, and vocab
- field-type counts for processed features vs tokenizer fields
- encoder-input summaries such as mean history length, token length, and empty-history rate
- bias summaries grouped by primary bank and dominant payment format
- Nature-style blue/teal plots in `data/processed/ibm_aml_li_medium/stat_plot/plots`

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

## Validation Metrics

Pretraining validation now uses one shared metric function for both quick validation and full validation.

- `quick` validation runs every `valid_every` and uses `max_valid_batches`
- `full` validation runs every `full_valid_every` and evaluates the whole validation loader
- both modes write the same `valid_*` keys into `metrics.jsonl`
- the difference is coverage, not metric schema

Current validation metrics include:

- `valid_loss`, `valid_perplexity`, `valid_masked_accuracy`, `valid_top1_acc`, `valid_top5_acc`
- `valid_acc_categorical`, `valid_acc_numerical`, `valid_acc_text`, plus matching `valid_loss_*`
- `valid_acc_token_mask`, `valid_acc_key_mask`, `valid_acc_event_mask`, plus matching `valid_loss_*`
- key-level metrics such as `valid_acc_by_key_*` and `valid_loss_by_key_*`
- numeric reconstruction metrics such as `valid_num_bucket_mae`, `valid_num_within_1_acc`, and `valid_num_within_2_acc`

## TransXion

Use the unified shell entry point:

```bash
bash scripts/benchmarks/run_transxion_benchmark.sh <action> <scale>
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
bash scripts/benchmarks/run_transxion_benchmark.sh prepare mini
bash scripts/benchmarks/run_transxion_benchmark.sh prepare small
NPROC_PER_NODE=2 bash scripts/benchmarks/run_transxion_benchmark.sh train small
```

## IBM AML

Download the Kaggle files:

```bash
conda activate pragma-lite

KAGGLE_FILE=LI-Small_Trans.csv bash scripts/download/download_ibm_aml_kaggle.sh
KAGGLE_FILE=LI-Small_accounts.csv bash scripts/download/download_ibm_aml_kaggle.sh
KAGGLE_FILE=LI-Medium_Trans.csv bash scripts/download/download_ibm_aml_kaggle.sh
KAGGLE_FILE=LI-Medium_accounts.csv bash scripts/download/download_ibm_aml_kaggle.sh
```

Prepare a static LMDB dataset:

```bash
RAW_CSV=LI-Small_Trans.csv bash scripts/prepare/streaming/prepare_ibm_aml_lmdb.sh
```

Train on IBM AML:

```bash
TRAIN_BATCH_SIZE=16 bash scripts/train/train_ibm_aml_lmdb.sh
```

For medium-scale streaming prepare + train:

```bash
conda activate pragma-lite
bash scripts/train/run_ibm_aml_medium_streaming.sh
```

Useful overrides:

```bash
ROWS_PER_SHARD=250000 \
TRAIN_BATCH_SIZE=16 \
NPROC_PER_NODE=4 \
PRECISION=bf16 \
TOKENIZE_NUM_WORKERS=8 \
bash scripts/train/run_ibm_aml_medium_streaming.sh
```

### Split Protocols

IBM AML experiments now keep two split protocols in parallel:

| Split Protocol | Default Data Root | Record Policy | Split Rule | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| Random / Hash Split | `data/streaming/ibm_aml_li_medium` | One account-level evaluation point per encoded sample in the legacy streaming pipeline | Hash-by-entity with `train/valid/test` fractions | Fast iteration, backward-compatible baselines, quick MLM debugging |
| PRA-lite Leakage-Prevent Split | `data/streaming/ibm_aml_li_medium_pragma_lite_full` | Multi-evaluation-point account-centric records tied to real transaction timestamps with Figure 4-aligned tokenization/model updates | Global `evaluation_time` split with `train / embargo / valid / calibration / embargo / test` | Financial-style temporal validation, leakage control, isolated preprocessing and pretraining runs |

The legacy random/hash split entry point remains:

```bash
bash scripts/train/run_ibm_aml_medium_streaming.sh
```

The leakage-prevent path uses a separate data root, tokenizer, manifest, logs, plots, and checkpoints:

```bash
bash scripts/prepare/pragma_c/prepare_ibm_aml_li_pragma_lite_full.sh
```

Legacy `pragma_c` scripts are still kept in the repo for reference, but the current leakage-prevent preprocessing path is rooted at `pragma_lite_full`.

The Stage C train entry point defaults to `MAX_EVENTS=256` and `split_mode=pragma_c`. You can still fall back to the legacy split logic without changing training code:

```bash
SPLIT_MODE=random bash scripts/train/run_ibm_aml_li_medium_pragma_c_pretrain.sh
```

When the current leakage-prevent preprocessing path is launched, processed data is written to:

```text
data/streaming/ibm_aml_li_medium_pragma_lite_full/
```

### Stage C Training Notes

- Stage C shards can contain batches where some DDP ranks have no masked MLM targets. The trainer now keeps all ranks on the same backward path by using a zero loss on target-free ranks instead of letting those ranks `continue`.
- Because of that safeguard, you may occasionally see `train_loss=-0.0000` in `train.log`. This is expected for the local rank on that step and does not mean the whole global step is invalid.
- A step is only fully skipped when **all** ranks have no supervised MLM targets. In that case the log prints `skipped=no_masked_targets`.
- Older Stage C runs may still contain `Grad strides do not match bucket view strides` warnings near the top of `train.log`. The current model code canonicalizes the grad layout for the learnable CLS tokens, so check the most recent launch block before assuming the warning is still active.
- If `train.log` contains multiple restarts, read it by launch block rather than assuming one continuous run. The newest run begins at the latest `Using TORCHRUN_BIN=...` line.

### Figure 4 Alignment

- The current leakage-prevent split and controls stay unchanged. Figure 4 alignment is applied on top of the existing Stage C temporal protocol, not by relaxing the temporal split.
- The current tokenizer now uses `tokenizer_version=2`, which stores `field_value_types`, `categorical_values`, `max_value_tokens_per_field`, and optional text tokenizer metadata in `tokenizer.json`.
- Structured encoding now routes fields by type: numeric fields use bucket tokens with an optional `#ZERO` bucket, categorical fields use field-specific `[UNK]`, and textual fields can expand to multiple shared `T:*` tokens.
- Multi-value fields now replicate the field key and increment `value_pos` as `0, 1, 2, ...`, which activates the existing within-field positional embedding path in the model.
- History time anchoring is now configurable during preprocessing: `evaluation` reproduces the original PRA-lite anchor, `last_event` follows PRAGMA's "time to last event" history anchor, and `decoupled` uses the PRAGMA anchor while also adding `seconds_since_last_event` as a numeric profile feature. The current default is `last_event`.
- The model now prepends `[USR]` and `[EVT]` by reading them directly from the shared token embedding table instead of maintaining separate learned CLS parameters.
- The model now uses a two-layer calendar projection MLP and can tie MLM logits to the shared token embedding table, which is closer to PRAGMA Figure 4 without changing the Stage C split semantics.
- History ablations can also disable the extra additive time projection and history order embedding through `use_additive_time_proj` and `use_history_order_emb` in the model config.
- Training now supports event-count bucketed token-budget dynamic batching. PRA-lite still uses rectangular padded tensors, but batches are no longer fixed-size: within each bucket, records are greedily packed under a token budget and collated to the batch-local max active lengths. This improves padding efficiency without implementing PRAGMA's varlen event-token packing kernel.
- Because `tokenizer_version=2` changes token ids and sequence layouts, new tokenized shards must be rebuilt under the leakage-prevent data root before running a clean comparison against older runs.

## Notes

- `--split all` runs on the full tokenized dataset without split filtering.
- If `--split_dir` is omitted, inference scripts try to infer it from `--data_dir`.
- Checkpoints store both model weights and the tokenizer path used during training.
