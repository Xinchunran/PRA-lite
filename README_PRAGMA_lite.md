# PRAGMA-lite

PRAGMA-lite is an open, research-oriented implementation plan for a PRAGMA-inspired model for financial event sequences.

The goal is not to reproduce the proprietary Revolut PRAGMA model exactly. Instead, this project implements a lightweight, reproducible framework for studying the same core ideas:

- key-value-time tokenization for heterogeneous financial records;
- masked modelling over event fields and full events;
- separate profile, event, and history encoders;
- downstream evaluation on fraud, AML, risk, and tabular-sequence prediction tasks;
- comparison against strong tabular and sequence baselines.

## Background

PRAGMA is described as a foundation model for multi-source banking event sequences. It pre-trains a Transformer-based encoder with masked modelling on heterogeneous banking event records and adapts the resulting embeddings to downstream tasks such as credit scoring, fraud detection, lifetime value prediction, recurrent transaction detection, communication engagement, and product recommendation.

This repository implements a PRAGMA-inspired experimental framework, not the original PRAGMA model, dataset, tokenizer, or weights.

Useful references:

- PRAGMA: Revolut Foundation Model: https://arxiv.org/abs/2604.08649
- TabFormer: Tabular Transformers for Modeling Multivariate Time Series: https://arxiv.org/abs/2011.01843
- IBM TabFormer codebase: https://github.com/IBM/TabFormer
- TransXion AML benchmark: https://arxiv.org/abs/2604.17420
- TransXion codebase: https://github.com/chaos-max/TransXion

## Project Structure

```text
pragma-lite/
├── configs/
│   ├── data/
│   │   ├── transxion.yaml
│   │   ├── tabformer.yaml
│   │   └── census_income.yaml
│   ├── model/
│   │   ├── pragma_lite_small.yaml
│   │   ├── pragma_lite_base.yaml
│   │   ├── ft_transformer.yaml
│   │   ├── tabformer_bert.yaml
│   │   ├── lightgbm.yaml
│   │   └── catboost.yaml
│   └── train/
│       ├── pretrain_mlm.yaml
│       ├── finetune_binary.yaml
│       └── linear_probe.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data_downloader/
│   ├── splitter/
│   ├── tokenizer/
│   ├── model/
│   │   ├── pragma_lite/
│   │   └── baseline/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── results/
│   └── plot/
├── tests/
├── scripts/
└── README.md
```

## Installation

```bash
conda create -n pragma-lite python=3.10 -y
conda activate pragma-lite
pip install -r requirements.txt
```

Suggested core dependencies:

```text
torch
pytorch-lightning
transformers
pandas
numpy
scikit-learn
pyarrow
lightgbm
catboost
xgboost
matplotlib
seaborn
tqdm
omegaconf
hydra-core
pytest
```

Optional dependencies:

```text
wandb
mlflow
faiss-cpu
polars
duckdb
```

---

# 1. data_downloader

The `data_downloader` module downloads or prepares datasets used in PRAGMA-lite experiments.

Supported datasets:

| Dataset | Type | Recommended Usage |
|---|---|---|
| `transxion` | transaction sequence + profiles | main PRAGMA-lite reproduction dataset |
| `tabformer` | synthetic credit-card transaction sequence | engineering reference / sequence baseline |
| `census_income` | fixed-schema tabular rows | non-sequential sanity check only |

## 1.1 Download TransXion

TransXion is the recommended primary dataset because it contains transaction histories, timestamps, entity attributes, and AML labels.

```bash
python -m src.data_downloader.download \
  --dataset transxion \
  --output_dir data/raw/transxion
```

Expected raw files:

```text
data/raw/transxion/
├── accounts.csv
├── persons.csv
├── merchants.csv
├── transactions.csv
└── metadata.json
```

If the upstream filenames differ, define the mapping in:

```text
configs/data/transxion.yaml
```

Example configuration:

```yaml
dataset: transxion
raw_dir: data/raw/transxion
processed_dir: data/processed/transxion

entity_id_col: account_id
timestamp_col: timestamp
label_col: is_laundering

profile_files:
  - persons.csv
  - merchants.csv

transaction_file: transactions.csv

transaction_columns:
  - timestamp
  - sender_id
  - receiver_id
  - amount
  - currency
  - payment_format
  - sender_bank
  - receiver_bank
  - is_laundering
```

## 1.2 Download TabFormer Dataset

TabFormer can be used as an engineering reference because it already follows the tabular time-series Transformer setting.

```bash
python -m src.data_downloader.download \
  --dataset tabformer \
  --output_dir data/raw/tabformer
```

Expected raw files:

```text
data/raw/tabformer/
├── card_transaction.v1.csv
└── metadata.json
```

## 1.3 Prepare Census-Income

Census-Income is not a temporal event-sequence dataset. It should only be used to test whether the tokenizer and tabular baselines run correctly.

```bash
python -m src.data_downloader.prepare_local \
  --dataset census_income \
  --columns_file data/raw/census_income/census-bureau.columns \
  --data_file data/raw/census_income/census-bureau.data \
  --output_dir data/processed/census_income
```

## 1.4 Convert Raw Data to Event Format

For PRAGMA-lite, every dataset must be converted into a common event schema:

```json
{
  "entity_id": "A123",
  "evaluation_time": "2024-01-31T23:59:59",
  "profile": {
    "age_bucket": "35_44",
    "entity_type": "person",
    "region": "UK"
  },
  "events": [
    {
      "timestamp": "2024-01-01T12:30:00",
      "fields": {
        "event_type": "transaction",
        "amount": 120.50,
        "currency": "GBP",
        "payment_format": "card",
        "receiver_type": "merchant"
      }
    }
  ],
  "label": 0
}
```

Run:

```bash
python -m src.data_downloader.build_events \
  --config configs/data/transxion.yaml
```

Output:

```text
data/processed/transxion/
├── events.parquet
├── profiles.parquet
├── labels.parquet
├── vocab_stats.json
└── schema.json
```

---

# 2. splitter

The `splitter` module creates train, validation, and test splits.

## 2.1 Split Modes

Supported split modes:

| Split Mode | Description | Recommended For |
|---|---|---|
| `entity` | split by account/entity id | avoids leakage across entity histories |
| `time` | train on earlier period, test on later period | deployment-like evaluation |
| `stratified` | preserve label distribution | small datasets |
| `random` | random record split | debugging only |

For PRAGMA-lite, the default should be entity-level splitting.

## 2.2 Create Entity-Level Splits

```bash
python -m src.splitter.make_splits \
  --input_dir data/processed/transxion \
  --output_dir data/splits/transxion \
  --split_mode entity \
  --train_size 0.70 \
  --valid_size 0.15 \
  --test_size 0.15 \
  --label_col is_laundering \
  --seed 42
```

Output:

```text
data/splits/transxion/
├── train_ids.txt
├── valid_ids.txt
├── test_ids.txt
└── split_summary.json
```

## 2.3 Create Time-Based Splits

```bash
python -m src.splitter.make_splits \
  --input_dir data/processed/transxion \
  --output_dir data/splits/transxion_time \
  --split_mode time \
  --train_end "2024-08-31" \
  --valid_end "2024-10-31" \
  --test_end "2024-12-31"
```

## 2.4 Split Validation

Always inspect split quality:

```bash
python -m src.splitter.check_splits \
  --processed_dir data/processed/transxion \
  --split_dir data/splits/transxion
```

Expected checks:

- no entity leakage across splits;
- no duplicate event histories;
- stable label distribution;
- reasonable timestamp distribution;
- class imbalance report;
- profile and event coverage report.

---

# 3. model

This repository contains two model families:

1. `PragmaLite`: PRAGMA-inspired key-value-time event encoder.
2. `baseline`: FT-Transformer, TabFormer-style BERT, LightGBM, CatBoost, XGBoost, and logistic regression.

---

## 3.1 PRAGMA-lite

PRAGMA-lite encodes each account or user record as:

```text
profile state + ordered event history
```

Each profile field or event field is represented as:

```text
key token + value token + time embedding
```

### 3.1.1 Tokenization

PRAGMA-lite supports:

| Field Type | Tokenization |
|---|---|
| categorical | vocabulary id |
| numerical | percentile bucket id |
| text | BPE/subword token id or hashed token |
| timestamp | log-time delta + calendar features |
| missing | `[MISSING]` token |
| masked | `[MASK]` token |
| unknown | `[UNK]` token |

Build vocabulary:

```bash
python -m src.tokenizer.build_vocab \
  --processed_dir data/processed/transxion \
  --output_dir data/processed/transxion/tokenizer \
  --num_buckets 100 \
  --min_freq 5
```

Tokenize dataset:

```bash
python -m src.tokenizer.encode_dataset \
  --processed_dir data/processed/transxion \
  --tokenizer_dir data/processed/transxion/tokenizer \
  --output_dir data/processed/transxion/tokenized \
  --max_events 512 \
  --max_event_tokens 24 \
  --max_profile_tokens 200
```

### 3.1.2 Architecture

PRAGMA-lite contains three Transformer encoders:

```text
ProfileStateEncoder:
  encodes static/profile attributes

EventEncoder:
  encodes each event independently from key-value tokens

HistoryEncoder:
  encodes the ordered sequence of [USR] and [EVT] embeddings
```

Default small configuration:

```yaml
model:
  name: pragma_lite_small

  d_model: 192
  d_ffn: 768
  n_heads: 3
  dropout: 0.1
  activation: gelu
  norm: pre_norm

  profile_layers: 1
  event_layers: 4
  history_layers: 2

  max_profile_tokens: 200
  max_event_tokens: 24
  max_events: 512

  time_encoding:
    use_log_delta: true
    use_calendar_features: true
    use_rope: true

  pooling:
    type: usr_token
```

### 3.1.3 Self-Supervised Pretraining

PRAGMA-lite uses masked event modelling.

Masking options:

| Mask Type | Default Probability |
|---|---:|
| token-level value masking | 0.15 |
| full-event masking | 0.10 |
| key-level masking | 0.10 |
| random `[UNK]` corruption | 0.05 |

Run pretraining:

```bash
python -m src.training.pretrain_mlm \
  --config configs/train/pretrain_mlm.yaml \
  --model_config configs/model/pragma_lite_small.yaml \
  --data_dir data/processed/transxion/tokenized \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/pragma_lite_small_mlm
```

Example pretraining config:

```yaml
training:
  task: masked_event_modeling
  batch_size: 128
  max_steps: 100000
  learning_rate: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 5000
  precision: bf16
  gradient_clip_val: 1.0
  optimizer: adamw
  scheduler: cosine

masking:
  token_mask_prob: 0.15
  event_mask_prob: 0.10
  key_mask_prob: 0.10
  unk_replace_prob: 0.05

loss:
  type: cross_entropy
  label_smoothing: 0.05
```

### 3.1.4 Fine-Tuning

Fine-tune on binary AML or fraud labels:

```bash
python -m src.training.finetune \
  --config configs/train/finetune_binary.yaml \
  --checkpoint checkpoints/pragma_lite_small_mlm/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/pragma_lite_small_finetune
```

Fine-tuning config:

```yaml
training:
  task: binary_classification
  batch_size: 128
  max_epochs: 20
  learning_rate: 1.0e-4
  weight_decay: 0.01
  optimizer: adamw
  scheduler: cosine
  early_stopping_metric: valid_pr_auc
  early_stopping_patience: 5

loss:
  type: bce_with_logits
  pos_weight: auto
```

### 3.1.5 Linear Probe

Freeze the pretrained backbone and train a linear classifier on top of `[USR]` embeddings:

```bash
python -m src.training.linear_probe \
  --checkpoint checkpoints/pragma_lite_small_mlm/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/pragma_lite_small_probe
```

---

## 3.2 Baseline Models

### 3.2.1 FT-Transformer

FT-Transformer is used as a fixed-schema tabular baseline. For sequence datasets, first aggregate event histories into account-level features.

Build aggregate features:

```bash
python -m src.model.baseline.build_aggregate_features \
  --processed_dir data/processed/transxion \
  --split_dir data/splits/transxion \
  --output_dir data/processed/transxion/aggregate_features
```

Train FT-Transformer:

```bash
python -m src.model.baseline.train_ft_transformer \
  --config configs/model/ft_transformer.yaml \
  --data_dir data/processed/transxion/aggregate_features \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/ft_transformer
```

### 3.2.2 TabFormer-Style BERT

Train a TabFormer-style sequence Transformer baseline:

```bash
python -m src.model.baseline.train_tabformer_bert \
  --config configs/model/tabformer_bert.yaml \
  --data_dir data/processed/transxion/tokenized \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/tabformer_bert
```

### 3.2.3 LightGBM

```bash
python -m src.model.baseline.train_lightgbm \
  --config configs/model/lightgbm.yaml \
  --data_dir data/processed/transxion/aggregate_features \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/lightgbm
```

### 3.2.4 CatBoost

```bash
python -m src.model.baseline.train_catboost \
  --config configs/model/catboost.yaml \
  --data_dir data/processed/transxion/aggregate_features \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/catboost
```

### 3.2.5 Logistic Regression

```bash
python -m src.model.baseline.train_logreg \
  --data_dir data/processed/transxion/aggregate_features \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/logreg
```

---

# 4. test

The `tests/` folder contains unit tests and integration tests.

## 4.1 Run All Tests

```bash
pytest tests/ -v
```

## 4.2 Test Data Loading

```bash
pytest tests/test_data_downloader.py -v
pytest tests/test_event_builder.py -v
```

Expected coverage:

- raw CSV loading;
- event schema construction;
- profile-event joining;
- timestamp parsing;
- missing-value handling;
- label extraction.

## 4.3 Test Splitter

```bash
pytest tests/test_splitter.py -v
```

Expected coverage:

- no entity leakage;
- correct train/valid/test ratio;
- stable label distribution;
- deterministic split with seed.

## 4.4 Test Tokenizer

```bash
pytest tests/test_tokenizer.py -v
```

Expected coverage:

- categorical vocabulary mapping;
- numerical bucketization;
- missing-token handling;
- mask-token handling;
- timestamp feature creation;
- max-event and max-token truncation.

## 4.5 Test PRAGMA-lite Forward Pass

```bash
pytest tests/test_pragma_lite_model.py -v
```

Expected coverage:

- profile encoder shape;
- event encoder shape;
- history encoder shape;
- MLM head output shape;
- binary classifier output shape;
- masking loss is finite;
- inference works with batch size 1.

## 4.6 Smoke Test

Run a small end-to-end experiment on 1,000 records:

```bash
python -m scripts.smoke_test \
  --dataset transxion \
  --num_records 1000 \
  --output_dir runs/smoke_test
```

---

# 5. inference

The `inference` module loads trained checkpoints and produces predictions or embeddings.

## 5.1 Predict Labels

```bash
python -m src.inference.predict \
  --checkpoint checkpoints/pragma_lite_small_finetune/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split test \
  --output_file outputs/predictions/pragma_lite_test_predictions.parquet
```

Output schema:

```text
entity_id
record_id
timestamp
label
probability
prediction
```

## 5.2 Extract Embeddings

```bash
python -m src.inference.extract_embeddings \
  --checkpoint checkpoints/pragma_lite_small_mlm/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split test \
  --pooling usr_token \
  --output_file outputs/embeddings/pragma_lite_test_embeddings.parquet
```

Output schema:

```text
entity_id
record_id
embedding_0
embedding_1
...
embedding_d
label
```

## 5.3 Batch Inference

```bash
python -m src.inference.predict \
  --checkpoint checkpoints/pragma_lite_small_finetune/best.ckpt \
  --input_file data/processed/transxion/tokenized/test.parquet \
  --batch_size 256 \
  --output_file outputs/predictions/test.parquet
```

## 5.4 Single-Record Inference

```bash
python -m src.inference.predict_one \
  --checkpoint checkpoints/pragma_lite_small_finetune/best.ckpt \
  --input_json examples/single_record.json
```

Example input:

```json
{
  "entity_id": "A123",
  "profile": {
    "entity_type": "person",
    "region": "UK"
  },
  "events": [
    {
      "timestamp": "2024-01-01T12:30:00",
      "fields": {
        "amount": 120.5,
        "currency": "GBP",
        "payment_format": "card"
      }
    }
  ]
}
```

---

# 6. results

The `results` module computes metrics and generates experiment summaries.

## 6.1 Evaluate One Model

```bash
python -m src.results.evaluate \
  --prediction_file outputs/predictions/pragma_lite_test_predictions.parquet \
  --output_file outputs/results/pragma_lite_metrics.json
```

Metrics:

| Metric | Use Case |
|---|---|
| ROC-AUC | general binary ranking |
| PR-AUC | imbalanced fraud/AML detection |
| F1 | thresholded classification |
| F0.5 | precision-oriented AML screening |
| precision | alert quality |
| recall | suspicious-case coverage |
| balanced accuracy | imbalanced classification |
| log loss | probability quality |
| Brier score | calibration |
| ECE | calibration |

## 6.2 Compare Models

```bash
python -m src.results.compare \
  --prediction_files \
    outputs/predictions/pragma_lite_test_predictions.parquet \
    outputs/predictions/tabformer_bert_test_predictions.parquet \
    outputs/predictions/ft_transformer_test_predictions.parquet \
    outputs/predictions/lightgbm_test_predictions.parquet \
    outputs/predictions/catboost_test_predictions.parquet \
  --model_names \
    pragma_lite \
    tabformer_bert \
    ft_transformer \
    lightgbm \
    catboost \
  --output_file outputs/results/model_comparison.csv
```

Output:

```text
outputs/results/
├── pragma_lite_metrics.json
├── tabformer_bert_metrics.json
├── ft_transformer_metrics.json
├── lightgbm_metrics.json
├── catboost_metrics.json
└── model_comparison.csv
```

## 6.3 Example Result Table

| Model | ROC-AUC | PR-AUC | F1 | F0.5 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| FT-Transformer | TBD | TBD | TBD | TBD | TBD | TBD |
| TabFormer-BERT | TBD | TBD | TBD | TBD | TBD | TBD |
| PRAGMA-lite Scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| PRAGMA-lite MLM + Fine-tune | TBD | TBD | TBD | TBD | TBD | TBD |
| PRAGMA-lite MLM + Linear Probe | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.4 Recommended Experiment Matrix

| Experiment | Purpose |
|---|---|
| PRAGMA-lite scratch | measures architecture value without pretraining |
| PRAGMA-lite MLM + fine-tune | measures benefit of masked event pretraining |
| PRAGMA-lite MLM + linear probe | measures representation quality |
| PRAGMA-lite without profile encoder | profile-state ablation |
| PRAGMA-lite without time encoding | temporal encoding ablation |
| PRAGMA-lite without event masking | masking-strategy ablation |
| TabFormer-BERT | open tabular-sequence Transformer baseline |
| FT-Transformer | fixed-schema deep tabular baseline |
| LightGBM / CatBoost | strong classical tabular baselines |

---

# 7. plot

The `plot` module creates figures for reports and papers.

## 7.1 Plot Metric Comparison

```bash
python -m src.plot.plot_metrics \
  --comparison_file outputs/results/model_comparison.csv \
  --metric pr_auc \
  --output_file outputs/plots/pr_auc_comparison.png
```

## 7.2 Plot ROC Curve

```bash
python -m src.plot.plot_roc \
  --prediction_files \
    outputs/predictions/pragma_lite_test_predictions.parquet \
    outputs/predictions/lightgbm_test_predictions.parquet \
    outputs/predictions/catboost_test_predictions.parquet \
  --model_names \
    pragma_lite \
    lightgbm \
    catboost \
  --output_file outputs/plots/roc_curve.png
```

## 7.3 Plot Precision-Recall Curve

```bash
python -m src.plot.plot_pr \
  --prediction_files \
    outputs/predictions/pragma_lite_test_predictions.parquet \
    outputs/predictions/lightgbm_test_predictions.parquet \
    outputs/predictions/catboost_test_predictions.parquet \
  --model_names \
    pragma_lite \
    lightgbm \
    catboost \
  --output_file outputs/plots/pr_curve.png
```

## 7.4 Plot Calibration Curve

```bash
python -m src.plot.plot_calibration \
  --prediction_file outputs/predictions/pragma_lite_test_predictions.parquet \
  --output_file outputs/plots/calibration_curve.png
```

## 7.5 Plot Ablation Results

```bash
python -m src.plot.plot_ablation \
  --ablation_file outputs/results/ablation_results.csv \
  --metric pr_auc \
  --output_file outputs/plots/ablation_pr_auc.png
```

Expected plots:

```text
outputs/plots/
├── pr_auc_comparison.png
├── roc_auc_comparison.png
├── roc_curve.png
├── pr_curve.png
├── calibration_curve.png
├── ablation_pr_auc.png
└── confusion_matrix.png
```

---

# Recommended End-to-End Pipeline

## Step 1: Download and Build Data

```bash
python -m src.data_downloader.download \
  --dataset transxion \
  --output_dir data/raw/transxion

python -m src.data_downloader.build_events \
  --config configs/data/transxion.yaml
```

## Step 2: Split

```bash
python -m src.splitter.make_splits \
  --input_dir data/processed/transxion \
  --output_dir data/splits/transxion \
  --split_mode entity \
  --train_size 0.70 \
  --valid_size 0.15 \
  --test_size 0.15 \
  --seed 42
```

## Step 3: Tokenize

```bash
python -m src.tokenizer.build_vocab \
  --processed_dir data/processed/transxion \
  --output_dir data/processed/transxion/tokenizer

python -m src.tokenizer.encode_dataset \
  --processed_dir data/processed/transxion \
  --tokenizer_dir data/processed/transxion/tokenizer \
  --output_dir data/processed/transxion/tokenized
```

## Step 4: Pretrain PRAGMA-lite

```bash
python -m src.training.pretrain_mlm \
  --config configs/train/pretrain_mlm.yaml \
  --model_config configs/model/pragma_lite_small.yaml \
  --data_dir data/processed/transxion/tokenized \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/pragma_lite_small_mlm
```

## Step 5: Fine-Tune

```bash
python -m src.training.finetune \
  --config configs/train/finetune_binary.yaml \
  --checkpoint checkpoints/pragma_lite_small_mlm/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/pragma_lite_small_finetune
```

## Step 6: Train Baselines

```bash
python -m src.model.baseline.build_aggregate_features \
  --processed_dir data/processed/transxion \
  --split_dir data/splits/transxion \
  --output_dir data/processed/transxion/aggregate_features

python -m src.model.baseline.train_lightgbm \
  --config configs/model/lightgbm.yaml \
  --data_dir data/processed/transxion/aggregate_features \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/lightgbm

python -m src.model.baseline.train_catboost \
  --config configs/model/catboost.yaml \
  --data_dir data/processed/transxion/aggregate_features \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/catboost

python -m src.model.baseline.train_ft_transformer \
  --config configs/model/ft_transformer.yaml \
  --data_dir data/processed/transxion/aggregate_features \
  --split_dir data/splits/transxion \
  --output_dir checkpoints/ft_transformer
```

## Step 7: Inference

```bash
python -m src.inference.predict \
  --checkpoint checkpoints/pragma_lite_small_finetune/best.ckpt \
  --data_dir data/processed/transxion/tokenized \
  --split test \
  --output_file outputs/predictions/pragma_lite_test_predictions.parquet
```

## Step 8: Evaluate

```bash
python -m src.results.evaluate \
  --prediction_file outputs/predictions/pragma_lite_test_predictions.parquet \
  --output_file outputs/results/pragma_lite_metrics.json
```

## Step 9: Plot

```bash
python -m src.plot.plot_metrics \
  --comparison_file outputs/results/model_comparison.csv \
  --metric pr_auc \
  --output_file outputs/plots/pr_auc_comparison.png
```

---

# Notes on Census-Income

Census-Income is useful for debugging fixed-schema tabular modelling, but it should not be treated as a PRAGMA reproduction dataset. It does not contain event histories, timestamps, user trajectories, or multi-source financial records.

Recommended Census-Income use:

```text
Tokenizer sanity check
FT-Transformer baseline
LightGBM / CatBoost benchmark
Row-level masked feature modelling
```

Not recommended:

```text
PRAGMA-style event-history modelling
TabFormer-style transaction sequence modelling
Temporal representation learning
```

---

# Citation

If you use this repository, cite the original methodological inspirations:

- PRAGMA: Revolut Foundation Model
- TabFormer: Tabular Transformers for Modeling Multivariate Time Series
- TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering

