# Scripts Layout

This directory is organized by function:

- `benchmarks/`: top-level benchmark entry points and reproducible runs
- `download/`: raw dataset download helpers
- `prepare/pragma_c/`: pragma-C and pragma-lite-full data preparation pipeline
- `prepare/streaming/`: streaming and LMDB data preparation helpers
- `train/`: pretraining launchers and shared training entry points
- `slurm/prepare/`: SLURM wrappers for data preparation
- `slurm/train/`: SLURM wrappers for training jobs
- `testing/`: smoke and validation helpers

Common entry points:

- `scripts/train/run_ibm_aml_li_medium_pragma_c_pretrain.sh`
- `scripts/train/run_ibm_aml_medium_streaming.sh`
- `scripts/benchmarks/run_ibm_aml_benchmark.sh`
- `scripts/benchmarks/run_transxion_benchmark.sh`
- `scripts/prepare/pragma_c/prepare_ibm_aml_li_pragma_lite_full.sh`
