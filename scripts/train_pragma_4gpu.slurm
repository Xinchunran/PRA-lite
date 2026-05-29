#!/bin/bash
#SBATCH --job-name=pragma-ddp
#SBATCH --partition=batch_gpu
#SBATCH --account=csb_gpu_acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home/ranx/PRAGMA/logs/%x-%j.out
#SBATCH --error=/home/ranx/PRAGMA/logs/%x-%j.err

set -euo pipefail

cd /home/ranx/PRAGMA
mkdir -p logs

source ~/.bashrc
conda activate pragma-lite

export PYTHONPATH=/home/ranx/PRAGMA
export CUDA_VISIBLE_DEVICES=0,1,2,3

NPROC_PER_NODE=4 \
CONFIG=configs/train/pretrain_mlm.yaml \
MODEL_CONFIG=configs/model/pragma_lite_small.yaml \
DATA_DIR=runs/smoke_test/data/processed/transxion/tokenized \
SPLIT_DIR=runs/smoke_test/data/splits/transxion \
TOKENIZER_DIR=runs/smoke_test/data/processed/transxion/tokenizer \
OUTPUT_DIR=runs/pretrain_ddp_4gpu_smoke \
bash scripts/train_pretrain_ddp.sh
