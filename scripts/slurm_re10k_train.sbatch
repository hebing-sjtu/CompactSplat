#!/usr/bin/env bash
# CptSplat RE10K training on 2 nodes x 8 H200.
#
#   export REPO=/path/to/CompactSplat
#   export VENV=/path/to/venv
#   export DATA_ROOT=/path/to/re10k
#   mkdir -p slurm-logs        # SLURM opens the log before the script runs
#   sbatch --partition=<your-partition> scripts/slurm_re10k_train.sbatch
#
# Override any tunable from the command line, e.g.
#   sbatch --export=ALL,MAX_STEPS=150000,BATCH_SIZE=14 scripts/slurm_re10k_train.sbatch
#
# ---------------------------------------------------------------------------
# Why the resource request looks like this
#
# src/main.py builds the Trainer with
#     devices=torch.cuda.device_count()
#     num_nodes=cfg.trainer.num_nodes
#     strategy='ddp'
# and under srun Lightning auto-detects SLURMEnvironment, which maps
# SLURM_LOCALID -> local_rank and then picks cuda:local_rank.
#
# So every task must see ALL 8 GPUs on its node: device_count() has to equal
# --ntasks-per-node. That means --gres=gpu:8, and explicitly NOT
# --gpus-per-task=1 -- with per-task binding each task sees one GPU,
# device_count() returns 1, and Lightning aborts on the devices/ntasks mismatch.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cptsplat-re10k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.out
# Lightning's SLURMEnvironment requeues on SIGUSR1, which together with
# checkpointing.resume=true lets a 300k-step run survive the wall clock limit.
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue

set -euo pipefail

# --- paths (must resolve identically on BOTH nodes) ------------------------
: "${REPO:?Set REPO to the shared repository directory}"
: "${VENV:?Set VENV to the shared Python environment}"
# Checkpoints and the resume scan both live here, so it has to be shared storage.
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/re10k}"
# datasets/re10k/{train,test}/*.torch + index.json
: "${DATA_ROOT:?Set DATA_ROOT to the prepared RealEstate10K dataset}"

# --- training scale -------------------------------------------------------
# DepthSplat's README asks that (num_gpus x batch_size x max_steps) be held
# constant when changing the GPU count. config/experiment/re10k.yaml ships
# batch_size=14, max_steps=300_001 for a single node. Going 8 -> 16 GPUs
# therefore halves the step count. Adjust if your reference run differed.
BATCH_SIZE="${BATCH_SIZE:-14}"
MAX_STEPS="${MAX_STEPS:-150000}"
VIT="${VIT:-vitl}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

mkdir -p slurm-logs "$OUTPUT_DIR"

# --- preflight ------------------------------------------------------------
# All 16 ranks would otherwise fail a couple of minutes in, after NCCL init, with
# a much less obvious error.
MONODEPTH="$REPO/pretrained/depth_anything_v2_$VIT.pth"
[[ -s "$MONODEPTH" ]] || {
  echo "!! missing $MONODEPTH"
  echo "   run: bash scripts/download_pretrained.sh $VIT"
  exit 1
}
[[ -s "$REPO/pretrained/gmflow-scale1-things-e9887eda.pth" ]] || {
  echo "!! missing pretrained/gmflow-scale1-things-e9887eda.pth"
  echo "   run: bash scripts/download_pretrained.sh $VIT"
  exit 1
}
[[ -d "$DATA_ROOT/train" && -f "$DATA_ROOT/train/index.json" ]] || {
  echo "!! $DATA_ROOT/train is not a prepared RE10K chunk directory"
  echo "   expected $DATA_ROOT/train/{000000.torch,...,index.json}"
  exit 1
}

# --- environment ----------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO"

export XFORMERS_DISABLED=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$NUM_WORKERS"
export TOKENIZERS_PARALLELISM=false

# Derive a per-job port so concurrent jobs cannot collide on the rendezvous.
# Lightning's SLURMEnvironment reads MASTER_PORT when it is set.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# Cluster-specific fabric selection. If the job hangs at NCCL init, this is the
# first thing to set -- pick the interface that carries inter-node traffic.
# export NCCL_SOCKET_IFNAME=ibp
# export NCCL_IB_HCA=mlx5

echo "=== job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST"
echo "=== tasks/node=$SLURM_NTASKS_PER_NODE  world=$SLURM_NTASKS"
echo "=== batch_size=$BATCH_SIZE/gpu  max_steps=$MAX_STEPS  vit=$VIT"
echo "=== effective samples = $SLURM_NTASKS x $BATCH_SIZE x $MAX_STEPS = $((SLURM_NTASKS * BATCH_SIZE * MAX_STEPS))"
echo "=== output_dir=$OUTPUT_DIR"

# --- launch ---------------------------------------------------------------
# srun starts one process per task; Lightning must NOT spawn its own, which is
# why use_plugins stays false (use_plugins=true swaps in LightningEnvironment
# and would break SLURM rank discovery).
#
# hydra.output_subdir=null stops all 16 ranks racing to write the same
# .hydra/config.yaml under a shared run directory.
srun --kill-on-bad-exit=1 python -m src.main +experiment=re10k \
  trainer.num_nodes="$SLURM_JOB_NUM_NODES" \
  trainer.max_steps="$MAX_STEPS" \
  data_loader.train.batch_size="$BATCH_SIZE" \
  data_loader.train.num_workers="$NUM_WORKERS" \
  dataset.roots="[$DATA_ROOT]" \
  dataset.test_chunk_interval=10 \
  model.encoder.num_scales=2 \
  model.encoder.upsample_factor=2 \
  model.encoder.lowest_feature_resolution=4 \
  model.encoder.monodepth_vit_type="$VIT" \
  checkpointing.pretrained_monodepth="pretrained/depth_anything_v2_$VIT.pth" \
  checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
  checkpointing.resume=true \
  use_plugins=false \
  output_dir="$OUTPUT_DIR" \
  hydra.output_subdir=null
