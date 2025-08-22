#!/bin/bash
#SBATCH --job-name=hourglass
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=70:00:00
#SBATCH --output=logs/denoise_%j.out
#SBATCH --error=logs/denoise_%j.err

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hourglass


# ── 3. Run training ──────────────────────────────────────────────────
python train.py \
  --config configs/foj_transformer_v2_one_obj.json \
  --batch-size 32 \
  --checkpointing \
  --evaluate-every 0 \
  --start-method fork \
  --num-workers 4 \
  --name foj_diffusion_one_obj \
  --demo-every 10000 \
  --end-step 10000000
