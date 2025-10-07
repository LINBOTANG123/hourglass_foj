#!/bin/bash
#SBATCH --job-name=hourglass
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=70:00:00
#SBATCH --output=logs/denoise_%j.out
#SBATCH --error=logs/denoise_%j.err

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hourglass

# (Optional) If you use wandb, set your key here or rely on prior `wandb login`
export WANDB_API_KEY=9bcfe4681805f0f644ba969c52fe0ac821b6e768
export WANDB_DIR=./wandb
export WANDB_MODE=online   # or offline



# ── 3. Run training ──────────────────────────────────────────────────
python train.py \
  --config configs/foj_transformer_v2_udf_10k_imgtok_withsamecolor.json \
  --batch-size 32 \
  --checkpointing \
  --start-method fork \
  --num-workers 4 \
  --name foj_diffusion_imgtok_withsamecolor \
  --evaluate-every 10000 \
  --demo-every 100000 \
  --end-step 10000000 \
  --wandb-project foj_diffusion_img_tok \
  --wandb-entity linbotang0204
