#!/bin/bash
#BSUB -J thesis
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -B
#BSUB -N
#BSUB -o codebase/dqn_slice_based/report/%J.out
#BSUB -e codebase/dqn_slice_based/report/%J.err

source /zhome/55/7/202529/env.sh

python -m dqn_slice_based.main_v2 \
    --train_dir data/ct_like/rapids-p/train \
    --val_dir data/ct_like/rapids-p/val \
    --epochs 50 \
    --episodes_per_epoch 10 \
    --slices_per_episode 64 \
    --model_type unet \
    --foreground_weight 2.0