#!/bin/bash
#SBATCH --job-name=gr00t-n1.5-ft-allex-bimanual-cube
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# srun --comment "gr00t training" --gpus=1 --nodes=1 --pty /bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t
echo "✅ Conda environment 'gr00t' activated."

mkdir -p tmp 2>/dev/null
mkdir -p checkpoints 2>/dev/null

python scripts/gr00t_finetune.py \
   --dataset-path ./demo_data/allex_cube \
   --num-gpus 1 \
   --output-dir checkpoints/allex-bimanual-cube  \
   --max-steps 10000 \
   --save-steps 5000 \
   --data-config allex_bimanual \
   --video-backend torchvision_av \
   --action_dim 42 \
   > tmp/slurm-$SLURM_JOB_ID-policy.log 2>&1

# --dataset-path 는 데이터 존재하는 path, 보통 demo_data 아래 있음
# --output-dir 는 생성될 checkpoint 폴더명
# --data-config 는 demo_data아래 있는 폴더명
