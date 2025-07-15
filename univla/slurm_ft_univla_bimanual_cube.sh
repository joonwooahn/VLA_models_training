#!/bin/bash

#SBATCH --job-name=univla-ft-allex-bimanual-cube
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# srun --comment "univla training" --gpus=1 --nodes=1 --pty /bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate univla_train
echo "✅ Conda environment 'univla_train' activated."

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla_scripts/finetune_rlwrld.py \
    --data_root_dir "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/allex-cube-dataset_single_view_converted_state_action" \
    --batch_size 16 \
    --max_steps 20000 \
    --run_id_note "allex_state_action_filter_side_view" \

   # --data_root_dir 는 데이터 존재하는 path, 보통 dataset 아래 있음
   # --run_id_note 는 checkout 이름 라벨 (optional)