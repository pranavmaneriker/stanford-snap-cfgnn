#!/bin/bash
#SBATCH --account PAS2030
#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -J conformal_5_1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o /users/PAS2065/adityatv/jobs/slurm-out-%A.txt
#SBATCH -e /users/PAS2065/adityatv/jobs/slurm-err-%A.txt

. /users/PAS2065/adityatv/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate fairgraph

# __script_start__
cd $HOME/stanford-snap-cfgnn
srun python train.py \
    --model GCN \
    --dataset $1 \
    --device cuda \
    --alpha $2 \
    --optimal \
    --use_fixed_aps