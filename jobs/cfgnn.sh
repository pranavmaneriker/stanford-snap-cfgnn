#!/bin/bash

#SBATCH -N 1                    # use 2 nodes
#SBATCH -n 16                   # use 16 processes
#SBATCH -t 4:00:00             # 0 days, 4 hours, 0 minutes, 0 seconds
#SBATCH -p a100                 # use the a100 gpu partition
#SBATCH -J conformal_5_1_cora        # Job name
#SBATCH --mem=32000
#SBATCH -G 1                    # 1 GPU
#SBATCH -e /home/maneriker.1/conformalized-gnn/jobs/logs/%x_%j.err
#SBATCH -o /home/maneriker.1/conformalized-gnn/jobs/logs/%x_%j.out


module purge
source /home/maneriker.1/.bashrc

cd /home/maneriker.1/conformalized-gnn/
conda activate conformalized-gnn
srun python train.py \
    --model GCN \
    --dataset $1 \
    --device cuda \
    --alpha $2 \
    --use_fixed_aps \
    --optimal
