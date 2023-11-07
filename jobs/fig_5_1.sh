#!/bin/bash

#SBATCH -N 1                    # use 2 nodes
#SBATCH -n 16                   # use 16 processes
#SBATCH -t 01:00:00             # 0 days, 1 hours, 0 minutes, 0 seconds
#SBATCH -p a100                 # use the a100 gpu partition
#SBATCH -J conformal_5_1_cora        # Job name
#SBATCH --mem=32000
#SBATCH -G 1                    # 1 GPU
#SBATCH -e /home/maneriker.1/conformalized-gnn/jobs/logs/%x_%j.err
#SBATCH -o /home/maneriker.1/conformalized-gnn/jobs/logs/%x_%j.out
                                # scontrol show partition shows options
# clear any modules for a clean env
module purge
source /home/maneriker.1/.bashrc

cd /home/maneriker.1/conformalized-gnn/
conda activate conformalized-gnn
python train.py --model GCN --dataset $data --device cuda --optimal --alpha $alpha
