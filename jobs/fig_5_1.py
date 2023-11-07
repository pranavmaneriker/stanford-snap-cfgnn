import os
import sys
import subprocess

if __name__ == "__main__":
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        os.environ["data"] = "Cora_ML_CF"
        os.environ["alpha"] = str(alpha)
        subprocess.call(["sbatch", "fig_5_1.sh"])

