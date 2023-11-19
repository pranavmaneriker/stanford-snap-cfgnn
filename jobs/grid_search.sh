for dataset in Cora_ML_CF
do
    for alpha in 0.05 0.1 0.15 0.2 0.25 0.3
    do
        sbatch cfgnn.sh $dataset $alpha
    done
done
