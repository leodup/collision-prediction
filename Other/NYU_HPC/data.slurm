#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2023sp_14

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=12:00:00
#SBATCH --output=logs/data_%j.out
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL23SP/dataset_v2.sqsh /tmp
echo "Dataset is copied to /tmp"

singularity exec --nv \
--bind /scratch \
--overlay /tmp/dataset_v2.sqsh \
/share/apps/images/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c "
python3 unlabeled.py
"
