#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -J "AlLi-2L"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu
#SBATCH -A hpc-prf-mlpfits


source /pc2/users/u/usrtr200/load_GRACE.sh 

module load system/CUDA/13.0.0 

# Optional: confirm GPU is visible
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run your application here.
# Below is an example to run the "Hello World" program with LIKWID.
gracemaker --seed 2