#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu 3GB
#SBATCH -J "TENSOR"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-mlpfits

source /pc2/users/u/usrtr200/load_GRACE.sh 
module load system/CUDA/13.0.0 

# ==========================================
# 1. TensorFlow GPU Memory Growth
# ==========================================
# Prevents TF from allocating 100% of the A100 memory on startup
export TF_FORCE_GPU_ALLOW_GROWTH=true

# ==========================================
# 2. CPU Threading Configuration
# ==========================================
# Limit OpenMP and NumPy backends to the requested 32 cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: Disable thread affinity if HPC system strictly pins threads
# export KMP_AFFINITY=disabled 

# ==========================================
# 3. Job Execution
# ==========================================
# Optional: confirm GPU is visible
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run your application
gracemaker