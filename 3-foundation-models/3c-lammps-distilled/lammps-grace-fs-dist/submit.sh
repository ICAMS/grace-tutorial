#!/bin/bash
#SBATCH -t 0:15:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu 32GB
#SBATCH -J "dist-fs-lmp"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-mlpfits

source /pc2/users/u/usrtr200/load_GRACE.sh 

module load system/CUDA/13.0.0 mpi/OpenMPI/5.0.8-GCC-14.3.0

# Optional: confirm GPU is visible

# Run your application here.
lmp_kk -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps 
