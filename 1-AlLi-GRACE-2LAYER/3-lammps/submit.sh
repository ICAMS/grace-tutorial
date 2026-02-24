#!/bin/bash
#SBATCH -t 0:15:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu 32GB
#SBATCH -J "AlLi-2L-lammps"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1

source /pc2/users/u/usrtr200/load_GRACE.sh 

module load system/CUDA/13.0.0 mpi/OpenMPI/5.0.8-GCC-14.3.0

# Optional: confirm GPU is visible
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run your application here.
# Below is an example to run the "Hello World" program with LIKWID.
/pc2/users/u/usrtr200/mlpfits-2026-grace/soft/lammps/build/lmp -in in.lammps
/pc2/users/u/usrtr200/mlpfits-2026-grace/soft/lammps/build/lmp -in in.lammps.chunked