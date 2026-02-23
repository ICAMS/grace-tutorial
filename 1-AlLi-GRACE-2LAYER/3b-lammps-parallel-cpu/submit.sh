#!/bin/bash
#SBATCH -t 0:15:00
#SBATCH -N 2                   # Number of nodes
#SBATCH --ntasks-per-node=4    # Number of MPI ranks per node
#SBATCH -n 8                   # Total number of MPI ranks (2 nodes * 4 ranks/node)
#SBATCH --cpus-per-task=1      # Number of CPUs per MPI rank
#SBATCH --mem-per-cpu=32GB     # Memory per CPU core
#SBATCH -J "cpu-AlLi-2L-lammps"
#SBATCH -p normal

source /pc2/users/u/usrtr200/load_GRACE.sh 

module load system/CUDA/13.0.0 mpi/OpenMPI/5.0.8-GCC-14.3.0

mpirun /pc2/users/u/usrtr200/mlpfits-2026-grace/soft/lammps/build/lmp -in in.lammps