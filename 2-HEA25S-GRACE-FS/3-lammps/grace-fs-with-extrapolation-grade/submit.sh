#!/bin/bash
#SBATCH -t 0:15:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -J "HEA25-FS-lammps"
#SBATCH -p normal
#SBATCH -A hpc-prf-mlpfits

source /pc2/users/u/usrtr200/load_GRACE.sh 


# Add the GCC module before OpenMPI
module load compiler/GCC/14.3.0 system/CUDA/13.0.0 mpi/OpenMPI/5.0.8-GCC-14.3.0

# Optional: confirm GPU is visible

# Run your application here.
/pc2/groups/hpc-prf-mlpfits/GRACE/bin/lmp -in in.lammps