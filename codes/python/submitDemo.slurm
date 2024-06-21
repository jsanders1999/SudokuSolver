#!/bin/sh
#
#SBATCH --job-name="Sudoku-Solver-Demo"
#SBATCH --partition=compute
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-EEMCS-MSc-AM

module load .2024rc1
modula load py-numpy/1.26.1
module load py-scipy/1.11.3
module load py-numba/0.57.0

srun python fastestSolverDemo.py