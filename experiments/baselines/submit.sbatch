#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --job-name=baselines
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --cpu-freq=2000000

module load release/23.10  GCC/11.3.0  OpenMPI/4.1.4 Python

cd experiments/baselines

source PATH_TO_YOUR_PYTHON_ENV/bin/activate

python runBaselines.py
