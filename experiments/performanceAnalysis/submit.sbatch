#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --job-name=performanceAnalysis
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --cpu-freq=2000000

module load release/23.10  GCC/11.3.0  OpenMPI/4.1.4 Python Score-P
cd experiments/benchmark

source PATH_TO_YOUR_PYTHON_ENV/bin/activate
export PYTHONPATH=PATH_TO_SCOREP_PYTHON/scorep:$PYTHONPATH

export SCOREP_ENABLE_TRACING=1
export SCOREP_ENABLE_PROFILING=1
export SCOREP_TOTAL_MEMORY=3500m
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=1000
export SCOREP_FILTERING_FILE=events.filter

python -m scorep runBenchmark.py iks
python -m scorep runBenchmark.py studd
