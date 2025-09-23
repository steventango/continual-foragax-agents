#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=hypers_result-mitigations-in-2-biome_foragax-sweep_ForagaxTwoBiome-v1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=../slurm-%j.out
#SBATCH --time=00:30:00

module load arrow/19

tar -xf venv.tar.xz -C $SLURM_TMPDIR

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORMS=cpu

$SLURM_TMPDIR/.venv/bin/python experiments/result-mitigations-in-2-biome/foragax-sweep/ForagaxTwoBiome-v1/hypers.py
