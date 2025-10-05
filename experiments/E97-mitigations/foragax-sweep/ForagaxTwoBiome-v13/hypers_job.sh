#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E97_hypers-mitigations_foragax-sweep_ForagaxTwoBiome-v13
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --output=../slurm-%j.out
#SBATCH --time=00:30:00

module load arrow/19

cp -R .venv $SLURM_TMPDIR

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORMS=cpu

$SLURM_TMPDIR/.venv/bin/python experiments/E97-mitigations/foragax-sweep/ForagaxTwoBiome-v13/hypers.py

$SLURM_TMPDIR/.venv/bin/python experiments/E97-mitigations/foragax/ForagaxTwoBiome-v13/generate_frozen_config.py
