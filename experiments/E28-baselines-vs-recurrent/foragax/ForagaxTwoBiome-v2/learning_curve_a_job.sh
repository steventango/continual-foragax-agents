#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=learning_curve_a_E28-baselines-vs-recurrent_foragax_ForagaxTwoBiome-v2
#SBATCH --mem-per-cpu=128G
#SBATCH --exclude=rack08-11
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

$SLURM_TMPDIR/.venv/bin/python experiments/E28-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v2/learning_curve_a.py
