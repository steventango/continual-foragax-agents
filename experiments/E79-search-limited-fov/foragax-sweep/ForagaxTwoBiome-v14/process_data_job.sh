#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E79-search-limited-fov_foragax-sweep_ForagaxTwoBiome-v14_process_data
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --output=../slurm-%j.out
#SBATCH --time=00:15:00

module load arrow/19

rsync -azP .venv/ $SLURM_TMPDIR/

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORMS=cpu

$SLURM_TMPDIR/.venv/bin/python src/process_data.py experiments/E79-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v14
