#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E123-two-biome_foragax-sweep_ForagaxSineTwoBiome-v1_process_data
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --output=../slurm-%j.out
#SBATCH --time=00:15:00

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

$SLURM_TMPDIR/.venv/bin/python src/process_data.py experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1
