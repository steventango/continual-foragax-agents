#!/bin/bash

#SBATCH --account=rrg-whitem
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=8
#SBATCH --time=01:00:00

module load python/3.11 arrow/19 gcc opencv rust swig

cp $path/pyproject.toml $SLURM_TMPDIR/
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate

pip install -e .

cp .venv $path/

pip freeze
