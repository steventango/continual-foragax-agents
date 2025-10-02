#!/bin/bash

#SBATCH --time=00:55:00
#SBATCH --account=rrg-whitem

module load python/3.11 arrow/19 gcc opencv rust swig

cp $path/pyproject.toml $SLURM_TMPDIR/
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate

pip install -e .

cp .venv $path/

pip freeze
