#!/bin/bash

#SBATCH --time=00:55:00
#SBATCH --account=rrg-whitem

module load python/3.11 rust

cp $path/requirements.txt $SLURM_TMPDIR/
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate

# TODO: for some reason, pip cannot install any of the current wheels for this package.
# this is a pretty bad hack, but...
pip install --platform manylinux_2_28_x86_64 --no-deps --target .venv/lib/python3.11/site-packages connectorx

pip install .

tar -cavf venv.tar.xz .venv
cp venv.tar.xz $path/

pip freeze
