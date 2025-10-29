python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v1/Baselines/Search-Brown-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v1/Baselines/Search-Oracle.json
