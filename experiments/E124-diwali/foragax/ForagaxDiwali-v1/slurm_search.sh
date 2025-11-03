python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 00:30:00 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v1/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v1/Baselines/Search-Oracle.json
