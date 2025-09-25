python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E43-baselines/foragax/ForagaxTwoBiome-v7/Baselines/Random.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E43-baselines/foragax/ForagaxTwoBiome-v7/Baselines/Search-Brown-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E43-baselines/foragax/ForagaxTwoBiome-v7/Baselines/Search-Brown.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E43-baselines/foragax/ForagaxTwoBiome-v7/Baselines/Search-Morel-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E43-baselines/foragax/ForagaxTwoBiome-v7/Baselines/Search-Morel.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E43-baselines/foragax/ForagaxTwoBiome-v7/Baselines/Search-Oracle.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E43-baselines/foragax/ForagaxTwoBiome-v7/Baselines/Search-Oyster.json
