python scripts/slurm.py --cluster clusters/fir-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E82-search-limited-fov/foragax/ForagaxTwoBiome-v15/9/DQN.json
python scripts/slurm.py --cluster clusters/fir-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E82-search-limited-fov/foragax/ForagaxTwoBiome-v15/9/DQN_frozen.json

python scripts/slurm.py --cluster clusters/fir-cpu-45m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E82-search-limited-fov/foragax/ForagaxTwoBiome-v15/Baselines/Search-Brown-Avoid-Green.json
python scripts/slurm.py --cluster clusters/fir-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E82-search-limited-fov/foragax/ForagaxTwoBiome-v15/Baselines/Search-Oracle.json
