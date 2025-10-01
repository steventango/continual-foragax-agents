python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E69-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E69-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-45m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E69-search-limited-fov/foragax/ForagaxTwoBiome-v10/Baselines/Search-Brown-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E69-search-limited-fov/foragax/ForagaxTwoBiome-v10/Baselines/Search-Oracle.json
