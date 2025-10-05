python scripts/slurm.py --cluster clusters/vulcan-cpu-2h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13/9/DQN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13/9/DQN_frozen_5M.json
