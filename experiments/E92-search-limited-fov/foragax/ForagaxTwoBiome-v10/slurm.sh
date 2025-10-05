python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-45m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-2h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN_frozen_5M.json
