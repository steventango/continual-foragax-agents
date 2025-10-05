python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10/9/DQN_frozen_5M.json
