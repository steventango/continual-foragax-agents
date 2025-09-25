python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E26-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v2/9/DQN_B1000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E26-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v2/9/DQN_B10000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E26-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v2/9/DQN_B100000.json
