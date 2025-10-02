python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E76-search-limited-fov/foragax/ForagaxTwoBiome-v13/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E76-search-limited-fov/foragax/ForagaxTwoBiome-v13/9/DQN_frozen.json
