python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E85-search-limited-fov/foragax/ForagaxTwoBiome-v16/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E85-search-limited-fov/foragax/ForagaxTwoBiome-v16/9/DQN_frozen.json
