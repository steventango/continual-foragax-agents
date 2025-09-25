python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v7/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v7/15/DQN_world.json
