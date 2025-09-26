python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-1d.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN_world.json
