python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E84-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v15/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E84-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v15/15/DQN_world.json
