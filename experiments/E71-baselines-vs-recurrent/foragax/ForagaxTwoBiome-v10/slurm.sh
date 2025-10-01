python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E71-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v10/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E71-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v10/15/DQN_world.json
