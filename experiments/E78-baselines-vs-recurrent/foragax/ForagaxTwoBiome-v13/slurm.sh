python scripts/slurm.py --cluster clusters/fir-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E78-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v13/15/DQN.json
python scripts/slurm.py --cluster clusters/fir-gpu-mps-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E78-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v13/15/DQN_world.json
