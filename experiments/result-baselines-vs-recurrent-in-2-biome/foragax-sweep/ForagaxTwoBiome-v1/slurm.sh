python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-6h.json --runs 5 --entry src/continuing_main.py --force -e experiments/result-baselines-vs-recurrent-in-2-biome/foragax-sweep/ForagaxTwoBiome-v1/15/DQN_B1000.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-6h.json --runs 5 --entry src/continuing_main.py --force -e experiments/result-baselines-vs-recurrent-in-2-biome/foragax-sweep/ForagaxTwoBiome-v1/15/DQN_B10000.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-6h.json --runs 5 --entry src/continuing_main.py --force -e experiments/result-baselines-vs-recurrent-in-2-biome/foragax-sweep/ForagaxTwoBiome-v1/15/DQN_B100000.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 5 --entry src/continuing_main.py --force -e experiments/result-baselines-vs-recurrent-in-2-biome/foragax-sweep/ForagaxTwoBiome-v1/15/DQN_world_B1000.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 5 --entry src/continuing_main.py --force -e experiments/result-baselines-vs-recurrent-in-2-biome/foragax-sweep/ForagaxTwoBiome-v1/15/DQN_world_B10000.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 5 --entry src/continuing_main.py --force -e experiments/result-baselines-vs-recurrent-in-2-biome/foragax-sweep/ForagaxTwoBiome-v1/15/DQN_world_B100000.json
