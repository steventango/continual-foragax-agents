python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/5/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/7/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/15/DQN.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/5/DQN_L2.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/7/DQN_L2.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/9/DQN_L2.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/5/DQN_SWR.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/7/DQN_SWR.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E123-two-biome/foragax-sweep/ForagaxSineTwoBiome-v1/9/DQN_SWR.json
