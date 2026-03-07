# foragax-sweep (5 runs)
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/9/DQN_Shrink_and_Perturb.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN.json


# foragax (5 runs)
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_Shrink_and_Perturb.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN.json
