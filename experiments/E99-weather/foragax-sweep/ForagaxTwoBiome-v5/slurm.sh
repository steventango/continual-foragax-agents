python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/9/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/9/DQN_Shrink_and_Perturb.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-2h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-5h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/15/DQN_privileged.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-5h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5/15/DQN_world.json
