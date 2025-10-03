python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E74-mitigations/foragax-sweep/ForagaxTwoBiome-v10/9/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E74-mitigations/foragax-sweep/ForagaxTwoBiome-v10/9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E74-mitigations/foragax-sweep/ForagaxTwoBiome-v10/9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E74-mitigations/foragax-sweep/ForagaxTwoBiome-v10/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E74-mitigations/foragax-sweep/ForagaxTwoBiome-v10/9/DQN_Shrink_and_Perturb.json
