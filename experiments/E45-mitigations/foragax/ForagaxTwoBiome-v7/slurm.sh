python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E45-mitigations/foragax/ForagaxTwoBiome-v7/9/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-9h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E45-mitigations/foragax/ForagaxTwoBiome-v7/9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-9h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E45-mitigations/foragax/ForagaxTwoBiome-v7/9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E45-mitigations/foragax/ForagaxTwoBiome-v7/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E45-mitigations/foragax/ForagaxTwoBiome-v7/9/DQN_Shrink_and_Perturb.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E45-mitigations/foragax/ForagaxTwoBiome-v7/9/DQN.json
