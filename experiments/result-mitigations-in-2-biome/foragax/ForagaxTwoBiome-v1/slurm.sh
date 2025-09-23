python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-mitigations-in-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-mitigations-in-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-mitigations-in-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-mitigations-in-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-mitigations-in-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_Shrink_and_Perturb.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-mitigations-in-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN.json

