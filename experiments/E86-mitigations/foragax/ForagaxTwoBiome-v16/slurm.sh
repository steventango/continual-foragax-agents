python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_Shrink_and_Perturb.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_Hare_and_Tortoise_frozen.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_L2_Init_frozen.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_LN_frozen.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_Reset_Head_frozen.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16/9/DQN_Shrink_and_Perturb_frozen.json
