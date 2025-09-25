python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E28-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v2/15/DQN_B1000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E28-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v2/15/DQN_B10000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E28-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v2/15/DQN_B100000.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E28-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v2/15/DQN_world_B1000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E28-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v2/15/DQN_world_B10000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E28-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v2/15/DQN_world_B100000.json
