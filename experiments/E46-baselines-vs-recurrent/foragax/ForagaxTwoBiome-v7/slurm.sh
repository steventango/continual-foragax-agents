python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN0.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN00.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN_world.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN_world0.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E46-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v7/15/DQN_world00.json
