python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-3/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-5/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-7/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-5h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-11/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-13/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-7h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/DQN.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Random.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Search-Brown-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Search-Brown.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Search-Morel-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Search-Morel.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-45m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Search-Oracle.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxTwoBiome-v1/ForagaxTwoBiome-v1-15/Search-Oyster.json
