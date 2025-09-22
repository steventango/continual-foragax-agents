python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-3/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-5/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-7/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-5h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-11/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-13/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-7h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-15/DQN.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-45m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-3/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-5/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-7/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-5h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-11/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-13/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-15/DQN_L2_Init.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-3/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-5/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-7/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-5h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-11/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-7h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-13/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-15/DQN_LN.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-15/Random.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-45m.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-15/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-5h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-15/Search-Oracle.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-2h.json --runs 30 --entry src/continuing_main.py --force -e experiments/foragax/ForagaxWeather-v1/ForagaxWeather-v1-15/Search-Oyster.json
