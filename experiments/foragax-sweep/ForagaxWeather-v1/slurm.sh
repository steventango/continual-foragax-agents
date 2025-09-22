python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-15m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-3/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-5/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-7/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-45m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-11/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-45m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-13/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-15/DQN.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-15m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-3/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-5/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-7/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-45m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-11/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-45m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-13/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-15/DQN_L2_Init.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-15m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-3/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-5/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-7/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-30m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-45m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-11/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-45m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-13/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-1h.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-15/DQN_LN.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-15/Random.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-15/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 5 --entry src/continuing_main.py --force -e experiments/foragax-sweep/ForagaxWeather-v1/ForagaxWeather-v1-15/Search-Oracle.json
