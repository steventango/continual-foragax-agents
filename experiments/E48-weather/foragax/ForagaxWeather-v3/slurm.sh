python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_Hare_and_Tortoise_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_L2_Init_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_LN_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_Reset_Head_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_Shrink_and_Perturb.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_Shrink_and_Perturb_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/15/DQN_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/15/DQN_privileged.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/15/DQN_privileged_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-12h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/15/DQN_world.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/15/DQN_world_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/Baselines/Search-Oracle.json
