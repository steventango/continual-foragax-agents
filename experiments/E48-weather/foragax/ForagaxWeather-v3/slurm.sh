python scripts/slurm.py --cluster clusters/vulcan-cpu-8h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/9/DQN_frozen.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E48-weather/foragax/ForagaxWeather-v3/Baselines/Search-Oracle.json
