python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E95-weather/foragax/ForagaxWeather-v4/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E95-weather/foragax/ForagaxWeather-v4/Baselines/Search-Oracle.json
