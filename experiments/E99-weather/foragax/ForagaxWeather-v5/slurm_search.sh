python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/Baselines/Search-Oracle.json
