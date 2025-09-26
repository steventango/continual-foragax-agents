python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E54-weather/foragax/ForagaxWeather-v3/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E54-weather/foragax/ForagaxWeather-v3/Baselines/Search-Oracle.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E54-weather/foragax/ForagaxWeather-v3/Baselines/Search-Nearest_bug.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/E54-weather/foragax/ForagaxWeather-v3/Baselines/Search-Oracle_bug.json
