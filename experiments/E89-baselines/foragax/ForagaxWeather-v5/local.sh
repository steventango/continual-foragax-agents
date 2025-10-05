python scripts/local.py --runs 1 --entry src/continuing_main.py -e experiments/E89-baselines/foragax/ForagaxWeather-v5/Baselines/*.json
python src/process_data.py experiments/E89-baselines/foragax/ForagaxWeather-v5
python src/learning_curve.py experiments/E89-baselines/foragax/ForagaxWeather-v5
python src/mushroom_curve.py experiments/E89-baselines/foragax/ForagaxWeather-v5
python src/biome_curve.py experiments/E89-baselines/foragax/ForagaxWeather-v5
