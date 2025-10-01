python scripts/local.py --runs 30 --entry src/continuing_main.py -e experiments/E68-baselines/foragax/ForagaxTwoBiome-v12/Baselines/*.json
python src/process_data.py experiments/E68-baselines/foragax/ForagaxTwoBiome-v12
python experiments/E68-baselines/foragax/ForagaxTwoBiome-v12/learning_curve.py
