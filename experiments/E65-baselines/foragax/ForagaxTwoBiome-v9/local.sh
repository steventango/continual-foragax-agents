python scripts/local.py --runs 30 --entry src/continuing_main.py -e experiments/E65-baselines/foragax/ForagaxTwoBiome-v9/Baselines/*.json
python src/process_data.py experiments/E65-baselines/foragax/ForagaxTwoBiome-v9
python experiments/E65-baselines/foragax/ForagaxTwoBiome-v9/learning_curve.py
