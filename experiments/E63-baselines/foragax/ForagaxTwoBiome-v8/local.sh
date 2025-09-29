python scripts/local.py --runs 30 --entry src/continuing_main.py -e experiments/E63-baselines/foragax/ForagaxTwoBiome-v8/Baselines/*.json
python src/process_data.py experiments/E63-baselines/foragax/ForagaxTwoBiome-v8
python experiments/E63-baselines/foragax/ForagaxTwoBiome-v8/learning_curve.py
