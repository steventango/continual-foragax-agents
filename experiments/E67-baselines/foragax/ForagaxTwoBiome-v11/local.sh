python scripts/local.py --runs 30 --entry src/continuing_main.py -e experiments/E67-baselines/foragax/ForagaxTwoBiome-v11/Baselines/*.json
python src/process_data.py experiments/E67-baselines/foragax/ForagaxTwoBiome-v11
python experiments/E67-baselines/foragax/ForagaxTwoBiome-v11/learning_curve.py
