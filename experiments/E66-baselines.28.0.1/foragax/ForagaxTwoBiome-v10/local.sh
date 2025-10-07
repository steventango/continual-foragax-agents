python scripts/local.py --runs 30 --entry src/continuing_main.py -e experiments/E66-baselines.28.0.1/foragax/ForagaxTwoBiome-v10/Baselines/*.json
python scripts/local.py --runs 1 --entry src/continuing_main.py -e experiments/E66-baselines.28.0.1/foragax/ForagaxTwoBiome-v10/9/DQN.json
python src/process_data.py experiments/E66-baselines.28.0.1/foragax/ForagaxTwoBiome-v10
python experiments/E66-baselines.28.0.1/foragax/ForagaxTwoBiome-v10/learning_curve.py
