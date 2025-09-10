python scripts/local.py --runs 30 -e experiments/foragax/ForagaxTwoBiomeSmall-15/Random.json --entry src/continuing_main.py --gpu
python scripts/local.py --runs 30 -e experiments/foragax/ForagaxTwoBiomeSmall-15/Search-Nearest.json --entry src/continuing_main.py --gpu
python scripts/local.py --runs 30 -e experiments/foragax/ForagaxTwoBiomeSmall-15/Search-Oracle.json --entry src/continuing_main.py
python experiments/foragax/learning_curve.py
python experiments/foragax/auc_fov.py
