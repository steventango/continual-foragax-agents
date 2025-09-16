python scripts/local.py --runs 30 -e experiments/foragax-old-exploration/ForagaxTwoBiomeSmall-15/Random.json --entry src/continuing_main.py
python scripts/local.py --runs 30 -e experiments/foragax-old-exploration/ForagaxTwoBiomeSmall-15/Search-Nearest.json --entry src/continuing_main.py
python scripts/local.py --runs 30 -e experiments/foragax-old-exploration/ForagaxTwoBiomeSmall-15/Search-Oracle.json --entry src/continuing_main.py
python experiments/foragax-old-exploration/learning_curve.py
python experiments/foragax-old-exploration/auc_fov.py
