python scripts/local.py --runs 30 -e experiments/foragax-new-exploration/ForagaxTwoBiomeSmall-15/Random.json --entry src/continuing_main.py
python scripts/local.py --runs 30 -e experiments/foragax-new-exploration/ForagaxTwoBiomeSmall-15/Search-Nearest.json --entry src/continuing_main.py
python scripts/local.py --runs 30 -e experiments/foragax-new-exploration/ForagaxTwoBiomeSmall-15/Search-Oracle.json --entry src/continuing_main.py
python experiments/foragax-new-exploration/learning_curve.py
python experiments/foragax-new-exploration/auc_fov.py
