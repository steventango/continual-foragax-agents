python scripts/local.py --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-15/Random.json --entry src/continuing_main.py --gpu
python scripts/local.py --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-15/Search-Nearest.json --entry src/continuing_main.py --gpu
python scripts/local.py --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-15/Search-Oracle.json --entry src/continuing_main.py --gpu
python experiments/foragax-sweep/hypers.py
python experiments/foragax-sweep/learning_curve.py
