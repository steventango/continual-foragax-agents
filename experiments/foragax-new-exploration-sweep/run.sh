python scripts/local.py --runs 5 -e experiments/foragax-old-exploration-sweep/ForagaxTwoBiomeSmall-15/Random.json --entry src/continuing_main.py &
python scripts/local.py --runs 5 -e experiments/foragax-old-exploration-sweep/ForagaxTwoBiomeSmall-15/Search-Nearest.json --entry src/continuing_main.py &
python scripts/local.py --runs 5 -e experiments/foragax-old-exploration-sweep/ForagaxTwoBiomeSmall-15/Search-Oracle.json --entry src/continuing_main.py &
wait
python experiments/foragax-old-exploration-sweep/hypers.py
python experiments/foragax-old-exploration-sweep/learning_curve.py
