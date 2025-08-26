python scripts/local.py --runs 1 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-3/DQN.json --entry src/continuing_main.py
python scripts/local.py --runs 1 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-3/EQRC.json --entry src/continuing_main.py
python scripts/local.py --runs 1 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-5/DQN.json --entry src/continuing_main.py
python scripts/local.py --runs 1 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-5/EQRC.json --entry src/continuing_main.py
python experiments/foragax-sweep/hypers.py
python experiments/foragax-sweep/learning_curve.py
