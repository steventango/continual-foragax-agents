python scripts/local.py --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall/DQN.json --entry src/continuing_main.py
python scripts/local.py --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall/EQRC.json --entry src/continuing_main.py
python experiments/foragax-sweep/learning_curve.py
python scripts/local.py --runs 30 -e experiments/foragax/ForagaxTwoBiomeSmall/DQN.json --entry src/continuing_main.py
python scripts/local.py --runs 30 -e experiments/foragax/ForagaxTwoBiomeSmall/EQRC.json --entry src/continuing_main.py
python experiments/foragax/learning_curve.py
