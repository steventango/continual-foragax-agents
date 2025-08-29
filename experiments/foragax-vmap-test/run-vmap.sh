mkdir -p results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs

{ time python src/continuing_main.py --gpu -e experiments/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN.json -i 0 1 2 3 4; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/vmap.log
