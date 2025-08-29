export XLA_PYTHON_CLIENT_MEM_FRACTION=0.16
mkdir -p results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs
{ time python src/continuing_main.py --gpu -e experiments/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN.json -i 0; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/0-multiprocess.log &
{ time python src/continuing_main.py --gpu -e experiments/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN.json -i 1; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/1-multiprocess.log &
{ time python src/continuing_main.py --gpu -e experiments/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN.json -i 2; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/2-multiprocess.log &
{ time python src/continuing_main.py --gpu -e experiments/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN.json -i 3; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/3-multiprocess.log &
{ time python src/continuing_main.py --gpu -e experiments/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN.json -i 4; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/4-multiprocess.log &
wait
