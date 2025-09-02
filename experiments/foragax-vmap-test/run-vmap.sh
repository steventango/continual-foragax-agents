mkdir -p results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs\

start=0
end=199
indices=($(seq $start $end))

{ time python src/continuing_main.py --gpu -e experiments/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN.json -i ${indices[@]}; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/vmap.log
