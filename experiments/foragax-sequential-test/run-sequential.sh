mkdir -p results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs

# for loop from 0 to 99
for i in {0..9}
do
    { time python src/continuing_main_old.py --gpu -e experiments/foragax-sequential-test/ForagaxTwoBiomeSmall-15/DQN.json -i $i; } &> results/foragax-vmap-test/ForagaxTwoBiomeSmall-15/DQN/logs/$i-sequential.log
done
