mkdir -p results/logs/
for i in {0..0}
do
  # { time python src/continuing_main.py -e experiments/foragax-sweep/ForagaxTwoBiomeSmall/DQN.json -i $i &> results/logs/DQN_$i.log; } 2> results/logs/DQN_$i.time &
  { time python src/continuing_main.py -e experiments/foragax-sweep/ForagaxTwoBiomeSmall/EQRC.json -i $i &> results/logs/EQRC_$i.log; } 2> results/logs/EQRC_$i.time &
done
wait
python experiments/foragax-sweep/learning_curve.py
