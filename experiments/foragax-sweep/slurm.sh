python scripts/slurm.py --cluster clusters/fir-t1-c8-m2-s32.json --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-3/*.json --entry src/continuing_main.py
python scripts/slurm.py --cluster clusters/fir-t1-c8-m2-s32.json --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-5/DQN.json --entry src/continuing_main.py
python scripts/slurm.py --cluster clusters/fir-t1-c8-m2-s4.json --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-7/DQN.json experiments/foragax-sweep/ForagaxTwoBiomeSmall-7/EQRC.json --entry src/continuing_main.py
python scripts/slurm.py --cluster clusters/fir-t1-c8-m2-s4.json --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-9/DQN.json experiments/foragax-sweep/ForagaxTwoBiomeSmall-9/EQRC.json --entry src/continuing_main.py
python scripts/slurm.py --cluster clusters/fir-t1-c8-m2-s4.json --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-11/*.json --entry src/continuing_main.py
python scripts/slurm.py --cluster clusters/fir-t1-c8-m2-s4.json --runs 5 -e experiments/foragax-sweep/ForagaxTwoBiomeSmall-13/*.json experiments/foragax-sweep/ForagaxTwoBiomeSmall-15/*.json --entry src/continuing_main.py
