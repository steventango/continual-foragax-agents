python src/learning_curve.py experiments/E136-big/foragax-sweep/ForagaxBig-v1 --plot-name fig1a_big_sweep --filter-alg-apertures PPO:5 PPO-RTU:5 PPO:9 PPO-RTU:9
python src/learning_curve.py experiments/E136-big/foragax-sweep/ForagaxBig-v1 --plot-name fig1a_big_sweep_seeds --filter-alg-apertures PPO:5 PPO-RTU:5 PPO:9 PPO-RTU:9 --ylim "-0.1" 0.25 --subplot-by-seed
