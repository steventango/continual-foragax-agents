python src/learning_curve.py experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v2 --plot-name fig1a_diwali_sweep --filter-alg-apertures DQN:5 PPO:5 PPO-RTU:5 DQN:9 PPO:9 PPO-RTU:9
python src/learning_curve.py experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v2 --plot-name fig1a_diwali_sweep_seeds --filter-alg-apertures DQN:5 PPO:5 PPO-RTU:5 DQN:9 PPO:9 PPO-RTU:9 --ylim "-0.1" 0.25 --subplot-by-seed
python src/learning_curve.py experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v2 --plot-name fig1a_diwali_5_sweep --filter-alg-apertures PPO:5 PPO-RTU:5 PPO_L2:5 PPO-RTU_L2:5
python src/learning_curve.py experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v2 --plot-name fig1a_diwali_9_sweep --filter-alg-apertures PPO:9 PPO-RTU:9 PPO_L2:9 PPO-RTU_L2:9
