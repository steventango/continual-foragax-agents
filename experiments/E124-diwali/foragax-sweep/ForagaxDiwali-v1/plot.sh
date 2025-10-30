python src/learning_curve.py experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1 --plot-name fig1a_diwali_sweep --filter-alg-apertures DQN:5 PPO:5 --ylim "-0.05" 0.25
python src/learning_curve.py experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1 --plot-name fig1a_diwali_sweep_seeds --filter-alg-apertures DQN:5 PPO:5 --ylim "-0.1" 0.25 --subplot-by-seed
