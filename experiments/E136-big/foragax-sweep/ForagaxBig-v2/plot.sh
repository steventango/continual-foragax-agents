python src/learning_curve.py experiments/E136-big/foragax-sweep/ForagaxBig-v2 --plot-name fig1a_big_sweep_mean --filter-alg-apertures Search-5 Search-7 Search-9 Search-15 Search-Oracle --metric mean_reward 
python src/learning_curve.py experiments/E136-big/foragax-sweep/ForagaxBig-v2 --plot-name fig1a_big_sweep_cum_mean --filter-alg-apertures Search-5 Search-7 Search-9 Search-15 Search-Oracle --metric cum_mean_reward
python src/learning_curve.py experiments/E136-big/foragax-sweep/ForagaxBig-v2 --plot-name fig1a_big_sweep_rolling --filter-alg-apertures Search-5 Search-7 Search-9 Search-15 Search-Oracle --metric rolling_reward_100000 
for i in {1..9}; do
python src/learning_curve.py experiments/E136-big/foragax-sweep/ForagaxBig-v2 --plot-name fig1a_big_sweep_$i --filter-alg-apertures Search-5 Search-7 Search-9 Search-15 Search-Oracle --metric ewm_reward_$i 
done