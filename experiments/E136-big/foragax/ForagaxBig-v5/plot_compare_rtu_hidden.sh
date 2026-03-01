for fov in 5 9 15; do
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v5 --plot-name compare_rtu_hidden/fig_big_fov${fov}_reward --filter-alg-aperture PPO-RTU_LN_128:${fov} PPO-RTU_LN_128_512:${fov} Search-5 --metric rolling_reward_10000
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v5 --plot-name compare_rtu_hidden/fig_big_fov${fov}_regret --filter-alg-aperture PPO-RTU_LN_128:${fov} PPO-RTU_LN_128_512:${fov} Search-5 --metric rolling_biome_regret_10000
done
