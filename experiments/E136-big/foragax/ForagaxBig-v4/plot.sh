for fov in 5 9 15; do
    # python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov${fov}_reward --filter-alg-apertures PPO_LN_128:${fov} DQN_LN:${fov} PPO-RTU_LN_128_512:${fov} Search-5 --metric rolling_reward_10000
    # python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov${fov}_reward_seeds --filter-alg-apertures PPO_LN_128:${fov} DQN_LN:${fov} PPO-RTU_LN_128_512:${fov} Search-5 --plot-all-seeds --metric rolling_reward_10000
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name dqn_fov${fov}_reward_seeds --filter-alg-apertures DQN_LN:${fov} Search-5 --plot-all-seeds --metric rolling_reward_10000 --legend
    # python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov${fov}_regret --filter-alg-apertures PPO_LN_128:${fov} DQN_LN:${fov} PPO-RTU_LN_128_512:${fov} Search-5 --metric rolling_biome_regret_10000
done


