for fov in 5 9 15; do
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_state_construction/ppo_fov${fov}_reward --filter-alg-aperture PPO_LN_128:${fov} PPO_LN_RT_128:${fov} PPO-RTU_LN_128_512:${fov} Search-5 --metric rolling_reward_10000
    # python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_state_construction/ppo_fov${fov}_regret --filter-alg-aperture PPO_LN_128:${fov} PPO_LN_RT_128:${fov} PPO-RTU_LN_128_512:${fov} Search-5 --metric rolling_biome_regret_10000
    # python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_state_construction/dqn_fov${fov}_reward --filter-alg-aperture DQN_LN:${fov} DQN_LN_RT:${fov} DRQN_LN_1_1:${fov} Search-5 --metric rolling_reward_10000
    # python src/learning_bar.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_state_construction/dqn_fov${fov}_reward_bar --filter-alg-apertures DQN_LN:${fov} DQN_LN_RT:${fov} DRQN_LN_1_1:${fov} Search-5 --metric mean_reward
    # python src/learning_bar.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_state_construction/dqn_fov${fov}_regret_bar --filter-alg-apertures DQN_LN:${fov} DQN_LN_RT:${fov} DRQN_LN_1_1:${fov} Search-5 --metric biome_regret
done
