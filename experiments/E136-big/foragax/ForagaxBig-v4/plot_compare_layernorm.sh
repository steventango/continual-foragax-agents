for fov in 5 9 15; do
    for metric in reward biome_regret; do
        python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_layernorm/fig_big_dqn_fov${fov}_${metric} --filter-alg-apertures DQN:${fov} DQN_LN:${fov} Search-5 --metric rolling_${metric}_10000
        python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_layernorm/fig_big_ppo_fov${fov}_${metric} --filter-alg-apertures PPO_128:${fov} PPO_LN_128:${fov} Search-5 --metric rolling_${metric}_10000
        python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_layernorm/fig_big_rtu_fov${fov}_${metric} --filter-alg-apertures PPO-RTU_128:${fov} PPO-RTU_LN_128:${fov} Search-5 --metric rolling_${metric}_10000
    done
done