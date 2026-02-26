for fov in 5 9 15; do
    for metric in reward biome_regret; do
        # python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_drqn/fig_big_drqn_configs_fov${fov}_${metric} --filter-alg-apertures DRQN_LN_1_1:${fov} DRQN_LN_0_2:${fov} Search-5 --metric rolling_${metric}_10000
        python src/learning_bar.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_drqn/fig_big_drqn_configs_fov${fov}_${metric}_bar --filter-alg-apertures DRQN_LN_1_1:${fov} DRQN_LN_0_2:${fov} --metric mean_${metric}
    done
done
