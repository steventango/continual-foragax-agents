for fov in 5 9 15; do
    for metric in reward biome_regret; do
        for alg in dqn ppo rtu drqn_1_1 drqn_0_2; do
            case $alg in
                dqn) base="DQN"; ln="DQN_LN" ;;
                ppo) base="PPO_128"; ln="PPO_LN_128" ;;
                rtu) base="PPO-RTU_128"; ln="PPO-RTU_LN_128" ;;
                drqn_1_1) base="DRQN_1_1"; ln="DRQN_LN_1_1" ;;
                drqn_0_2) base="DRQN_0_2"; ln="DRQN_LN_0_2" ;;
            esac
            python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_layernorm/fig_big_${alg}_fov${fov}_${metric} --filter-alg-apertures ${base}:${fov} ${ln}:${fov} Search-5 --metric rolling_${metric}_10000
        done
    done
done