for alg in DQN_LN PPO_LN_128 PPO-RTU_LN_128_512; do
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name compare_fov/${alg}_reward --filter-alg-aperture ${alg}:5 ${alg}:9 ${alg}:15 Search-5 --metric rolling_reward_10000
done
