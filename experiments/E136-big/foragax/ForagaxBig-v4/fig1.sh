python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_ppo_rollout --filter-alg-apertures PPO_32:5 PPO_64:5 PPO_128:5 PPO_256:5 PPO_512:5 --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rtu_rollout --filter-alg-apertures PPO-RTU_32:5 PPO-RTU_64:5 PPO-RTU_128:5 PPO-RTU_256:5 PPO-RTU_512:5 --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_ppo_rollout --filter-alg-apertures PPO_32:9 PPO_64:9 PPO_128:9 PPO_256:9 PPO_512:9 --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rtu_rollout --filter-alg-apertures PPO-RTU_32:9 PPO-RTU_64:9 PPO-RTU_128:9 PPO-RTU_256:9 PPO-RTU_512:9 --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_ppo_rollout --filter-alg-apertures PPO_32:15 PPO_64:15 PPO_128:15 PPO_256:15 PPO_512:15 --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rtu_rollout --filter-alg-apertures PPO-RTU_32:15 PPO-RTU_64:15 PPO-RTU_128:15 PPO-RTU_256:15 PPO-RTU_512:15 --metric rolling_reward_1000000

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rollout32_search --filter-alg-apertures PPO_32:5 PPO-RTU_32:5 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rollout64_search --filter-alg-apertures PPO_64:5 PPO-RTU_64:5 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rollout128_search --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rollout256_search --filter-alg-apertures PPO_256:5 PPO-RTU_256:5 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rollout512_search --filter-alg-apertures PPO_512:5 PPO-RTU_512:5 Search-Oracle Search-Nearest --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rollout32_search --filter-alg-apertures PPO_32:9 PPO-RTU_32:9 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rollout64_search --filter-alg-apertures PPO_64:9 PPO-RTU_64:9 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rollout128_search --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rollout256_search --filter-alg-apertures PPO_256:9 PPO-RTU_256:9 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rollout512_search --filter-alg-apertures PPO_512:9 PPO-RTU_512:9 Search-Oracle Search-Nearest --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rollout32_search --filter-alg-apertures PPO_32:15 PPO-RTU_32:15 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rollout64_search --filter-alg-apertures PPO_64:15 PPO-RTU_64:15 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rollout128_search --filter-alg-apertures PPO_128:15 PPO-RTU_128:15 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rollout256_search --filter-alg-apertures PPO_256:15 PPO-RTU_256:15 Search-Oracle Search-Nearest --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rollout512_search --filter-alg-apertures PPO_512:15 PPO-RTU_512:15 Search-Oracle Search-Nearest --metric ewm_reward_5



python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_search --filter-alg-apertures Search-Oracle Search-Nearest --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_ppo_rollout32_frozen --filter-alg-apertures PPO_32:5 PPO_32_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_ppo_rollout64_frozen --filter-alg-apertures PPO_64:5 PPO_64_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_ppo_rollout128_frozen --filter-alg-apertures PPO_128:5 PPO_128_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_ppo_rollout256_frozen --filter-alg-apertures PPO_256:5 PPO_256_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_ppo_rollout512_frozen --filter-alg-apertures PPO_512:5 PPO_512_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rtu_rollout32_frozen --filter-alg-apertures PPO-RTU_32:5 PPO-RTU_32_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rtu_rollout64_frozen --filter-alg-apertures PPO-RTU_64:5 PPO-RTU_64_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rtu_rollout128_frozen --filter-alg-apertures PPO-RTU_128:5 PPO-RTU_128_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rtu_rollout256_frozen --filter-alg-apertures PPO-RTU_256:5 PPO-RTU_256_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov5_rtu_rollout512_frozen --filter-alg-apertures PPO-RTU_512:5 PPO-RTU_512_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_ppo_rollout32_frozen --filter-alg-apertures PPO_32:9 PPO_32_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_ppo_rollout64_frozen --filter-alg-apertures PPO_64:9 PPO_64_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_ppo_rollout128_frozen --filter-alg-apertures PPO_128:9 PPO_128_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_ppo_rollout256_frozen --filter-alg-apertures PPO_256:9 PPO_256_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_ppo_rollout512_frozen --filter-alg-apertures PPO_512:9 PPO_512_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rtu_rollout32_frozen --filter-alg-apertures PPO-RTU_32:9 PPO-RTU_32_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rtu_rollout64_frozen --filter-alg-apertures PPO-RTU_64:9 PPO-RTU_64_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rtu_rollout128_frozen --filter-alg-apertures PPO-RTU_128:9 PPO-RTU_128_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rtu_rollout256_frozen --filter-alg-apertures PPO-RTU_256:9 PPO-RTU_256_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov9_rtu_rollout512_frozen --filter-alg-apertures PPO-RTU_512:9 PPO-RTU_512_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_ppo_rollout32_frozen --filter-alg-apertures PPO_32:15 PPO_32_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_ppo_rollout64_frozen --filter-alg-apertures PPO_64:15 PPO_64_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_ppo_rollout128_frozen --filter-alg-apertures PPO_128:15 PPO_128_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_ppo_rollout256_frozen --filter-alg-apertures PPO_256:15 PPO_256_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_ppo_rollout512_frozen --filter-alg-apertures PPO_512:15 PPO_512_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rtu_rollout32_frozen --filter-alg-apertures PPO-RTU_32:15 PPO-RTU_32_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rtu_rollout64_frozen --filter-alg-apertures PPO-RTU_64:15 PPO-RTU_64_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rtu_rollout128_frozen --filter-alg-apertures PPO-RTU_128:15 PPO-RTU_128_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rtu_rollout256_frozen --filter-alg-apertures PPO-RTU_256:15 PPO-RTU_256_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v4 --plot-name fig_big_fov35_rtu_rollout512_frozen --filter-alg-apertures PPO-RTU_512:15 PPO-RTU_512_frozen_5M:15 --filter-seeds 0 --metric ewm_reward_5
