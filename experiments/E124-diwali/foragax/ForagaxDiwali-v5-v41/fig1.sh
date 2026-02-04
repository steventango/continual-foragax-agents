

python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v5-v41 --plot-name fig_diwali_fov5_ppo_rollout128_frozen --filter-alg-apertures PPO_128:5 PPO_128_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v5-v41 --plot-name fig_diwali_fov5_rollout128_search --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 Search-Oracle Search-Nearest --metric ewm_reward_5

python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v5-v41 --plot-name fig_diwali_fov5_rtu_rollout128_frozen --filter-alg-apertures PPO-RTU_128:5 PPO-RTU_128_frozen_5M:5 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v5-v41 --plot-name fig_diwali_fov9_ppo_rollout128_frozen --filter-alg-apertures PPO_128:9 PPO_128_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5

python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v5-v41 --plot-name fig_diwali_fov9_rollout128_search --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 Search-Oracle Search-Nearest --metric ewm_reward_5

python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v5-v41 --plot-name fig_diwali_fov9_rtu_rollout128_frozen --filter-alg-apertures PPO-RTU_128:9 PPO-RTU_128_frozen_5M:9 --filter-seeds 0 --metric ewm_reward_5
