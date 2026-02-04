python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov5_rollout128_search --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 Search-Oracle Search-Nearest --metric rolling_reward_1000000
python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov9_rollout128_search --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 Search-Oracle Search-Nearest --metric rolling_reward_1000000
python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov15_rollout128_search --filter-alg-apertures PPO_128:15 PPO-RTU_128:15 Search-Oracle Search-Nearest --metric rolling_reward_1000000


for seed in {0..2}; do
    python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov5_ppo_rollout128_frozen_$seed --filter-alg-apertures PPO_128:5 PPO_128_frozen_5M:5 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov5_rtu_rollout128_frozen_$seed --filter-alg-apertures PPO-RTU_128:5 PPO-RTU_128_frozen_5M:5 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov9_ppo_rollout128_frozen_$seed --filter-alg-apertures PPO_128:9 PPO_128_frozen_5M:9 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov9_rtu_rollout128_frozen_$seed --filter-alg-apertures PPO-RTU_128:9 PPO-RTU_128_frozen_5M:9 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov15_ppo_rollout128_frozen_$seed --filter-alg-apertures PPO_128:15 PPO_128_frozen_5M:15 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v6 --plot-name fig_diwali_fov15_rtu_rollout128_frozen_$seed --filter-alg-apertures PPO-RTU_128:15 PPO-RTU_128_frozen_5M:15 --filter-seeds $seed --metric ewm_reward_5
done
