python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov5_rollout128_search --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 Search-Oracle --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov7_rollout128_search --filter-alg-apertures PPO_128:7 PPO-RTU_128:7 Search-Oracle --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov9_rollout128_search --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 Search-Oracle --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov15_rollout128_search --filter-alg-apertures PPO_128:15 PPO-RTU_128:15 Search-Oracle --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_search_regret --filter-alg-apertures Search-5 Search-7 Search-9 Search-Oracle --metric rolling_biome_regret_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_fovs --filter-alg-apertures PPO_128:5 PPO_128:7 PPO_128:9 PPO_128:15 --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_rtu_fovs --filter-alg-apertures PPO-RTU_128:5 PPO-RTU_128:7 PPO-RTU_128:9 PPO-RTU_128:15 --metric rolling_reward_1000000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_rank --filter-alg-apertures PPO_128:5 PPO_128:7 PPO_128:9 PPO_128:15 Search-Oracle Search-5 --metric rolling_biome_rank_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_rtu_rank --filter-alg-apertures PPO-RTU_128:5 PPO-RTU_128:7 PPO-RTU_128:9 PPO-RTU_128:15 Search-Oracle Search-5 --metric rolling_biome_rank_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_regret --filter-alg-apertures PPO_128:5 PPO_128:7 PPO_128:9 PPO_128:15 Search-Oracle Search-5 --metric rolling_biome_regret_100000 --ylim 0 0.8
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_rtu_regret --filter-alg-apertures PPO-RTU_128:5 PPO-RTU_128:7 PPO-RTU_128:9 PPO-RTU_128:15 Search-Oracle Search-5 --metric rolling_biome_regret_100000 --ylim 0 0.8

# PPO vs RTU Regret comparisons
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_regret_fov5 --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 Search-Oracle Search-5 --metric rolling_biome_regret_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_regret_fov7 --filter-alg-apertures PPO_128:7 PPO-RTU_128:7 Search-Oracle Search-5 --metric rolling_biome_regret_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_regret_fov9 --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 Search-Oracle Search-5 --metric rolling_biome_regret_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_regret_fov15 --filter-alg-apertures PPO_128:15 PPO-RTU_128:15 Search-Oracle Search-5 --metric rolling_biome_regret_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_regret_extremes --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 PPO_128:15 PPO-RTU_128:15 Search-Oracle Search-5 --metric rolling_biome_regret_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_regret_all --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 PPO_128:7 PPO-RTU_128:7 PPO_128:9 PPO-RTU_128:9 PPO_128:15 PPO-RTU_128:15 Search-Oracle Search-5 --metric rolling_biome_regret_100000

# PPO vs RTU Rank comparisons
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_rank_fov5 --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 Search-Oracle Search-5 --metric rolling_biome_rank_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_rank_fov7 --filter-alg-apertures PPO_128:7 PPO-RTU_128:7 Search-Oracle Search-5 --metric rolling_biome_rank_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_rank_fov9 --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 Search-Oracle Search-5 --metric rolling_biome_rank_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_rank_fov15 --filter-alg-apertures PPO_128:15 PPO-RTU_128:15 Search-Oracle Search-5 --metric rolling_biome_rank_100000
python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_ppo_vs_rtu_rank_extremes --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 PPO_128:15 PPO-RTU_128:15 Search-Oracle Search-5 --metric rolling_biome_rank_100000

for seed in {0..2}; do
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov5_ppo_rollout128_frozen_$seed --filter-alg-apertures PPO_128:5 PPO_128_frozen_5M:5 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov5_rtu_rollout128_frozen_$seed --filter-alg-apertures PPO-RTU_128:5 PPO-RTU_128_frozen_5M:5 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov9_ppo_rollout128_frozen_$seed --filter-alg-apertures PPO_128:9 PPO_128_frozen_5M:9 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov9_rtu_rollout128_frozen_$seed --filter-alg-apertures PPO-RTU_128:9 PPO-RTU_128_frozen_5M:9 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov25_ppo_rollout128_frozen_$seed --filter-alg-apertures PPO_128:15 PPO_128_frozen_5M:15 --filter-seeds $seed --metric ewm_reward_5
    python src/learning_curve.py experiments/E136-big/foragax/ForagaxBig-v2 --plot-name fig_big_fov25_rtu_rollout128_frozen_$seed --filter-alg-apertures PPO-RTU_128:15 PPO-RTU_128_frozen_5M:15 --filter-seeds $seed --metric ewm_reward_5
done
