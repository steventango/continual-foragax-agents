python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig1a_diwali_fov_5 --filter-alg-apertures Search-Oracle Search-Nearest DQN:5 PPO:5 PPO-RTU:5
python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig1a_diwali_fov_9 --filter-alg-apertures Search-Oracle Search-Nearest DQN:9 PPO:9 PPO-RTU:9
python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig1a_diwali_search_seeds --filter-alg-apertures Search-Oracle Search-Nearest --ylim "-0.1" 0.25 --subplot-by-seed

python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov5_rollout128_search --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 Search-Oracle Search-Nearest 
python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov9_rollout128_search --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 Search-Oracle Search-Nearest 
python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov5_rollout128_frozen --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 PPO_128_frozen_5M:5 PPO-RTU_128_frozen_5M:5 
python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov9_rollout128_frozen --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 PPO_128_frozen_5M:9 PPO-RTU_128_frozen_5M:9


# python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov5_rollout128 --filter-alg-apertures PPO_128:5 PPO-RTU_128:5 PPO_128_frozen_5M:5 PPO-RTU_128_frozen_5M:5 Search-Oracle Search-Nearest 
# python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov9_rollout128 --filter-alg-apertures PPO_128:9 PPO-RTU_128:9 PPO_128_frozen_5M:9 PPO-RTU_128_frozen_5M:9 Search-Oracle Search-Nearest 
# python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov5_rollout512 --filter-alg-apertures PPO_512:5 PPO-RTU_512:5 PPO_512_frozen_5M:5 PPO-RTU_512_frozen_5M:5 Search-Oracle Search-Nearest 
# python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov9_rollout512 --filter-alg-apertures PPO_512:9 PPO-RTU_512:9 PPO_512_frozen_5M:9 PPO-RTU_512_frozen_5M:9 Search-Oracle Search-Nearest 
# python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov5_rollout2048 --filter-alg-apertures PPO_2048:5 PPO-RTU_2048:5 PPO_2048_frozen_5M:5 PPO-RTU_2048_frozen_5M:5 Search-Oracle Search-Nearest 
# python src/learning_curve.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig_diwali_fov9_rollout2048 --filter-alg-apertures PPO_2048:9 PPO-RTU_2048:9 PPO_2048_frozen_5M:9 PPO-RTU_2048_frozen_5M:9 Search-Oracle Search-Nearest 

python src/biome_stacked_bar.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name fig1c_2biome_dqn_biome_occupancy --sort-seeds --filter-alg-apertures DQN:9 DQN_frozen_5M:9 ActorCriticMLP:9 ActorCriticMLP_frozen_5M:9 DQN:15 DQN_frozen_5M:15 ActorCriticMLP:15 ActorCriticMLP_frozen_5M:15 Search-Oracle --sample-types 999000:1000000:500 4999000:5000000:500 9999000:10000000:500

# python src/biome_ternary.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name afig1a --bars "DQN|9|999000:1000000:500|" "DQN_L2|9|999000:1000000:500|" "DQN_frozen_5M|9|999000:1000000:500|" "DQN_greedy_frozen_5M|9|999000:1000000:500|" "Search-Oracle||999000:1000000:500|" &
# python src/biome_ternary.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name afig1b --bars "DQN|9|4999000:5000000:500|" "DQN_L2|9|4999000:5000000:500|" "DQN_frozen_5M|9|4999000:5000000:500|" "DQN_greedy_frozen_5M|9|4999000:5000000:500|" "Search-Oracle||4999000:5000000:500|" &
# python src/biome_ternary.py experiments/E124-diwali/foragax/ForagaxDiwali-v4 --plot-name afig1c --bars "DQN|9|9999000:10000000:500|" "DQN_L2|9|9999000:10000000:500|" "DQN_frozen_5M|9|9999000:10000000:500|" "DQN_greedy_frozen_5M|9|9999000:10000000:500|" "Search-Oracle||9999000:10000000:500|" &
