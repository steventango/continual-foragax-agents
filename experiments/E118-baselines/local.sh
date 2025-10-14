python scripts/local.py --runs 30 --entry src/continuing_main.py -e experiments/E118-baselines/foragax/ForagaxWeather-v5/Baselines/*.json
python src/process_data.py experiments/E118-baselines/foragax/ForagaxWeather-v5
python src/learning_curve.py experiments/E118-baselines/foragax/ForagaxWeather-v5 --metrics ewm_reward temperatures_1 temperatures_2
